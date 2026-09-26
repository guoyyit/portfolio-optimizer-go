package portfolio

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

// Optimize builds a portfolio from precomputed estimates subject to hard
// constraints. Returned weights satisfy all hard constraints (to 1e-8)
// whenever Status is Optimal, Suboptimal or Timeout. Infeasible constraint
// sets yield StatusInfeasible; malformed inputs yield StatusInvalidInput
// plus a non-nil error. The solver is fully deterministic.
func Optimize(ctx context.Context, est Estimates, c *Constraints, opts Options) (Result, error) {
	n := len(est.Mu)
	if n == 0 {
		return Result{Status: StatusInvalidInput, Message: "empty estimates"}, errors.New("portfolio: empty estimates")
	}
	if len(est.Cov) != n {
		return Result{Status: StatusInvalidInput, Message: "covariance dimension mismatch"},
			fmt.Errorf("portfolio: cov is %d x ?, mu is %d", len(est.Cov), n)
	}
	for i := range est.Cov {
		if len(est.Cov[i]) != n {
			return Result{Status: StatusInvalidInput, Message: "covariance not square"},
				fmt.Errorf("portfolio: cov row %d has %d cols, want %d", i, len(est.Cov[i]), n)
		}
		for _, v := range est.Cov[i] {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return Result{Status: StatusInvalidInput, Message: "covariance contains non-finite values"},
					errors.New("portfolio: non-finite covariance")
			}
		}
	}
	for _, v := range est.Mu {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return Result{Status: StatusInvalidInput, Message: "mu contains non-finite values"},
				errors.New("portfolio: non-finite mu")
		}
	}
	if cerr := validateConstraints(c, n); cerr != nil {
		st := StatusInvalidInput
		if cerr.infeasible {
			st = StatusInfeasible
		}
		return Result{Status: st, Message: cerr.msg}, nil
	}

	// objective + gradient
	riskAversion := opts.RiskAversion
	if riskAversion <= 0 {
		riskAversion = 1
	}
	var f func(w []float64) float64
	var grad func(w []float64, out []float64)
	switch opts.Objective {
	case ObjectiveRiskParity:
		f = riskParityObjective(est.Cov)
		grad = riskParityGradient(est.Cov)
	default: // ObjectiveMeanVariance
		mu := est.Mu
		cov := est.Cov
		ra := riskAversion
		f = func(w []float64) float64 {
			return -dot(mu, w) + 0.5*ra*quadForm(cov, w)
		}
		grad = func(w []float64, out []float64) {
			sw := matVec(cov, w)
			for i := range out {
				out[i] = -mu[i] + ra*sw[i]
			}
		}
	}

	maxIter := opts.MaxIterations
	if maxIter <= 0 {
		maxIter = 2000
	}
	tol := opts.Tolerance
	if tol <= 0 {
		tol = 1e-9
	}
	projIters := opts.ProjectionIters

	// initial feasible point: equal weight under budget, projected
	r := resolve(c, n)
	w0 := make([]float64, n)
	for i := range w0 {
		w0[i] = r.budget / float64(n)
	}
	w, ok := project(w0, c, n, max(projIters, 2000))
	if !ok {
		return Result{
			Status:  StatusInfeasible,
			Message: "could not find a feasible point for the constraint set",
		}, nil
	}

	// projected gradient descent with deterministic backtracking
	var deadline time.Time
	if opts.TimeBudget > 0 {
		deadline = time.Now().Add(opts.TimeBudget)
	}
	status := StatusSuboptimal
	message := "iteration budget exhausted; returning best feasible point"
	step := 1.0
	g := make([]float64, n)
	fw := f(w)
	iters := 0
	timedOut := false
	for it := 0; it < maxIter; it++ {
		iters = it + 1
		if !deadline.IsZero() && time.Now().After(deadline) {
			timedOut = true
			break
		}
		if err := ctx.Err(); err != nil {
			timedOut = true
			break
		}
		grad(w, g)
		accepted := false
		t := step
		var wNew []float64
		var fNew float64
		for bt := 0; bt < 60; bt++ {
			cand := make([]float64, n)
			for i := range cand {
				cand[i] = w[i] - t*g[i]
			}
			proj, ok := project(cand, c, n, projIters)
			if !ok {
				t *= 0.5
				continue
			}
			fc := f(proj)
			// Armijo sufficient decrease along projected arc
			var d2 float64
			for i := range w {
				d := proj[i] - w[i]
				d2 += d * d
			}
			if fc <= fw-1e-4*d2/t || d2 == 0 {
				wNew, fNew = proj, fc
				accepted = true
				break
			}
			t *= 0.5
		}
		if !accepted {
			// no productive direction: converged (KKT-ish for projection)
			status = StatusOptimal
			message = "converged: no feasible descent direction"
			break
		}
		var maxDiff float64
		for i := range w {
			if d := math.Abs(wNew[i] - w[i]); d > maxDiff {
				maxDiff = d
			}
		}
		w, fw = wNew, fNew
		step = t * 1.5 // grow step cautiously for next iteration
		if maxDiff < tol {
			status = StatusOptimal
			message = "converged within tolerance"
			break
		}
	}
	if timedOut {
		status = StatusTimeout
		message = "time budget exhausted; returning best feasible point"
	}

	// final feasibility enforcement (hard guarantee)
	w, ok = project(w, c, n, max(projIters, 5000))
	if !ok || !satisfies(w, c, n, 1e-8) {
		return Result{
			Status:  StatusInfeasible,
			Message: "solver could not produce weights satisfying all hard constraints",
		}, nil
	}
	fw = f(w)

	diags := append([]Diagnostic(nil), est.Diagnostics...)
	if r.turnover > 0 {
		var to float64
		for i := range w {
			to += math.Abs(w[i] - r.current[i])
		}
		if to > r.turnover-1e-6 {
			diags = append(diags, Diagnostic{
				Flag:    DiagTurnoverCapped,
				Message: fmt.Sprintf("turnover constraint active at %.6f (cap %.6f)", to, r.turnover),
			})
		}
	}

	return Result{
		Status:      status,
		Weights:     w,
		Metrics:     computeMetrics(est.Mu, est.Cov, w, r.current),
		Diagnostics: diags,
		Iterations:  iters,
		Objective:   fw,
		Message:     message,
	}, nil
}

// riskParityObjective: sum_i (a_i - mean(a))^2 with a_i = w_i*(Cw)_i.
func riskParityObjective(cov [][]float64) func(w []float64) float64 {
	return func(w []float64) float64 {
		sw := matVec(cov, w)
		n := len(w)
		var T float64
		for i := range w {
			T += w[i] * sw[i]
		}
		if T <= 0 {
			return math.Inf(1)
		}
		b := T / float64(n)
		var g float64
		for i := range w {
			d := w[i]*sw[i] - b
			g += d * d
		}
		return g
	}
}

// riskParityGradient: d/dw_k g = 2*( r_k*s_k + (C'(r.*w))_k ), r_i =
// w_i*s_i - T/n, since sum(r) = 0 eliminates the remaining term.
func riskParityGradient(cov [][]float64) func(w, out []float64) {
	return func(w, out []float64) {
		n := len(w)
		sw := matVec(cov, w)
		var T float64
		for i := range w {
			T += w[i] * sw[i]
		}
		if T <= 0 {
			for i := range out {
				out[i] = 0
			}
			return
		}
		b := T / float64(n)
		rw := make([]float64, n)
		for i := range w {
			rw[i] = (w[i]*sw[i] - b) * w[i]
		}
		crw := matVec(cov, rw) // C is symmetric: C'(r.*w) = C(r.*w)
		for k := range out {
			rk := w[k]*sw[k] - b
			out[k] = 2 * (rk*sw[k] + crw[k])
		}
	}
}

// computeMetrics summarizes a feasible portfolio.
func computeMetrics(mu []float64, cov [][]float64, w, current []float64) RiskMetrics {
	n := len(w)
	sw := matVec(cov, w)
	variance := dot(w, sw)
	if variance < 0 {
		variance = 0
	}
	rc := make([]float64, n)
	if variance > 0 {
		for i := range w {
			rc[i] = w[i] * sw[i] / variance
		}
	} else {
		for i := range w {
			rc[i] = 1 / float64(n)
		}
	}
	var hhi, rc2 float64
	for i := range w {
		hhi += w[i] * w[i]
		rc2 += rc[i] * rc[i]
	}
	eb := 0.0
	if rc2 > 0 {
		eb = 1 / rc2
	}
	var to float64
	if len(current) == n {
		for i := range w {
			to += math.Abs(w[i] - current[i])
		}
	}
	return RiskMetrics{
		ExpectedReturn:    dot(mu, w),
		Volatility:        math.Sqrt(variance),
		Variance:          variance,
		RiskContributions: rc,
		Herfindahl:        hhi,
		EffectiveBets:     eb,
		Turnover:          to,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
