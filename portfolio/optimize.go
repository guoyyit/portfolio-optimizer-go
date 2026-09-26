package portfolio

import (
	"math"
	"time"
)

// optimize.go: constrained mean-variance and risk-parity solvers.
//
// Method: (accelerated) projected gradient descent where the projection
// onto the intersection of all hard constraints is computed with Dykstra's
// alternating-projection algorithm. Every set projection uses a
// deterministic bisection, so the solver is deterministic: identical inputs
// always produce bitwise identical outputs (unless a wall-clock time budget
// interrupts the solve).

// solveBudget carries iteration/time limits.
type solveBudget struct {
	maxIters int
	deadline int64 // unix nano; 0 = no limit
}

// solveOut is the raw solver output.
type solveOut struct {
	w         []float64
	obj       float64
	iters     int
	converged bool
	feasible  bool
}

// projParams bundles everything the projection needs.
type projParams struct {
	lo, hi    []float64 // per-asset box bounds
	leverage  float64   // sum |w| cap; <=0 disables
	groups    []GroupCap
	turnover  float64   // sum |w - w0| cap; <=0 disables
	w0        []float64 // reference weights for turnover
	targetSum float64   // equality sum(w) = targetSum
}

const (
	projTol      = 1e-12
	projMaxCycle = 500
	bisectIters  = 200
	gradTol      = 1e-10
)

// projectBoxSum projects onto {lo <= w <= hi, sum w = s} exactly via
// bisection on the shift multiplier. Returns false if the set is empty.
func projectBoxSum(w []float64, lo, hi []float64, s float64) bool {
	n := len(w)
	var slo, shi float64
	for i := 0; i < n; i++ {
		slo += lo[i]
		shi += hi[i]
	}
	if slo > s+1e-12 || shi < s-1e-12 {
		return false
	}
	// Find nu with sum clip(w_i - nu, lo_i, hi_i) = s; the left side is
	// non-increasing in nu.
	minW, maxW := w[0], w[0]
	maxHi, minLo := hi[0], lo[0]
	for i := 0; i < n; i++ {
		if w[i] < minW {
			minW = w[i]
		}
		if w[i] > maxW {
			maxW = w[i]
		}
		if hi[i] > maxHi {
			maxHi = hi[i]
		}
		if lo[i] < minLo {
			minLo = lo[i]
		}
	}
	a := minW - maxHi - 1
	b := maxW - minLo + 1
	for it := 0; it < bisectIters; it++ {
		mid := 0.5 * (a + b)
		sum := 0.0
		for i := 0; i < n; i++ {
			v := w[i] - mid
			if v < lo[i] {
				v = lo[i]
			} else if v > hi[i] {
				v = hi[i]
			}
			sum += v
		}
		if sum > s {
			a = mid
		} else {
			b = mid
		}
	}
	nu := 0.5 * (a + b)
	for i := 0; i < n; i++ {
		v := w[i] - nu
		if v < lo[i] {
			v = lo[i]
		} else if v > hi[i] {
			v = hi[i]
		}
		w[i] = v
	}
	return true
}

// projectL1Ball projects onto {sum |w - c| <= r} around center c.
// Exact via soft-thresholding with a bisected threshold.
func projectL1Ball(w, c []float64, r float64) {
	if r < 0 {
		r = 0
	}
	n := len(w)
	d := make([]float64, n)
	sum := 0.0
	for i := 0; i < n; i++ {
		d[i] = w[i] - c[i]
		sum += math.Abs(d[i])
	}
	if sum <= r {
		return
	}
	maxAbs := 0.0
	for i := 0; i < n; i++ {
		if av := math.Abs(d[i]); av > maxAbs {
			maxAbs = av
		}
	}
	a, b := 0.0, maxAbs
	for it := 0; it < bisectIters; it++ {
		mid := 0.5 * (a + b)
		s := 0.0
		for i := 0; i < n; i++ {
			if v := math.Abs(d[i]) - mid; v > 0 {
				s += v
			}
		}
		if s > r {
			a = mid
		} else {
			b = mid
		}
	}
	tau := 0.5 * (a + b)
	for i := 0; i < n; i++ {
		av := math.Abs(d[i]) - tau
		if av < 0 {
			av = 0
		}
		w[i] = c[i] + math.Copysign(av, d[i])
	}
}

// projectGroupCap projects onto the halfspace {sum_{i in G} w_i <= max}.
func projectGroupCap(w []float64, g GroupCap) {
	if len(g.Members) == 0 {
		return
	}
	sum := 0.0
	for _, i := range g.Members {
		sum += w[i]
	}
	excess := sum - g.Max
	if excess <= 0 {
		return
	}
	shift := excess / float64(len(g.Members))
	for _, i := range g.Members {
		w[i] -= shift
	}
}

// dykstraProject projects x onto the intersection of all constraint sets
// using Dykstra's algorithm. Returns false when no convergence within the
// cycle budget, which indicates an (almost) empty intersection.
func dykstraProject(x []float64, p *projParams) ([]float64, bool) {
	n := len(x)
	nSets := 1 // box + sum
	if p.leverage > 0 {
		nSets++
	}
	if p.turnover > 0 && p.w0 != nil {
		nSets++
	}
	nSets += len(p.groups)

	// Build the fixed cyclic list of set projections.
	sets := make([]func(v []float64) bool, 0, nSets)
	sets = append(sets, func(v []float64) bool {
		return projectBoxSum(v, p.lo, p.hi, p.targetSum)
	})
	if p.leverage > 0 {
		zero := make([]float64, n)
		sets = append(sets, func(v []float64) bool {
			projectL1Ball(v, zero, p.leverage)
			return true
		})
	}
	if p.turnover > 0 && p.w0 != nil {
		sets = append(sets, func(v []float64) bool {
			projectL1Ball(v, p.w0, p.turnover)
			return true
		})
	}
	for _, g := range p.groups {
		g := g
		sets = append(sets, func(v []float64) bool {
			projectGroupCap(v, g)
			return true
		})
	}

	inc := make([][]float64, nSets)
	for i := range inc {
		inc[i] = make([]float64, n)
	}
	cur := make([]float64, n)
	copy(cur, x)
	prev := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)

	// project applies set `set`'s projection to cur (via y/z buffers).
	// It returns the increment change so convergence can be judged on the
	// full Dykstra state (point AND increments), not just the point.
	project := func(set int) (bool, float64) {
		for i := 0; i < n; i++ {
			y[i] = cur[i] + inc[set][i]
		}
		copy(z, y) // pre-projection point
		if !sets[set](y) {
			return false, 0
		}
		incChange := 0.0
		for i := 0; i < n; i++ {
			ni := z[i] - y[i] // p <- (x + p) - P(x + p)
			if d := math.Abs(ni - inc[set][i]); d > incChange {
				incChange = d
			}
			inc[set][i] = ni
		}
		copy(cur, y)
		return true, incChange
	}

	for cycle := 0; cycle < projMaxCycle; cycle++ {
		copy(prev, cur)
		maxIncChange := 0.0
		for set := 0; set < nSets; set++ {
			ok, incChange := project(set)
			if !ok {
				return nil, false
			}
			if incChange > maxIncChange {
				maxIncChange = incChange
			}
		}
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			if d := math.Abs(cur[i] - prev[i]); d > maxDiff {
				maxDiff = d
			}
		}
		// Converged only when the full state (point and increments) is
		// stable; a stable point with moving increments is a plateau, not
		// a fixed point.
		if maxDiff < projTol && maxIncChange < projTol {
			if feasiblePoint(cur, p, 1e-7) {
				return cur, true
			}
			// State stable but not feasible: keep iterating; if it stays
			// stuck the cycle budget will declare infeasibility.
		}
	}
	// Budget exhausted: accept only if the last point happens to be feasible.
	if feasiblePoint(cur, p, 1e-7) {
		return cur, true
	}
	return nil, false
}

// feasiblePoint verifies all hard constraints within tolerance tol.
func feasiblePoint(w []float64, p *projParams, tol float64) bool {
	sum, sumAbs := 0.0, 0.0
	for i, v := range w {
		if v < p.lo[i]-tol || v > p.hi[i]+tol {
			return false
		}
		sum += v
		sumAbs += math.Abs(v)
	}
	if math.Abs(sum-p.targetSum) > tol {
		return false
	}
	if p.leverage > 0 && sumAbs > p.leverage+tol {
		return false
	}
	if p.turnover > 0 && p.w0 != nil {
		to := 0.0
		for i, v := range w {
			to += math.Abs(v - p.w0[i])
		}
		if to > p.turnover+tol {
			return false
		}
	}
	for _, g := range p.groups {
		gs := 0.0
		for _, m := range g.Members {
			gs += w[m]
		}
		if gs > g.Max+tol {
			return false
		}
	}
	return true
}

// --- objectives ---------------------------------------------------------

// mvObjective returns f(w) = 0.5*a*w'Σw - mu'w (minimized).
func mvObjective(w, mu []float64, sigma *mat, a float64) float64 {
	sw := sigma.matVec(w)
	quad, lin := 0.0, 0.0
	for i := range w {
		quad += w[i] * sw[i]
		lin += mu[i] * w[i]
	}
	return 0.5*a*quad - lin
}

func mvGradient(w, mu []float64, sigma *mat, a float64, g []float64) {
	sw := sigma.matVec(w)
	for i := range w {
		g[i] = a*sw[i] - mu[i]
	}
}

// rpObjective returns the scale-free risk-parity residual
// f(w) = sum_i (RC_i - mean RC)^2 / (w'Σw)^2, RC_i = w_i (Σw)_i.
func rpObjective(w []float64, sigma *mat) float64 {
	sw := sigma.matVec(w)
	total := 0.0
	for i := range w {
		total += w[i] * sw[i]
	}
	if total <= 0 {
		return math.Inf(1)
	}
	target := total / float64(len(w))
	f := 0.0
	for i := range w {
		d := w[i]*sw[i] - target
		f += d * d
	}
	return f / (total * total)
}

// rpGradient computes the exact gradient of rpObjective.
//
//	f = Σ r_k² / T², r_k = w_k(Σw)_k − T/n, T = w'Σw.
//	Since Σ r_k = 0 the gradient simplifies to
//	∂f/∂w_i = 2/T² · (r_i(Σw)_i + (Σ(r∘w))_i) − 4f(Σw)_i/T.
func rpGradient(w []float64, sigma *mat, g []float64) {
	n := len(w)
	sw := sigma.matVec(w)
	total := 0.0
	for i := 0; i < n; i++ {
		total += w[i] * sw[i]
	}
	if total <= 0 {
		for i := range g {
			g[i] = 0
		}
		return
	}
	target := total / float64(n)
	rw := make([]float64, n)
	f := 0.0
	for k := 0; k < n; k++ {
		rk := w[k]*sw[k] - target
		f += rk * rk
		rw[k] = rk * w[k]
	}
	f /= total * total
	srw := sigma.matVec(rw)
	for i := 0; i < n; i++ {
		ri := w[i]*sw[i] - target
		g[i] = 2*(ri*sw[i]+srw[i])/(total*total) - 4*f*sw[i]/total
	}
}

// --- solvers ------------------------------------------------------------

// feasibleStart returns the projection of the equal-weight vector, or
// false when the constraint intersection is empty.
func feasibleStart(p *projParams, n int) ([]float64, bool) {
	w := make([]float64, n)
	for i := range w {
		w[i] = p.targetSum / float64(n)
	}
	return dykstraProject(w, p)
}

// solveMeanVariance solves min 0.5*a*w'Σw - mu'w subject to constraints
// via FISTA (accelerated projected gradient) with feasible iterates.
func solveMeanVariance(mu []float64, sigma *mat, riskAversion float64, p *projParams, budget solveBudget) solveOut {
	n := len(mu)
	if riskAversion <= 0 {
		riskAversion = 1
	}
	// Lipschitz constant of the gradient: a * lambda_max(Σ).
	eig := jacobiEigen(sigma)
	lmax := 0.0
	for _, v := range eig {
		if v > lmax {
			lmax = v
		}
	}
	L := riskAversion * lmax
	if L <= 0 {
		L = 1
	}
	step := 1.0 / L

	w, ok := feasibleStart(p, n)
	if !ok {
		return solveOut{feasible: false}
	}

	g := make([]float64, n)
	y := make([]float64, n)
	copy(y, w)
	wPrev := make([]float64, n)
	cand := make([]float64, n)
	tk := 1.0
	out := solveOut{feasible: true}
	fy := mvObjective(y, mu, sigma, riskAversion)

	for iter := 0; iter < budget.maxIters; iter++ {
		out.iters = iter + 1
		if budget.deadline > 0 && time.Now().UnixNano() > budget.deadline {
			break
		}
		mvGradient(y, mu, sigma, riskAversion, g)
		for i := 0; i < n; i++ {
			cand[i] = y[i] - step*g[i]
		}
		pw, ok := dykstraProject(cand, p)
		if !ok {
			return solveOut{feasible: false}
		}
		copy(wPrev, w)
		copy(w, pw)
		// FISTA momentum, re-projected to stay feasible.
		tNew := 0.5 * (1 + math.Sqrt(1+4*tk*tk))
		for i := 0; i < n; i++ {
			y[i] = w[i] + (tk-1)/tNew*(w[i]-wPrev[i])
		}
		tk = tNew
		py, ok := dykstraProject(y, p)
		if !ok {
			return solveOut{feasible: false}
		}
		copy(y, py)
		fyNew := mvObjective(y, mu, sigma, riskAversion)
		stepNorm := 0.0
		for i := 0; i < n; i++ {
			if d := math.Abs(w[i] - wPrev[i]); d > stepNorm {
				stepNorm = d
			}
		}
		if math.Abs(fyNew-fy) < gradTol*(1+math.Abs(fy)) && stepNorm < 1e-9 {
			out.converged = true
			break
		}
		fy = fyNew
	}
	out.w = w
	out.obj = -mvObjective(w, mu, sigma, riskAversion) // report utility
	return out
}

// solveRiskParity minimizes the risk-parity residual via projected gradient
// with deterministic Armijo backtracking.
func solveRiskParity(sigma *mat, p *projParams, budget solveBudget) solveOut {
	n := sigma.n
	w, ok := feasibleStart(p, n)
	if !ok {
		return solveOut{feasible: false}
	}

	g := make([]float64, n)
	out := solveOut{feasible: true}
	f := rpObjective(w, sigma)

	for iter := 0; iter < budget.maxIters; iter++ {
		out.iters = iter + 1
		if budget.deadline > 0 && time.Now().UnixNano() > budget.deadline {
			break
		}
		rpGradient(w, sigma, g)
		gNorm := 0.0
		for i := 0; i < n; i++ {
			if a := math.Abs(g[i]); a > gNorm {
				gNorm = a
			}
		}
		if gNorm < gradTol || f < gradTol*gradTol {
			out.converged = true
			break
		}
		// Backtracking line search with deterministic halving.
		step := 1.0
		accepted := false
		for ls := 0; ls < 60; ls++ {
			cand := make([]float64, n)
			for i := 0; i < n; i++ {
				cand[i] = w[i] - step*g[i]
			}
			pw, ok := dykstraProject(cand, p)
			if !ok {
				return solveOut{feasible: false}
			}
			fNew := rpObjective(pw, sigma)
			if fNew < f {
				copy(w, pw)
				f = fNew
				accepted = true
				break
			}
			step *= 0.5
		}
		if !accepted {
			out.converged = true // stationary: no improving step exists
			break
		}
	}
	out.w = w
	out.obj = -f // report negative residual (higher is better)
	return out
}
