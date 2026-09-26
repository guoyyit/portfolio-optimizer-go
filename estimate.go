package portfolio

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Estimator builds expected-return and covariance estimates from a return
// matrix. rows = observations (time), cols = assets. Missing values are
// represented as NaN (non-finite values are treated as missing too).
type Estimator interface {
	Estimate(returns [][]float64) (Estimates, error)
}

// NewSampleEstimator returns an estimator using plain sample statistics
// with pairwise-complete missing-data handling.
func NewSampleEstimator() Estimator { return &coreEstimator{shrinkage: 0} }

// NewShrinkageEstimator returns an estimator that shrinks the sample
// covariance toward its diagonal. shrinkage in [0,1] fixes the intensity;
// shrinkage < 0 selects an automatic Ledoit-Wolf style intensity.
func NewShrinkageEstimator(shrinkage float64) Estimator {
	return &coreEstimator{shrinkage: shrinkage}
}

// EstimateWith is a convenience wrapper estimating with explicit config.
func EstimateWith(returns [][]float64, est Estimator, cfg EstimationConfig) (Estimates, error) {
	if est == nil {
		est = NewShrinkageEstimator(-1)
	}
	if ce, ok := est.(*coreEstimator); ok {
		return ce.estimate(returns, cfg)
	}
	return est.Estimate(returns)
}

type coreEstimator struct{ shrinkage float64 }

func (e *coreEstimator) Estimate(returns [][]float64) (Estimates, error) {
	return e.estimate(returns, EstimationConfig{})
}

func (e *coreEstimator) estimate(returns [][]float64, cfg EstimationConfig) (Estimates, error) {
	var diags []Diagnostic
	// ---- validate shape ----
	if len(returns) == 0 {
		return Estimates{}, errors.New("portfolio: returns matrix is empty")
	}
	p := len(returns[0])
	if p == 0 {
		return Estimates{}, errors.New("portfolio: returns matrix has zero assets")
	}
	for i, row := range returns {
		if len(row) != p {
			return Estimates{}, fmt.Errorf("portfolio: row %d has %d columns, want %d", i, len(row), p)
		}
	}
	T := len(returns)
	if T < 2 {
		return Estimates{}, errors.New("portfolio: need at least 2 observations")
	}
	if p > T {
		diags = append(diags, Diagnostic{
			Flag:    DiagHighDimensional,
			Message: fmt.Sprintf("asset count %d exceeds sample count %d; sample covariance is singular, shrinkage/PSD repair applied", p, T),
		})
	}

	// ---- sanitize: treat non-finite as missing ----
	data := make([][]float64, T)
	missing := 0
	for t := 0; t < T; t++ {
		data[t] = make([]float64, p)
		for j := 0; j < p; j++ {
			v := returns[t][j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				data[t][j] = math.NaN()
				missing++
			} else {
				data[t][j] = v
			}
		}
	}
	if missing > 0 {
		diags = append(diags, Diagnostic{
			Flag:    DiagMissingData,
			Message: fmt.Sprintf("%d of %d observations missing/non-finite; pairwise-complete estimation used", missing, T*p),
		})
	}

	// ---- winsorize outliers per asset ----
	if cfg.WinsorizeLimit > 0 {
		clipped := winsorize(data, cfg.WinsorizeLimit)
		if clipped > 0 {
			diags = append(diags, Diagnostic{
				Flag:    DiagOutliersWinsorized,
				Message: fmt.Sprintf("%d observations clipped to median +/- %.3g*MAD", clipped, cfg.WinsorizeLimit),
			})
		}
	}

	// ---- per-asset means (pairwise means computed per pair below) ----
	mu := make([]float64, p)
	counts := make([]int, p)
	for j := 0; j < p; j++ {
		var sum float64
		for t := 0; t < T; t++ {
			if !math.IsNaN(data[t][j]) {
				sum += data[t][j]
				counts[j]++
			}
		}
		if counts[j] > 0 {
			mu[j] = sum / float64(counts[j])
		}
	}

	// ---- pairwise-complete sample covariance ----
	cov := make([][]float64, p)
	for i := range cov {
		cov[i] = make([]float64, p)
	}
	for i := 0; i < p; i++ {
		for j := i; j < p; j++ {
			var xs, ys []float64
			for t := 0; t < T; t++ {
				a, b := data[t][i], data[t][j]
				if !math.IsNaN(a) && !math.IsNaN(b) {
					xs = append(xs, a)
					ys = append(ys, b)
				}
			}
			k := len(xs)
			if k >= 2 {
				var mx, my float64
				for t := 0; t < k; t++ {
					mx += xs[t]
					my += ys[t]
				}
				mx /= float64(k)
				my /= float64(k)
				var s float64
				for t := 0; t < k; t++ {
					s += (xs[t] - mx) * (ys[t] - my)
				}
				c := s / float64(k-1)
				cov[i][j] = c
				cov[j][i] = c
			}
			// k < 2 handled by fallback below (left as zero for now)
		}
	}
	// Fallback for assets/pairs with insufficient data.
	var avgVar float64
	var nVar int
	for j := 0; j < p; j++ {
		if counts[j] >= 2 && cov[j][j] > 0 {
			avgVar += cov[j][j]
			nVar++
		}
	}
	if nVar > 0 {
		avgVar /= float64(nVar)
	} else {
		avgVar = 1.0 // no usable variance information at all
	}
	for j := 0; j < p; j++ {
		if counts[j] < 2 || cov[j][j] <= 0 {
			cov[j][j] = avgVar
		}
	}

	// ---- shrinkage toward diagonal target ----
	alpha := e.shrinkage
	if alpha < 0 {
		alpha = ledoitWolfIntensity(data, cov)
	}
	if alpha > 1 {
		alpha = 1
	}
	if alpha > 0 {
		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				if i != j {
					cov[i][j] *= 1 - alpha
				}
			}
		}
		diags = append(diags, Diagnostic{
			Flag:    DiagShrinkageApplied,
			Message: fmt.Sprintf("covariance shrunk toward diagonal with intensity %.4f", alpha),
		})
	}

	// ---- enforce symmetry + PSD, no NaN ----
	cov = symmetrize(cov)
	vals, _ := jacobiEigen(cov)
	lmax := vals[len(vals)-1]
	if lmax <= 0 {
		lmax = 1
	}
	if vals[0] < 1e-10*lmax {
		diags = append(diags, Diagnostic{
			Flag:    DiagNearSingular,
			Message: fmt.Sprintf("covariance near-singular: min/max eigenvalue ratio %.3g", vals[0]/lmax),
		})
	}
	floor := 1e-10 * math.Max(lmax, 1)
	cov, clipped := nearestPSD(cov, floor)
	if clipped {
		diags = append(diags, Diagnostic{
			Flag:    DiagEigenvalueClipped,
			Message: fmt.Sprintf("eigenvalues below %.3g clipped to enforce positive semi-definiteness", floor),
		})
	}
	sanitizeEstimates(mu, cov, avgVar, &diags)

	return Estimates{Mu: mu, Cov: cov, NSamples: T, Diagnostics: diags}, nil
}

// winsorize clips each asset's returns to median +- limit*MAD in place.
// Returns the number of clipped observations.
func winsorize(data [][]float64, limit float64) int {
	T := len(data)
	p := len(data[0])
	clipped := 0
	for j := 0; j < p; j++ {
		var col []float64
		for t := 0; t < T; t++ {
			if !math.IsNaN(data[t][j]) {
				col = append(col, data[t][j])
			}
		}
		if len(col) < 3 {
			continue
		}
		med := median(col)
		dev := make([]float64, len(col))
		for i, v := range col {
			dev[i] = math.Abs(v - med)
		}
		mad := median(dev)
		if mad <= 0 {
			continue
		}
		lo, hi := med-limit*mad, med+limit*mad
		for t := 0; t < T; t++ {
			v := data[t][j]
			if math.IsNaN(v) {
				continue
			}
			if v < lo {
				data[t][j] = lo
				clipped++
			} else if v > hi {
				data[t][j] = hi
				clipped++
			}
		}
	}
	return clipped
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return 0.5 * (s[n/2-1] + s[n/2])
}

// ledoitWolfIntensity estimates the optimal shrinkage intensity toward the
// scaled identity using complete observations only (Ledoit & Wolf 2004).
func ledoitWolfIntensity(data [][]float64, S [][]float64) float64 {
	p := len(S)
	// gather complete rows
	var rows [][]float64
	for t := range data {
		complete := true
		for j := 0; j < p; j++ {
			if math.IsNaN(data[t][j]) {
				complete = false
				break
			}
		}
		if complete {
			rows = append(rows, data[t])
		}
	}
	T := len(rows)
	if T < 2 {
		return 1 // not enough data: fully shrink to the safe diagonal target
	}
	// demean
	mu := make([]float64, p)
	for _, r := range rows {
		for j := 0; j < p; j++ {
			mu[j] += r[j]
		}
	}
	for j := range mu {
		mu[j] /= float64(T)
	}
	// muTarget = trace(S)/p ; d^2 = ||S - muTarget*I||_F^2
	var trace float64
	for j := 0; j < p; j++ {
		trace += S[j][j]
	}
	muTarget := trace / float64(p)
	var d2 float64
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			d := S[i][j]
			if i == j {
				d -= muTarget
			}
			d2 += d * d
		}
	}
	if d2 <= 0 {
		return 0
	}
	// b^2 = (1/T^2) * sum_t || x_t x_t' - S ||_F^2, capped by d^2
	x := make([]float64, p)
	var b2 float64
	for _, r := range rows {
		for j := 0; j < p; j++ {
			x[j] = r[j] - mu[j]
		}
		var norm float64
		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				d := x[i]*x[j] - S[i][j]
				norm += d * d
			}
		}
		b2 += norm
	}
	b2 /= float64(T) * float64(T)
	if b2 > d2 {
		b2 = d2
	}
	if b2 < 0 {
		b2 = 0
	}
	return b2 / d2
}

// sanitizeEstimates is a defensive final pass: no NaN/Inf may escape.
func sanitizeEstimates(mu []float64, cov [][]float64, fallbackVar float64, diags *[]Diagnostic) {
	fixed := false
	for j := range mu {
		if math.IsNaN(mu[j]) || math.IsInf(mu[j], 0) {
			mu[j] = 0
			fixed = true
		}
	}
	for i := range cov {
		for j := range cov[i] {
			if math.IsNaN(cov[i][j]) || math.IsInf(cov[i][j], 0) {
				if i == j {
					cov[i][j] = fallbackVar
				} else {
					cov[i][j] = 0
				}
				fixed = true
			}
		}
	}
	if fixed {
		*diags = append(*diags, Diagnostic{
			Flag:    DiagMissingData,
			Message: "non-finite estimates replaced with safe fallbacks",
		})
	}
}
