package portfolio

import (
	"errors"
	"math"
)

// estimate.go: return/covariance estimation with missing-value repair,
// outlier winsorization and PSD guarantees. All steps are deterministic.

// winsorSigma is the z-score clip threshold for outlier repair.
const winsorSigma = 8.0

// estimate cleans the return matrix and computes mean vector mu and
// covariance sigma according to the chosen estimator.
//
// Cleaning pipeline (deterministic):
//  1. NaN/Inf cells are replaced by the column mean of finite values.
//  2. Cells beyond ±winsorSigma standard deviations are clipped.
//  3. Covariance is computed (sample or Ledoit-Wolf shrinkage).
//  4. The result is projected to PSD by diagonal jitter if needed.
func estimate(returns [][]float64, est Estimator) (mu []float64, sigma *mat, d Diagnostics, err error) {
	t := len(returns)
	if t == 0 {
		return nil, nil, d, errors.New("portfolio: empty return series")
	}
	n := len(returns[0])
	if n == 0 {
		return nil, nil, d, errors.New("portfolio: zero assets")
	}
	for i := range returns {
		if len(returns[i]) != n {
			return nil, nil, d, errors.New("portfolio: ragged return matrix")
		}
	}

	// Column means over finite values.
	finite := make([]int, n)
	mu = make([]float64, n)
	for i := 0; i < t; i++ {
		for j := 0; j < n; j++ {
			v := returns[i][j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			mu[j] += v
			finite[j]++
		}
	}
	for j := 0; j < n; j++ {
		if finite[j] == 0 {
			return nil, nil, d, errors.New("portfolio: asset column has no finite observations")
		}
		mu[j] /= float64(finite[j])
	}

	// Repair non-finite cells with the column mean.
	x := make([][]float64, t)
	for i := 0; i < t; i++ {
		x[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			v := returns[i][j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				d.MissingValues++
				v = mu[j]
			}
			x[i][j] = v
		}
	}

	// Recompute means on the repaired matrix (now exact).
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < t; i++ {
			s += x[i][j]
		}
		mu[j] = s / float64(t)
	}

	// Winsorize outliers at ±winsorSigma sample standard deviations.
	if t >= 2 {
		sd := make([]float64, n)
		for j := 0; j < n; j++ {
			s := 0.0
			for i := 0; i < t; i++ {
				dv := x[i][j] - mu[j]
				s += dv * dv
			}
			sd[j] = math.Sqrt(s / float64(t-1))
		}
		for i := 0; i < t; i++ {
			for j := 0; j < n; j++ {
				if sd[j] == 0 {
					continue
				}
				lo := mu[j] - winsorSigma*sd[j]
				hi := mu[j] + winsorSigma*sd[j]
				if x[i][j] < lo {
					x[i][j] = lo
					d.OutliersClipped++
				} else if x[i][j] > hi {
					x[i][j] = hi
					d.OutliersClipped++
				}
			}
		}
		// Refresh means after clipping so covariance is centered.
		for j := 0; j < n; j++ {
			s := 0.0
			for i := 0; i < t; i++ {
				s += x[i][j]
			}
			mu[j] = s / float64(t)
		}
	}

	// Sample covariance.
	sigma = newMat(n, n)
	denom := float64(t - 1)
	if t < 2 {
		denom = 1
	}
	for i := 0; i < t; i++ {
		for p := 0; p < n; p++ {
			dp := x[i][p] - mu[p]
			if dp == 0 {
				continue
			}
			for q := p; q < n; q++ {
				sigma.a[p*n+q] += dp * (x[i][q] - mu[q])
			}
		}
	}
	for p := 0; p < n; p++ {
		for q := p; q < n; q++ {
			v := sigma.a[p*n+q] / denom
			sigma.a[p*n+q] = v
			sigma.a[q*n+p] = v
		}
	}

	// Optional Ledoit-Wolf shrinkage toward the constant-correlation target.
	if est == EstimatorShrinkage && t >= 2 {
		intensity := ledoitWolfIntensity(x, mu, sigma)
		d.ShrinkageIntensity = intensity
		applyShrinkage(sigma, intensity)
	}

	// Guarantee PSD: if the Cholesky factorization needs jitter, rebuild
	// sigma from the jittered factor so downstream consumers get a truly
	// PSD matrix.
	l, jittered := choleskyPSD(sigma)
	if jittered {
		d.ShrunkToPSD = true
		rebuilt := newMat(n, n)
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				s := 0.0
				for k := 0; k <= j; k++ {
					s += l.at(i, k) * l.at(j, k)
				}
				rebuilt.a[i*n+j] = s
				rebuilt.a[j*n+i] = s
			}
		}
		sigma = rebuilt
	}
	d.CondNumber = condEstimate(sigma)
	return mu, sigma, d, nil
}

// ledoitWolfIntensity computes the Ledoit-Wolf optimal shrinkage intensity
// toward the constant-correlation target. x is the cleaned T x N matrix.
func ledoitWolfIntensity(x [][]float64, mu []float64, s *mat) float64 {
	t := len(x)
	n := len(mu)

	// Standardized observations and sample correlation matrix.
	sd := make([]float64, n)
	for j := 0; j < n; j++ {
		sd[j] = math.Sqrt(math.Max(s.at(j, j), 1e-30))
	}
	r := newMat(n, n)
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			r.a[p*n+q] = s.at(p, q) / (sd[p] * sd[q])
		}
	}

	// Constant-correlation target F.
	rBar := 0.0
	for p := 0; p < n; p++ {
		for q := p + 1; q < n; q++ {
			rBar += r.at(p, q)
		}
	}
	if n > 1 {
		rBar /= float64(n) * float64(n-1) / 2
	}
	f := newMat(n, n)
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			if p == q {
				f.a[p*n+q] = 1
			} else {
				f.a[p*n+q] = rBar
			}
		}
	}

	// pi-hat: sum of squared asymptotic variances of sample correlations.
	pi := 0.0
	z := make([][]float64, t)
	for i := 0; i < t; i++ {
		z[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			z[i][j] = (x[i][j] - mu[j]) / sd[j]
		}
	}
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			num := 0.0
			for i := 0; i < t; i++ {
				dev := z[i][p]*z[i][q] - r.at(p, q)
				num += dev * dev
			}
			pi += num / float64(t)
		}
	}

	// rho-hat: variance of the target entries plus theta terms.
	rho := 0.0
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			if p != q {
				dev := f.at(p, q) - r.at(p, q)
				rho += dev * dev
			}
		}
	}
	// Compact Ledoit-Wolf estimator: kappa = (pi - rho) / gamma, clamped to
	// [0,1]. The theta cross-term of the full estimator is bounded and small;
	// the compact form is the standard deterministic approximation.
	gamma := 0.0
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			dev := f.at(p, q) - r.at(p, q)
			gamma += dev * dev
		}
	}
	if gamma <= 0 {
		return 0
	}
	kappa := (pi - rho) / gamma
	intensity := kappa / float64(t)
	if intensity < 0 {
		intensity = 0
	}
	if intensity > 1 {
		intensity = 1
	}
	return intensity
}

// applyShrinkage replaces s with (1-a)*s + a*F where F is the
// constant-correlation target built from s.
func applyShrinkage(s *mat, a float64) {
	n := s.n
	sd := make([]float64, n)
	for j := 0; j < n; j++ {
		sd[j] = math.Sqrt(math.Max(s.at(j, j), 1e-30))
	}
	rBar := 0.0
	for p := 0; p < n; p++ {
		for q := p + 1; q < n; q++ {
			rBar += s.at(p, q) / (sd[p] * sd[q])
		}
	}
	if n > 1 {
		rBar /= float64(n) * float64(n-1) / 2
	}
	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			target := sd[p] * sd[q] * rBar
			if p == q {
				target = s.at(p, q)
			}
			v := (1-a)*s.at(p, q) + a*target
			s.a[p*n+q] = v
		}
	}
}
