package portfolio

import "math"

// matrix.go: minimal dense linear algebra helpers (stdlib only).
// All routines are deterministic.

// mat is a row-major dense matrix.
type mat struct {
	n, m int
	a    []float64
}

func newMat(n, m int) *mat { return &mat{n: n, m: m, a: make([]float64, n*m)} }

func (x *mat) at(i, j int) float64     { return x.a[i*x.m+j] }
func (x *mat) set(i, j int, v float64) { x.a[i*x.m+j] = v }

// matVec computes A v for a square matrix.
func (x *mat) matVec(v []float64) []float64 {
	out := make([]float64, x.n)
	for i := 0; i < x.n; i++ {
		s := 0.0
		base := i * x.m
		for j := 0; j < x.m; j++ {
			s += x.a[base+j] * v[j]
		}
		out[i] = s
	}
	return out
}

// clone returns a deep copy.
func (x *mat) clone() *mat {
	y := newMat(x.n, x.m)
	copy(y.a, x.a)
	return y
}

// choleskyPSD computes L with A ≈ L L' for a symmetric PSD matrix.
// If A is not numerically PSD, escalating diagonal jitter is added until
// the factorization succeeds; jittered reports whether that happened.
func choleskyPSD(a *mat) (l *mat, jittered bool) {
	n := a.n
	// Scale of the diagonal for jitter sizing.
	scale := 0.0
	for i := 0; i < n; i++ {
		if d := math.Abs(a.at(i, i)); d > scale {
			scale = d
		}
	}
	if scale == 0 {
		scale = 1
	}
	jitter := 0.0
	for attempt := 0; attempt < 12; attempt++ {
		l = newMat(n, n)
		ok := true
		for i := 0; i < n && ok; i++ {
			for j := 0; j <= i; j++ {
				s := a.at(i, j)
				if i == j {
					s += jitter
				}
				for k := 0; k < j; k++ {
					s -= l.at(i, k) * l.at(j, k)
				}
				if i == j {
					if s <= 0 || math.IsNaN(s) || math.IsInf(s, 0) {
						ok = false
						break
					}
					l.set(i, j, math.Sqrt(s))
				} else {
					d := l.at(j, j)
					if d == 0 {
						ok = false
						break
					}
					l.set(i, j, s/d)
				}
			}
		}
		if ok {
			return l, jitter > 0
		}
		if jitter == 0 {
			jitter = 1e-10 * scale
		} else {
			jitter *= 10
		}
	}
	// Fall back to a diagonal matrix with the (clamped) diagonal of A.
	l = newMat(n, n)
	for i := 0; i < n; i++ {
		d := a.at(i, i)
		if d <= 0 || math.IsNaN(d) || math.IsInf(d, 0) {
			d = 1e-8 * scale
		}
		l.set(i, i, math.Sqrt(d))
	}
	return l, true
}

// jacobiEigen computes all eigenvalues of a symmetric matrix via cyclic
// Jacobi rotations. Deterministic (fixed sweep order, fixed sweep count).
func jacobiEigen(a *mat) []float64 {
	n := a.n
	d := a.clone()
	const sweeps = 30
	for sweep := 0; sweep < sweeps; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += d.at(p, q) * d.at(p, q)
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				apq := d.at(p, q)
				if apq == 0 {
					continue
				}
				app := d.at(p, p)
				aqq := d.at(q, q)
				theta := (aqq - app) / (2 * apq)
				t := math.Copysign(1/(math.Abs(theta)+math.Sqrt(theta*theta+1)), theta)
				if theta == 0 {
					t = 1
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := 0; k < n; k++ {
					dkp := d.at(k, p)
					dkq := d.at(k, q)
					d.set(k, p, c*dkp-s*dkq)
					d.set(k, q, s*dkp+c*dkq)
				}
				for k := 0; k < n; k++ {
					dpk := d.at(p, k)
					dqk := d.at(q, k)
					d.set(p, k, c*dpk-s*dqk)
					d.set(q, k, s*dpk+c*dqk)
				}
			}
		}
	}
	eig := make([]float64, n)
	for i := 0; i < n; i++ {
		eig[i] = d.at(i, i)
	}
	return eig
}

// condEstimate estimates the 2-norm condition number of a symmetric
// positive semidefinite matrix: |λmax| / max(|λmin|, tiny).
func condEstimate(a *mat) float64 {
	eig := jacobiEigen(a)
	maxAbs, minAbs := 0.0, math.Inf(1)
	for _, v := range eig {
		av := math.Abs(v)
		if av > maxAbs {
			maxAbs = av
		}
		if av < minAbs {
			minAbs = av
		}
	}
	if minAbs < 1e-30 {
		minAbs = 1e-30
	}
	return maxAbs / minAbs
}
