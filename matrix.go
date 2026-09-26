package portfolio

import "math"

// matVec computes A*x for a dense row-major matrix.
func matVec(A [][]float64, x []float64) []float64 {
	out := make([]float64, len(A))
	for i := range A {
		var s float64
		row := A[i]
		for j, v := range row {
			s += v * x[j]
		}
		out[i] = s
	}
	return out
}

// dot computes the inner product of two vectors.
func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// quadForm computes x'A x.
func quadForm(A [][]float64, x []float64) float64 {
	return dot(x, matVec(A, x))
}

// cloneVec returns a deep copy of v.
func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}

// cloneMat returns a deep copy of m.
func cloneMat(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = cloneVec(m[i])
	}
	return out
}

// symmetrize averages m with its transpose in place-safe manner and
// returns a new symmetric matrix.
func symmetrize(m [][]float64) [][]float64 {
	n := len(m)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			out[i][j] = 0.5 * (m[i][j] + m[j][i])
		}
	}
	return out
}

// jacobiEigen computes eigenvalues and eigenvectors of a real symmetric
// matrix using the classical cyclic Jacobi method. Deterministic.
// Returns eigenvalues (ascending) and eigenvectors as columns of V.
func jacobiEigen(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	A := cloneMat(a)
	V := make([][]float64, n)
	for i := range V {
		V[i] = make([]float64, n)
		V[i][i] = 1
	}
	const maxSweeps = 100
	const eps = 1e-15
	for sweep := 0; sweep < maxSweeps; sweep++ {
		var off float64
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += A[p][q] * A[p][q]
			}
		}
		if off < eps {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				if A[p][q] == 0 {
					continue
				}
				app, aqq, apq := A[p][p], A[q][q], A[p][q]
				theta := 0.5 * (aqq - app) / apq
				t := 1.0 / (absf(theta) + sqrt(1+theta*theta))
				if theta < 0 {
					t = -t
				}
				c := 1.0 / sqrt(1+t*t)
				s := t * c
				for k := 0; k < n; k++ {
					akp := A[k][p]
					akq := A[k][q]
					A[k][p] = c*akp - s*akq
					A[k][q] = s*akp + c*akq
				}
				for k := 0; k < n; k++ {
					apk := A[p][k]
					aqk := A[q][k]
					A[p][k] = c*apk - s*aqk
					A[q][k] = s*apk + c*aqk
				}
				for k := 0; k < n; k++ {
					vkp := V[k][p]
					vkq := V[k][q]
					V[k][p] = c*vkp - s*vkq
					V[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals = make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = A[i][i]
	}
	// Sort ascending with eigenvectors, simple insertion sort (deterministic).
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < n; i++ {
		for j := i; j > 0 && vals[idx[j-1]] > vals[idx[j]]; j-- {
			idx[j-1], idx[j] = idx[j], idx[j-1]
		}
	}
	sortedVals := make([]float64, n)
	sortedVecs := make([][]float64, n)
	for i := range sortedVecs {
		sortedVecs[i] = make([]float64, n)
	}
	for newPos, old := range idx {
		sortedVals[newPos] = vals[old]
		for k := 0; k < n; k++ {
			sortedVecs[k][newPos] = V[k][old]
		}
	}
	return sortedVals, sortedVecs
}

// nearestPSD clips eigenvalues of a symmetric matrix below floor to floor
// and rebuilds the matrix. Returns the rebuilt matrix and whether any
// clipping occurred.
func nearestPSD(m [][]float64, floor float64) ([][]float64, bool) {
	vals, vecs := jacobiEigen(m)
	clipped := false
	for i, v := range vals {
		if v < floor {
			vals[i] = floor
			clipped = true
		}
	}
	if !clipped {
		return m, false
	}
	n := len(m)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
		for j := range out[i] {
			var s float64
			for k := 0; k < n; k++ {
				s += vecs[i][k] * vals[k] * vecs[j][k]
			}
			out[i][j] = s
		}
	}
	return symmetrize(out), true
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func sqrt(x float64) float64 { return math.Sqrt(x) }
