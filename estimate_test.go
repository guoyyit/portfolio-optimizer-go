package portfolio

import (
	"math"
	"testing"
)

func makeReturns(T, p int, f func(t, j int) float64) [][]float64 {
	r := make([][]float64, T)
	for t := range r {
		r[t] = make([]float64, p)
		for j := range r[t] {
			r[t][j] = f(t, j)
		}
	}
	return r
}

func TestSampleEstimatorBasic(t *testing.T) {
	// deterministic pseudo-random-ish data
	r := makeReturns(200, 3, func(t, j int) float64 {
		return 0.001*float64(j) + 0.01*math.Sin(float64(t)*0.7+float64(j))
	})
	est, err := NewSampleEstimator().Estimate(r)
	if err != nil {
		t.Fatal(err)
	}
	if len(est.Mu) != 3 || len(est.Cov) != 3 {
		t.Fatalf("bad dims")
	}
	for i := 0; i < 3; i++ {
		if est.Cov[i][i] <= 0 {
			t.Fatalf("non-positive variance %v", est.Cov[i][i])
		}
		for j := 0; j < 3; j++ {
			if math.Abs(est.Cov[i][j]-est.Cov[j][i]) > 1e-12 {
				t.Fatalf("asymmetric cov")
			}
		}
	}
	vals, _ := jacobiEigen(est.Cov)
	if vals[0] < 0 {
		t.Fatalf("not PSD: min eigenvalue %v", vals[0])
	}
}

func TestEstimatorMissingAndHighDim(t *testing.T) {
	// p > T with missing values: must not crash, must stay PSD & finite.
	r := makeReturns(5, 8, func(t, j int) float64 {
		if (t+j)%7 == 0 {
			return math.NaN()
		}
		return 0.02 * math.Sin(float64(t*j+1))
	})
	est, err := EstimateWith(r, NewShrinkageEstimator(-1), EstimationConfig{})
	if err != nil {
		t.Fatal(err)
	}
	flags := map[DiagFlag]bool{}
	for _, d := range est.Diagnostics {
		flags[d.Flag] = true
	}
	if !flags[DiagMissingData] {
		t.Errorf("expected missing-data diagnostic, got %v", est.Diagnostics)
	}
	if !flags[DiagHighDimensional] {
		t.Errorf("expected high-dimensional diagnostic")
	}
	for i := range est.Cov {
		for j := range est.Cov[i] {
			if math.IsNaN(est.Cov[i][j]) || math.IsInf(est.Cov[i][j], 0) {
				t.Fatalf("non-finite covariance entry")
			}
		}
	}
	vals, _ := jacobiEigen(est.Cov)
	if vals[0] < -1e-12 {
		t.Fatalf("not PSD: %v", vals[0])
	}
}

func TestWinsorizeClipsOutliers(t *testing.T) {
	r := makeReturns(100, 2, func(t, j int) float64 {
		v := 0.01 * math.Sin(float64(t))
		if t == 50 && j == 0 {
			v = 100 // massive outlier
		}
		return v
	})
	est, err := EstimateWith(r, NewSampleEstimator(), EstimationConfig{WinsorizeLimit: 3})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, d := range est.Diagnostics {
		if d.Flag == DiagOutliersWinsorized {
			found = true
		}
	}
	if !found {
		t.Errorf("expected winsorize diagnostic")
	}
	if est.Cov[0][0] > 1 {
		t.Errorf("outlier leaked into variance: %v", est.Cov[0][0])
	}
}

func TestConstantReturnsNoNaN(t *testing.T) {
	r := makeReturns(30, 4, func(t, j int) float64 { return 0.001 })
	est, err := EstimateWith(r, NewShrinkageEstimator(-1), EstimationConfig{})
	if err != nil {
		t.Fatal(err)
	}
	vals, _ := jacobiEigen(est.Cov)
	if vals[0] <= 0 {
		t.Errorf("expected strictly positive floor, got %v", vals[0])
	}
}
