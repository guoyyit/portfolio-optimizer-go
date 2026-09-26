package portfolio

import (
	"math"
	"testing"
)

// psdCheck verifies sigma is symmetric positive semidefinite via Cholesky
// without requiring jitter.
func psdCheck(t *testing.T, sigma *mat) {
	t.Helper()
	if _, jittered := choleskyPSD(sigma); jittered {
		t.Fatal("covariance is not PSD without jitter")
	}
	for i := 0; i < sigma.n; i++ {
		for j := 0; j < sigma.n; j++ {
			if math.IsNaN(sigma.at(i, j)) || math.IsInf(sigma.at(i, j), 0) {
				t.Fatalf("non-finite covariance entry at (%d,%d)", i, j)
			}
		}
	}
}

func TestSampleCovariancePSD(t *testing.T) {
	r := mkReturns(200, 5, 41)
	_, sigma, d, err := estimate(r, EstimatorSample)
	if err != nil {
		t.Fatal(err)
	}
	psdCheck(t, sigma)
	if d.CondNumber <= 0 || math.IsNaN(d.CondNumber) {
		t.Fatalf("bad condition number: %v", d.CondNumber)
	}
}

func TestMoreAssetsThanSamples(t *testing.T) {
	// N=10 assets, T=8 samples: sample covariance is singular.
	r := mkReturns(8, 10, 43)
	_, sigma, d, err := estimate(r, EstimatorSample)
	if err != nil {
		t.Fatal(err)
	}
	psdCheck(t, sigma) // must be repaired to PSD
	if !d.ShrunkToPSD {
		t.Fatal("expected ShrunkToPSD diagnostic for singular covariance")
	}
	// Shrinkage estimator should also stay PSD.
	_, sigma2, d2, err := estimate(r, EstimatorShrinkage)
	if err != nil {
		t.Fatal(err)
	}
	psdCheck(t, sigma2)
	if d2.ShrinkageIntensity < 0 || d2.ShrinkageIntensity > 1 {
		t.Fatalf("shrinkage intensity out of [0,1]: %v", d2.ShrinkageIntensity)
	}
	// Full pipeline must still produce a valid portfolio.
	res := ConstructOnce(Problem{
		Returns:   r,
		Objective: ObjectiveRiskParity,
		Estimator: EstimatorShrinkage,
	})
	if res.Status != StatusOptimal && res.Status != StatusFeasibleSub {
		t.Fatalf("status=%v", res.Status)
	}
	checkHardConstraints(t, res, Constraints{}, 10)
}

func TestMissingValuesRepaired(t *testing.T) {
	r := mkReturns(100, 4, 47)
	r[3][1] = math.NaN()
	r[10][2] = math.Inf(1)
	r[55][0] = math.Inf(-1)
	_, sigma, d, err := estimate(r, EstimatorSample)
	if err != nil {
		t.Fatal(err)
	}
	if d.MissingValues != 3 {
		t.Fatalf("expected 3 missing values repaired, got %d", d.MissingValues)
	}
	psdCheck(t, sigma)
	// Full pipeline: no NaN anywhere.
	res := ConstructOnce(Problem{Returns: r, Objective: ObjectiveMeanVariance})
	if res.Status != StatusOptimal {
		t.Fatalf("status=%v", res.Status)
	}
	for i, w := range res.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("non-finite weight at %d", i)
		}
	}
	if res.Diagnostics.MissingValues != 3 {
		t.Fatalf("diagnostics lost missing-value count: %d", res.Diagnostics.MissingValues)
	}
}

func TestOutliersClipped(t *testing.T) {
	r := mkReturns(200, 3, 53)
	r[100][0] = 5.0 // massive outlier (~250 sigma)
	_, _, d, err := estimate(r, EstimatorSample)
	if err != nil {
		t.Fatal(err)
	}
	if d.OutliersClipped == 0 {
		t.Fatal("expected outlier to be clipped")
	}
}

func TestColumnAllMissing(t *testing.T) {
	r := mkReturns(50, 3, 59)
	for i := range r {
		r[i][1] = math.NaN()
	}
	_, _, _, err := estimate(r, EstimatorSample)
	if err == nil {
		t.Fatal("expected error for column with no finite observations")
	}
	res := ConstructOnce(Problem{Returns: r})
	if res.Status != StatusInvalidInput {
		t.Fatalf("expected invalid-input, got %v", res.Status)
	}
}

func TestNearSingularCovariance(t *testing.T) {
	// Two assets with nearly identical returns => near-singular covariance.
	base := mkReturns(150, 1, 61)
	r := make([][]float64, 150)
	for i := range r {
		r[i] = []float64{base[i][0], base[i][0] + 1e-12, base[i][0] * 0.999}
	}
	res := ConstructOnce(Problem{
		Returns:   r,
		Objective: ObjectiveMeanVariance,
		Estimator: EstimatorShrinkage,
	})
	if res.Status != StatusOptimal && res.Status != StatusFeasibleSub {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	for i, w := range res.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("non-finite weight at %d", i)
		}
	}
	if res.Diagnostics.CondNumber <= 0 {
		t.Fatal("condition number must be positive")
	}
}

func TestConstantAsset(t *testing.T) {
	// One asset with zero variance.
	r := mkReturns(100, 3, 67)
	for i := range r {
		r[i][2] = 0.001 // constant
	}
	res := ConstructOnce(Problem{
		Returns:   r,
		Objective: ObjectiveRiskParity,
		Estimator: EstimatorSample,
	})
	if res.Status != StatusOptimal && res.Status != StatusFeasibleSub {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, Constraints{}, 3)
}
