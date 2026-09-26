package portfolio

import (
	"context"
	"math"
	"testing"
)

// End-to-end: assets >> samples with missing data and outliers.
func TestEndToEndHighDimMessy(t *testing.T) {
	rngData := makeReturns(30, 50, func(t, j int) float64 {
		v := 0.01 * math.Sin(float64(t)*0.31+float64(j)*0.7)
		if (t*7+j*13)%41 == 0 {
			v = math.NaN()
		}
		if t == 3 && j == 5 {
			v = 50 // outlier
		}
		return v
	})
	e := NewEngine()
	res, err := e.Build(context.Background(), BuildRequest{
		Key:        "messy",
		Returns:    rngData,
		Estimation: EstimationConfig{WinsorizeLimit: 4},
		Options:    Options{Objective: ObjectiveRiskParity, MaxIterations: 800},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal && res.Status != StatusSuboptimal {
		t.Fatalf("bad status %v: %s", res.Status, res.Message)
	}
	for _, w := range res.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatal("NaN/Inf weight")
		}
	}
	if !satisfies(res.Weights, nil, 50, 1e-8) {
		t.Fatalf("infeasible weights")
	}
	flags := map[DiagFlag]bool{}
	for _, d := range res.Diagnostics {
		flags[d.Flag] = true
	}
	for _, want := range []DiagFlag{DiagMissingData, DiagHighDimensional, DiagOutliersWinsorized} {
		if !flags[want] {
			t.Errorf("missing diagnostic %q in %v", want, flags)
		}
	}
}

// Near-singular covariance: two almost perfectly correlated assets.
func TestNearSingularCovariance(t *testing.T) {
	est := Estimates{
		Mu: []float64{0.05, 0.05, 0.03},
		Cov: [][]float64{
			{0.04, 0.0399999, 0.001},
			{0.0399999, 0.04, 0.001},
			{0.001, 0.001, 0.01},
		},
		NSamples: 100,
	}
	res, err := Optimize(context.Background(), est, nil, Options{RiskAversion: 10})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status == StatusInfeasible || res.Status == StatusInvalidInput {
		t.Fatalf("near-singular should still solve: %v %s", res.Status, res.Message)
	}
	for _, w := range res.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatal("NaN/Inf weight")
		}
	}
	if !satisfies(res.Weights, nil, 3, 1e-8) {
		t.Fatalf("infeasible: %v", res.Weights)
	}
}

// Bit-identical reproducibility across independent engine instances.
func TestBitIdenticalReproducibility(t *testing.T) {
	req := buildRequestFor("rep", 1.5)
	req.Options = Options{Objective: ObjectiveRiskParity, Seed: 99, MaxIterations: 1000}
	r1, err := NewEngine().Build(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := NewEngine().Build(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(r1.Weights) != len(r2.Weights) {
		t.Fatal("dim mismatch")
	}
	for i := range r1.Weights {
		if math.Float64bits(r1.Weights[i]) != math.Float64bits(r2.Weights[i]) {
			t.Fatalf("not bit-identical at %d: %x vs %x", i,
				math.Float64bits(r1.Weights[i]), math.Float64bits(r2.Weights[i]))
		}
	}
	// robustness report reproducibility with fixed seed
	est, err := EstimateWith(req.Returns, nil, req.Estimation)
	if err != nil {
		t.Fatal(err)
	}
	rep1, _ := AssessRobustness(context.Background(), est, nil, req.Options, RobustnessConfig{Seed: 5, NumPerturbations: 4})
	rep2, _ := AssessRobustness(context.Background(), est, nil, req.Options, RobustnessConfig{Seed: 5, NumPerturbations: 4})
	for i := range rep1.WeightSensitivity {
		if math.Float64bits(rep1.WeightSensitivity[i]) != math.Float64bits(rep2.WeightSensitivity[i]) {
			t.Fatalf("robustness not bit-identical")
		}
	}
}

// Constraint boundary: budget exactly equals sum of min weights.
func TestBoundaryMinWeights(t *testing.T) {
	c := &Constraints{MinWeight: []float64{0.5, 0.3, 0.2}}
	res, err := Optimize(context.Background(), Estimates{
		Mu:  []float64{0.1, 0.1, 0.1},
		Cov: [][]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
	}, c, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal && res.Status != StatusSuboptimal {
		t.Fatalf("boundary case should solve: %v", res.Status)
	}
	for i, want := range []float64{0.5, 0.3, 0.2} {
		if math.Abs(res.Weights[i]-want) > 1e-6 {
			t.Fatalf("boundary weights %v, want ~[0.5 0.3 0.2]", res.Weights)
		}
	}
}

// Leverage constraint with long-short box.
func TestLeverageConstraint(t *testing.T) {
	c := &Constraints{
		MinWeight:   []float64{-1, -1, -1},
		MaxWeight:   []float64{1, 1, 1},
		MaxLeverage: 1.2,
	}
	est := Estimates{
		Mu:  []float64{0.10, -0.08, 0.02},
		Cov: [][]float64{{0.05, 0, 0}, {0, 0.05, 0}, {0, 0, 0.05}},
	}
	res, err := Optimize(context.Background(), est, c, Options{RiskAversion: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if !satisfies(res.Weights, c, 3, 1e-8) {
		t.Fatalf("leverage violated: %v", res.Weights)
	}
	var l1 float64
	for _, w := range res.Weights {
		l1 += math.Abs(w)
	}
	if l1 > 1.2+1e-8 {
		t.Fatalf("leverage %.4f > 1.2", l1)
	}
}
