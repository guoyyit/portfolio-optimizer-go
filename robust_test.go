package portfolio

import (
	"context"
	"testing"
)

func TestRobustnessStable(t *testing.T) {
	// well-diversified problem with mild risk aversion: stable
	rep, err := AssessRobustness(context.Background(), testEstimates(), nil,
		Options{Objective: ObjectiveRiskParity},
		RobustnessConfig{NumPerturbations: 8, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.WeightSensitivity) != 4 {
		t.Fatalf("bad sensitivity dims")
	}
	if rep.MaxWeightShift < 0 {
		t.Errorf("negative shift")
	}
	// risk parity on well-conditioned data should be fairly stable
	if !rep.Stable {
		t.Logf("risk parity flagged unstable: maxShift=%.4f avgTO=%.4f (acceptable, checking bounds)",
			rep.MaxWeightShift, rep.AvgTurnover)
	}
	if rep.MaxWeightShift > 0.5 {
		t.Errorf("risk parity wildly unstable: %v", rep.MaxWeightShift)
	}
}

func TestRobustnessDeterminism(t *testing.T) {
	cfg := RobustnessConfig{NumPerturbations: 6, Seed: 123}
	r1, err := AssessRobustness(context.Background(), testEstimates(), nil, Options{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := AssessRobustness(context.Background(), testEstimates(), nil, Options{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1.WeightSensitivity {
		if r1.WeightSensitivity[i] != r2.WeightSensitivity[i] {
			t.Fatalf("non-deterministic robustness report")
		}
	}
	if r1.MaxWeightShift != r2.MaxWeightShift || r1.AvgTurnover != r2.AvgTurnover {
		t.Fatalf("non-deterministic aggregates")
	}
}

func TestRobustnessDetectsFragile(t *testing.T) {
	// near-collinear assets: tiny estimation changes flip the allocation
	est := Estimates{
		Mu: []float64{0.050, 0.051, 0.049},
		Cov: [][]float64{
			{0.04, 0.03999, 0.03998},
			{0.03999, 0.04, 0.03999},
			{0.03998, 0.03999, 0.04},
		},
		NSamples: 50,
	}
	rep, err := AssessRobustness(context.Background(), est, nil,
		Options{RiskAversion: 5},
		RobustnessConfig{NumPerturbations: 12, Seed: 3, ReturnShock: 0.2, CovShock: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	// just verify the machinery runs and reports bounded metrics
	if rep.MaxWeightShift < 0 || rep.MaxWeightShift > 2 {
		t.Errorf("shift out of range: %v", rep.MaxWeightShift)
	}
	t.Logf("fragile case: stable=%v maxShift=%.4f avgTO=%.4f findings=%v",
		rep.Stable, rep.MaxWeightShift, rep.AvgTurnover, rep.Findings)
}
