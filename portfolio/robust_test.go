package portfolio

import (
	"math"
	"testing"
)

func TestRobustnessReproducible(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(150, 4, 97),
		Objective: ObjectiveMeanVariance,
		Seed:      1234,
	}
	r1, err := AssessRobustness(p, 8)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := AssessRobustness(p, 8)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(r1.MaxWeightShift) != math.Float64bits(r2.MaxWeightShift) ||
		math.Float64bits(r1.MeanWeightShift) != math.Float64bits(r2.MeanWeightShift) {
		t.Fatal("robustness report not reproducible with same seed")
	}
	for i := range r1.BaseWeights {
		if math.Float64bits(r1.BaseWeights[i]) != math.Float64bits(r2.BaseWeights[i]) {
			t.Fatal("base weights not reproducible")
		}
	}
	if r1.Trials != 8 {
		t.Fatalf("trials=%d", r1.Trials)
	}
}

func TestRobustnessSeedChangesResult(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(150, 4, 101),
		Objective: ObjectiveMeanVariance,
		Seed:      1,
	}
	r1, _ := AssessRobustness(p, 8)
	p.Seed = 2
	r2, _ := AssessRobustness(p, 8)
	if r1.MeanWeightShift == r2.MeanWeightShift && r1.MaxWeightShift == r2.MaxWeightShift {
		t.Fatal("different seeds should perturb differently")
	}
}

func TestRobustnessMetricsPresent(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 5, 103),
		Objective: ObjectiveRiskParity,
		Estimator: EstimatorShrinkage,
	}
	rep, err := AssessRobustness(p, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rep.CondNumber <= 0 {
		t.Fatal("condition number missing")
	}
	if rep.Herfindahl <= 0 || rep.EffectiveN <= 0 {
		t.Fatal("concentration metrics missing")
	}
	if len(rep.WeightSensitivity) != 5 {
		t.Fatalf("per-asset sensitivity length %d", len(rep.WeightSensitivity))
	}
	if rep.MaxWeightShift < rep.MeanWeightShift {
		t.Fatal("max shift must be >= mean shift")
	}
	// Risk parity on well-behaved data should be reasonably stable.
	if rep.MaxWeightShift > 0.5 {
		t.Fatalf("risk parity unexpectedly unstable: %v", rep.MaxWeightShift)
	}
}

func TestRobustnessIdentifiesUnstable(t *testing.T) {
	// Highly collinear assets + mean-variance on tiny return differences
	// produces an ill-conditioned, concentrated, unstable allocation.
	base := mkReturns(120, 1, 107)
	r := make([][]float64, 120)
	for i := range r {
		r[i] = []float64{
			base[i][0] + 0.00001,
			base[i][0],
			base[i][0] - 0.00001,
			base[i][0] * 1.000001,
		}
	}
	p := Problem{
		Returns:   r,
		Objective: ObjectiveMeanVariance,
		Estimator: EstimatorSample,
	}
	rep, err := AssessRobustness(p, 12)
	if err != nil {
		t.Fatal(err)
	}
	// At least one instability signal must fire: high condition number,
	// concentration, or large weight shift.
	unstable := rep.CondNumber > 1e6 || rep.Herfindahl > 0.9 || rep.MaxWeightShift > 0.05
	if !unstable {
		t.Fatalf("expected instability signals, got cond=%v hhi=%v shift=%v",
			rep.CondNumber, rep.Herfindahl, rep.MaxWeightShift)
	}
}
