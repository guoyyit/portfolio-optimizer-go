package portfolio

import (
	"context"
	"math"
	"testing"
	"time"
)

func testEstimates() Estimates {
	// 4 assets, diagonal-dominant covariance, distinct mu
	mu := []float64{0.08, 0.05, 0.03, 0.06}
	cov := [][]float64{
		{0.040, 0.006, 0.002, 0.004},
		{0.006, 0.030, 0.003, 0.002},
		{0.002, 0.003, 0.020, 0.001},
		{0.004, 0.002, 0.001, 0.025},
	}
	return Estimates{Mu: mu, Cov: cov, NSamples: 100}
}

func TestMeanVarianceBasic(t *testing.T) {
	res, err := Optimize(context.Background(), testEstimates(), nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal && res.Status != StatusSuboptimal {
		t.Fatalf("bad status %v: %s", res.Status, res.Message)
	}
	if !satisfies(res.Weights, nil, 4, 1e-8) {
		t.Fatalf("weights violate constraints: %v", res.Weights)
	}
	var sum float64
	for _, w := range res.Weights {
		sum += w
	}
	if math.Abs(sum-1) > 1e-8 {
		t.Fatalf("budget violated: %v", sum)
	}
	// low risk aversion concentrates in the high-return asset
	if res.Weights[0] < 0.5 {
		t.Errorf("expected tilt toward high-return asset 0: %v", res.Weights)
	}
	// high risk aversion diversifies
	res2, err := Optimize(context.Background(), testEstimates(), nil, Options{RiskAversion: 20})
	if err != nil {
		t.Fatal(err)
	}
	if res2.Metrics.EffectiveBets < 1.5 {
		t.Errorf("high risk aversion should diversify: %+v", res2.Metrics)
	}
	if res2.Metrics.Volatility >= res.Metrics.Volatility {
		t.Errorf("higher risk aversion should lower volatility: %v vs %v",
			res2.Metrics.Volatility, res.Metrics.Volatility)
	}
}

func TestRiskParityEqualRC(t *testing.T) {
	res, err := Optimize(context.Background(), testEstimates(), nil,
		Options{Objective: ObjectiveRiskParity, MaxIterations: 5000})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal && res.Status != StatusSuboptimal {
		t.Fatalf("bad status %v", res.Status)
	}
	rc := res.Metrics.RiskContributions
	for i, v := range rc {
		if math.Abs(v-0.25) > 0.02 {
			t.Errorf("risk contribution %d = %.4f, want ~0.25 (all %v)", i, v, rc)
		}
	}
}

func TestBoxConstraintsRespected(t *testing.T) {
	c := &Constraints{
		MinWeight: []float64{0.1, 0.1, 0.1, 0.1},
		MaxWeight: []float64{0.4, 0.4, 0.4, 0.4},
	}
	res, err := Optimize(context.Background(), testEstimates(), c, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range res.Weights {
		if w < 0.1-1e-9 || w > 0.4+1e-9 {
			t.Fatalf("box violated at %d: %v", i, res.Weights)
		}
	}
}

func TestGroupCapAndTurnover(t *testing.T) {
	c := &Constraints{
		GroupCaps:      []GroupCap{{Name: "risky", Members: []int{0, 3}, Cap: 0.5}},
		MaxTurnover:    0.4,
		CurrentWeights: []float64{0.25, 0.25, 0.25, 0.25},
	}
	res, err := Optimize(context.Background(), testEstimates(), c, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !satisfies(res.Weights, c, 4, 1e-8) {
		t.Fatalf("violates constraints: %v", res.Weights)
	}
	if res.Weights[0]+res.Weights[3] > 0.5+1e-8 {
		t.Fatalf("group cap violated: %v", res.Weights)
	}
	var to float64
	for i := range res.Weights {
		to += math.Abs(res.Weights[i] - 0.25)
	}
	if to > 0.4+1e-8 {
		t.Fatalf("turnover violated: %v", to)
	}
}

func TestInfeasibleStatus(t *testing.T) {
	c := &Constraints{MinWeight: []float64{0.9, 0.9, 0.9, 0.9}}
	res, err := Optimize(context.Background(), testEstimates(), c, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusInfeasible {
		t.Fatalf("want infeasible, got %v", res.Status)
	}
	if res.Weights != nil {
		t.Fatalf("infeasible must not return weights")
	}
}

func TestDeterminism(t *testing.T) {
	r1, _ := Optimize(context.Background(), testEstimates(), nil, Options{Seed: 42})
	r2, _ := Optimize(context.Background(), testEstimates(), nil, Options{Seed: 42})
	for i := range r1.Weights {
		if r1.Weights[i] != r2.Weights[i] {
			t.Fatalf("non-deterministic: %v vs %v", r1.Weights, r2.Weights)
		}
	}
}

func TestTimeoutReturnsFeasible(t *testing.T) {
	res, err := Optimize(context.Background(), testEstimates(), nil,
		Options{TimeBudget: time.Nanosecond, MaxIterations: 100000})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusTimeout && res.Status != StatusOptimal {
		t.Fatalf("want timeout/optimal, got %v", res.Status)
	}
	if res.Status == StatusTimeout && !satisfies(res.Weights, nil, 4, 1e-8) {
		t.Fatalf("timeout result infeasible: %v", res.Weights)
	}
}

func TestCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := Optimize(ctx, testEstimates(), nil, Options{MaxIterations: 100000})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusTimeout {
		t.Fatalf("cancelled context should yield timeout, got %v", res.Status)
	}
	if !satisfies(res.Weights, nil, 4, 1e-8) {
		t.Fatalf("cancelled result infeasible")
	}
}
