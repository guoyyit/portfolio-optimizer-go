package portfolio

import (
	"math"
	"strings"
	"testing"
)

func TestRebalanceFull(t *testing.T) {
	res, err := Rebalance(RebalanceRequest{
		Current: []float64{0.5, 0.5, 0},
		Target:  []float64{0, 0.5, 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	// No cap: should reach target exactly.
	for i := range res.NewWeights {
		if math.Abs(res.NewWeights[i]-[]float64{0, 0.5, 0.5}[i]) > 1e-9 {
			t.Fatalf("did not reach target: %v", res.NewWeights)
		}
	}
	sum := 0.0
	for _, w := range res.NewWeights {
		sum += w
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("new weights sum to %v", sum)
	}
	if math.Abs(res.Turnover-1.0) > 1e-9 {
		t.Fatalf("turnover %v, want 1.0", res.Turnover)
	}
}

func TestRebalanceTurnoverCap(t *testing.T) {
	res, err := Rebalance(RebalanceRequest{
		Current:     []float64{0.5, 0.5, 0},
		Target:      []float64{0, 0.5, 0.5},
		MaxTurnover: 0.4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Turnover > 0.4+1e-9 {
		t.Fatalf("turnover %v exceeds cap 0.4", res.Turnover)
	}
	// Trades must sum to zero (portfolio stays fully invested).
	sum := 0.0
	for _, tr := range res.Trades {
		sum += tr
	}
	if math.Abs(sum) > 1e-9 {
		t.Fatalf("trades sum to %v, want 0", sum)
	}
	// Should move toward target, not away.
	if res.Benefit <= 0 {
		t.Fatalf("expected positive benefit, got %v", res.Benefit)
	}
	if res.Explanation == "" {
		t.Fatal("expected explanation")
	}
}

func TestRebalanceNoTradeBand(t *testing.T) {
	res, err := Rebalance(RebalanceRequest{
		Current:     []float64{0.34, 0.33, 0.33},
		Target:      []float64{0.34 + 0.005, 0.33 - 0.002, 0.33 - 0.003},
		NoTradeBand: 0.01,
	})
	if err != nil {
		t.Fatal(err)
	}
	// All trades are below the band: nothing should happen.
	for i, tr := range res.Trades {
		if tr != 0 {
			t.Fatalf("trade %d = %v should be suppressed", i, tr)
		}
	}
	if res.Turnover != 0 {
		t.Fatalf("turnover %v, want 0", res.Turnover)
	}
	if len(res.Skipped) == 0 {
		t.Fatal("expected skipped indices")
	}
}

func TestRebalanceCostGate(t *testing.T) {
	// Asset 2 has cost rate 2.0 (> 1/aversion): trades into it are dropped.
	res, err := Rebalance(RebalanceRequest{
		Current:  []float64{0.5, 0.5, 0},
		Target:   []float64{0, 0.5, 0.5},
		CostRate: []float64{0.001, 0.001, 2.0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.Trades[2]) > 1e-12 {
		t.Fatalf("trade into over-priced asset not gated: %v", res.Trades[2])
	}
	if !strings.Contains(res.Explanation, "cost") {
		t.Fatalf("explanation should mention cost gate: %s", res.Explanation)
	}
	// Cost accounting must be consistent.
	want := 0.0
	for i, tr := range res.Trades {
		want += math.Abs(tr) * []float64{0.001, 0.001, 2.0}[i]
	}
	if math.Abs(res.Cost-want) > 1e-12 {
		t.Fatalf("cost %v != %v", res.Cost, want)
	}
}

func TestRebalanceDeterministic(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.4, 0.3, 0.3},
		Target:      []float64{0.2, 0.5, 0.3},
		MaxTurnover: 0.15,
		UniformCost: 0.002,
	}
	r1, err1 := Rebalance(req)
	r2, err2 := Rebalance(req)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	for i := range r1.Trades {
		if math.Float64bits(r1.Trades[i]) != math.Float64bits(r2.Trades[i]) {
			t.Fatal("rebalance not deterministic")
		}
	}
}

func TestRebalanceInvalid(t *testing.T) {
	if _, err := Rebalance(RebalanceRequest{Current: []float64{1}, Target: []float64{1, 0}}); err == nil {
		t.Fatal("expected length mismatch error")
	}
	if _, err := Rebalance(RebalanceRequest{
		Current: []float64{math.NaN()}, Target: []float64{1},
	}); err == nil {
		t.Fatal("expected NaN error")
	}
}
