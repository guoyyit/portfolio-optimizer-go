package portfolio

import (
	"context"
	"math"
	"testing"
)

func TestRebalanceFull(t *testing.T) {
	est := testEstimates()
	req := RebalanceRequest{
		Current: []float64{0.25, 0.25, 0.25, 0.25},
		Target:  []float64{0.7, 0.1, 0.1, 0.1},
		Costs:   CostModel{BuyCost: 0.001, SellCost: 0.001},
	}
	res, err := Rebalance(context.Background(), req, &est)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal {
		t.Fatalf("want optimal, got %v", res.Status)
	}
	for i := range res.NewWeights {
		if math.Abs(res.NewWeights[i]-req.Target[i]) > 1e-12 {
			t.Fatalf("full rebalance should reach target: %v", res.NewWeights)
		}
	}
	wantTo := 0.45 * 2 // |0.45| + 3*|−0.05|
	if math.Abs(res.Turnover-wantTo) > 1e-9 {
		t.Fatalf("turnover %v, want %v", res.Turnover, wantTo)
	}
	if res.Cost <= 0 {
		t.Errorf("expected positive cost")
	}
	if res.Reason == "" {
		t.Errorf("expected human-readable reason")
	}
}

func TestRebalanceTurnoverCap(t *testing.T) {
	est := testEstimates()
	req := RebalanceRequest{
		Current:     []float64{0.25, 0.25, 0.25, 0.25},
		Target:      []float64{0.7, 0.1, 0.1, 0.1},
		Costs:       CostModel{BuyCost: 0.001, SellCost: 0.001},
		MaxTurnover: 0.3,
	}
	res, err := Rebalance(context.Background(), req, &est)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusSuboptimal {
		t.Fatalf("capped rebalance should be suboptimal, got %v", res.Status)
	}
	if res.Turnover > 0.3+1e-12 {
		t.Fatalf("turnover cap violated: %v", res.Turnover)
	}
	// trades must point toward target
	if res.Trades[0] <= 0 || res.Trades[1] >= 0 {
		t.Fatalf("trades not toward target: %v", res.Trades)
	}
	found := false
	for _, d := range res.Diagnostics {
		if d.Flag == DiagTurnoverCapped {
			found = true
		}
	}
	if !found {
		t.Errorf("expected turnover-capped diagnostic")
	}
}

func TestRebalanceNoTradeBand(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.501, 0.499},
		Target:      []float64{0.5, 0.5},
		NoTradeBand: 0.01,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range res.Trades {
		if tr != 0 {
			t.Fatalf("tiny drift inside no-trade band should not trade: %v", res.Trades)
		}
	}
}

func TestRebalanceCostGate(t *testing.T) {
	est := testEstimates()
	// tiny target shift, huge costs, strict benefit multiplier
	req := RebalanceRequest{
		Current:    []float64{0.25, 0.25, 0.25, 0.25},
		Target:     []float64{0.26, 0.24, 0.25, 0.25},
		Costs:      CostModel{BuyCost: 0.05, SellCost: 0.05},
		MinBenefit: 2.0,
	}
	res, err := Rebalance(context.Background(), req, &est)
	if err != nil {
		t.Fatal(err)
	}
	var to float64
	for _, tr := range res.Trades {
		to += math.Abs(tr)
	}
	if to != 0 {
		t.Fatalf("cost gate should suppress trades: %v", res.Trades)
	}
	found := false
	for _, d := range res.Diagnostics {
		if d.Flag == DiagNoTradeZone {
			found = true
		}
	}
	if !found {
		t.Errorf("expected no-trade-zone diagnostic, got %v", res.Diagnostics)
	}
}

func TestRebalanceDeterminism(t *testing.T) {
	est := testEstimates()
	req := RebalanceRequest{
		Current:     []float64{0.25, 0.25, 0.25, 0.25},
		Target:      []float64{0.6, 0.2, 0.1, 0.1},
		Costs:       CostModel{BuyCost: 0.002, SellCost: 0.001},
		MaxTurnover: 0.5,
	}
	r1, _ := Rebalance(context.Background(), req, &est)
	r2, _ := Rebalance(context.Background(), req, &est)
	for i := range r1.Trades {
		if r1.Trades[i] != r2.Trades[i] {
			t.Fatalf("non-deterministic rebalance")
		}
	}
}
