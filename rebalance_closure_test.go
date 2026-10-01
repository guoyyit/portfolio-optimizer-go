package portfolio

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

const rebalanceTestTol = 1e-9

func validateRebalanceOutput(t *testing.T, req RebalanceRequest, res RebalanceResult) {
	t.Helper()
	n := len(req.Current)
	if len(res.NewWeights) != n || len(res.Trades) != n {
		t.Fatalf("result dimension mismatch: %v trades / %v weights for %d assets",
			len(res.Trades), len(res.NewWeights), n)
	}
	var sumW, sumTr, turnover float64
	for i := 0; i < n; i++ {
		sumW += res.NewWeights[i]
		sumTr += res.Trades[i]
		if math.Abs(res.NewWeights[i]-(req.Current[i]+res.Trades[i])) > rebalanceTestTol {
			t.Fatalf("weight %d inconsistent with trade: %v", i, res.NewWeights)
		}
		if req.NoTradeBand > 0 && res.Trades[i] != 0 &&
			math.Abs(res.Trades[i]) < req.NoTradeBand-rebalanceTestTol {
			t.Fatalf("trade %d below no-trade band: %v", i, res.Trades)
		}
		turnover += math.Abs(res.Trades[i])
	}
	if math.Abs(sumW-1) > rebalanceTestTol {
		t.Fatalf("new weights sum %v != 1: %v", sumW, res.NewWeights)
	}
	if math.Abs(sumTr) > rebalanceTestTol {
		t.Fatalf("trades sum %v != 0: %v", sumTr, res.Trades)
	}
	if math.Abs(turnover-res.Turnover) > rebalanceTestTol {
		t.Fatalf("reported turnover %v != computed %v", res.Turnover, turnover)
	}
	if req.MaxTurnover > 0 && turnover > req.MaxTurnover+rebalanceTestTol {
		t.Fatalf("turnover %v exceeds cap %v", turnover, req.MaxTurnover)
	}
	if res.Reason == "" {
		t.Fatalf("empty reason")
	}
	seen := map[DiagFlag]bool{}
	for _, d := range res.Diagnostics {
		if seen[d.Flag] {
			t.Fatalf("duplicate diagnostic flag %v", d.Flag)
		}
		seen[d.Flag] = true
	}
}

func hasDiagFlag(res RebalanceResult, flag DiagFlag) bool {
	for _, d := range res.Diagnostics {
		if d.Flag == flag {
			return true
		}
	}
	return false
}

// The worked example from the task: one sell is suppressed and the buy
// side is trimmed so weights re-close to 1.
func TestRebalancePartialSuppressionExample(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, req, res)
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal after suppression, got %v", res.Status)
	}
	wantW := []float64{0.63, 0.3, 0.07}
	for i := range wantW {
		if math.Abs(res.NewWeights[i]-wantW[i]) > rebalanceTestTol {
			t.Fatalf("weights %v, want %v", res.NewWeights, wantW)
		}
	}
	if res.Trades[1] != 0 {
		t.Fatalf("asset 1 must stay at current weight: %v", res.Trades)
	}
	if math.Abs(res.Turnover-0.06) > rebalanceTestTol {
		t.Fatalf("turnover %v, want 0.06", res.Turnover)
	}
	if !hasDiagFlag(res, DiagNoTradeZone) {
		t.Fatalf("expected no-trade-zone diagnostic: %v", res.Diagnostics)
	}
}

// Single-sided below-threshold cases: the only active side cannot be
// matched, so rebalancing is abandoned and current weights are retained.
func TestRebalanceSingleSidedAbandon(t *testing.T) {
	cases := []struct {
		name    string
		current []float64
		target  []float64
	}{
		{"all suppressed", []float64{0.34, 0.33, 0.33}, []float64{0.35, 0.325, 0.325}},
		{"only sells suppressed", []float64{0.6, 0.2, 0.2}, []float64{0.62, 0.19, 0.19}},
		{"only buys suppressed", []float64{0.3, 0.35, 0.35}, []float64{0.31, 0.345, 0.345}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := RebalanceRequest{
				Current:     tc.current,
				Target:      tc.target,
				NoTradeBand: 0.02,
			}
			res, err := Rebalance(context.Background(), req, nil)
			if err != nil {
				t.Fatal(err)
			}
			validateRebalanceOutput(t, req, res)
			if res.Turnover != 0 {
				t.Fatalf("expected no trades, turnover %v", res.Turnover)
			}
			for i := range tc.current {
				if res.NewWeights[i] != tc.current[i] {
					t.Fatalf("current weights must be retained: %v", res.NewWeights)
				}
			}
			if !hasDiagFlag(res, DiagRebalanceAbandoned) {
				t.Fatalf("expected abandoned diagnostic: %v", res.Diagnostics)
			}
			if !hasDiagFlag(res, DiagNoTradeZone) {
				t.Fatalf("expected no-trade-zone diagnostic listing suppressed assets: %v", res.Diagnostics)
			}
		})
	}
}

// Multiple suppressed assets on the same side: the other side is trimmed
// through a cascade and the budget still closes.
func TestRebalanceMultiSuppressedSameSide(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.7, 0.1, 0.2},
		Target:      []float64{0.75, 0.11, 0.14},
		NoTradeBand: 0.02,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, req, res)
	if res.Trades[1] != 0 {
		t.Fatalf("asset 1 should be suppressed: %v", res.Trades)
	}
	if res.NewWeights[2] > req.Current[2]+rebalanceTestTol {
		t.Fatalf("asset 2 should sell (remain on sell side): %v", res.NewWeights)
	}
	if math.Abs(res.Trades[0]) < req.NoTradeBand-rebalanceTestTol {
		t.Fatalf("buy trade must remain at/above band: %v", res.Trades)
	}
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal, got %v", res.Status)
	}
}

// Suppressed assets on opposite sides: closure is achieved by trimming the
// larger remaining side, without ever trading past target.
func TestRebalanceMultiSuppressedOppositeSides(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.5, 0.2, 0.1, 0.2},
		Target:      []float64{0.6, 0.18, 0.04, 0.18},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, req, res)
	if res.Trades[1] != 0 || res.Trades[3] != 0 {
		t.Fatalf("assets 1 and 3 must be suppressed: %v", res.Trades)
	}
	if res.Trades[0] <= 0 || res.Trades[2] >= 0 {
		t.Fatalf("expected buy on asset 0 and sell on asset 2: %v", res.Trades)
	}
	if math.Abs(res.Trades[0]+res.Trades[2]) > rebalanceTestTol {
		t.Fatalf("buy and sell must offset: %v", res.Trades)
	}
}

// All desired trades are below the band: hold, with sum still exactly 1.
func TestRebalanceAllBelowBand(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.501, 0.499},
		Target:      []float64{0.5, 0.5},
		NoTradeBand: 0.01,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, req, res)
	if res.Turnover != 0 {
		t.Fatalf("expected hold, turnover %v", res.Turnover)
	}
}

// Band and turnover cap jointly bind: closure, band and cap all hold.
func TestRebalanceBandPlusTurnoverCap(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.5, 0.2, 0.1, 0.2},
		Target:      []float64{0.59, 0.21, 0.2, 0.0},
		NoTradeBand: 0.04,
		MaxTurnover: 0.3,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, req, res)
	if !hasDiagFlag(res, DiagTurnoverCapped) {
		t.Fatalf("expected turnover-capped diagnostic: %v", res.Diagnostics)
	}
	if !hasDiagFlag(res, DiagNoTradeZone) {
		t.Fatalf("expected no-trade-zone diagnostic: %v", res.Diagnostics)
	}
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal under cap, got %v", res.Status)
	}
	for i, tr := range res.Trades {
		if tr != 0 && math.Abs(tr) < req.NoTradeBand-rebalanceTestTol {
			t.Fatalf("executed trade %d (%v) slipped below band", i, tr)
		}
	}

	// A cap below the minimum band-compliant round trip must abandon.
	small := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		NoTradeBand: 0.03,
		MaxTurnover: 0.05,
	}
	res2, err := Rebalance(context.Background(), small, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateRebalanceOutput(t, small, res2)
	if res2.Turnover != 0 || !hasDiagFlag(res2, DiagRebalanceAbandoned) {
		t.Fatalf("expected abandonment for tiny cap: turnover=%v diags=%v",
			res2.Turnover, res2.Diagnostics)
	}
	if !hasDiagFlag(res2, DiagTurnoverCapped) || !hasDiagFlag(res2, DiagNoTradeZone) {
		t.Fatalf("expected capped + no-trade-zone diagnostics: %v", res2.Diagnostics)
	}
}

// The turnover cap alone scales both sides equally and keeps closure.
func TestRebalanceTurnoverCapKeepsClosure(t *testing.T) {
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
	validateRebalanceOutput(t, req, res)
	// executed cost must equal the reported weighted traded notional
	var wantCost float64
	for _, tr := range res.Trades {
		if tr > 0 {
			wantCost += tr * req.Costs.BuyCost
		} else if tr < 0 {
			wantCost += -tr * req.Costs.SellCost
		}
	}
	if math.Abs(res.Cost-wantCost) > rebalanceTestTol {
		t.Fatalf("cost %v, want %v", res.Cost, wantCost)
	}
}

// Cost/benefit gate abandons: weights equal current, sum 1, cost 0.
func TestRebalanceCostGateAbandonKeepsCurrent(t *testing.T) {
	est := testEstimates()
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
	validateRebalanceOutput(t, req, res)
	for i := range req.Current {
		if res.NewWeights[i] != req.Current[i] {
			t.Fatalf("gate must retain current weights: %v", res.NewWeights)
		}
	}
	if res.Cost != 0 {
		t.Fatalf("abandoned rebalance reports zero executed cost, got %v", res.Cost)
	}
	if !hasDiagFlag(res, DiagRebalanceAbandoned) {
		t.Fatalf("expected abandoned diagnostic: %v", res.Diagnostics)
	}
}

func TestRebalanceInvalidInputsDistinguished(t *testing.T) {
	good := []float64{0.5, 0.5}
	cases := []struct {
		name string
		req  RebalanceRequest
	}{
		{"length mismatch", RebalanceRequest{Current: good, Target: []float64{1}}},
		{"current non-finite", RebalanceRequest{Current: []float64{math.NaN(), 0.5}, Target: good}},
		{"target non-finite", RebalanceRequest{Current: good, Target: []float64{0.5, math.Inf(1)}}},
		{"current sum", RebalanceRequest{Current: []float64{0.6, 0.5}, Target: good}},
		{"target sum", RebalanceRequest{Current: good, Target: []float64{0.4, 0.4}}},
		{"negative band", RebalanceRequest{Current: good, Target: good, NoTradeBand: -0.01}},
		{"negative turnover", RebalanceRequest{Current: good, Target: good, MaxTurnover: -0.01}},
		{"negative benefit", RebalanceRequest{Current: good, Target: good, MinBenefit: -1}},
		{"negative cost", RebalanceRequest{Current: good, Target: good, Costs: CostModel{BuyCost: -0.01}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := Rebalance(context.Background(), tc.req, nil)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if res.Status != StatusInvalidInput {
				t.Fatalf("want invalid-input for %s, got %v (%s)", tc.name, res.Status, res.Reason)
			}
			if res.Reason == "" {
				t.Fatalf("expected distinguishable reason for %s", tc.name)
			}
		})
	}
}

// Concurrent requests with different inputs must not contaminate each
// other; -race covers shared-state access.
func TestRebalanceConcurrentNoContamination(t *testing.T) {
	requests := []RebalanceRequest{
		{
			Current:     []float64{0.6, 0.3, 0.1},
			Target:      []float64{0.65, 0.28, 0.07},
			NoTradeBand: 0.03,
		},
		{
			Current:     []float64{0.5, 0.2, 0.1, 0.2},
			Target:      []float64{0.6, 0.18, 0.04, 0.18},
			NoTradeBand: 0.03,
			MaxTurnover: 0.2,
			Costs:       CostModel{BuyCost: 0.002, SellCost: 0.001},
		},
		{
			Current:    []float64{0.25, 0.25, 0.25, 0.25},
			Target:     []float64{0.26, 0.24, 0.25, 0.25},
			Costs:      CostModel{BuyCost: 0.05, SellCost: 0.05},
			MinBenefit: 2.0,
		},
		{
			Current: []float64{0.25, 0.25, 0.25, 0.25},
			Target:  []float64{0.7, 0.1, 0.1, 0.1},
		},
	}
	expected := make([]RebalanceResult, len(requests))
	for i, req := range requests {
		var est *Estimates
		if i >= 2 {
			e := testEstimates()
			est = &e
		}
		res, err := Rebalance(context.Background(), req, est)
		if err != nil {
			t.Fatal(err)
		}
		expected[i] = res
	}

	const goroutines = 32
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for round := 0; round < 25; round++ {
				i := (seed + round) % len(requests)
				var est *Estimates
				if i >= 2 {
					e := testEstimates()
					est = &e
				}
				res, err := Rebalance(context.Background(), requests[i], est)
				if err != nil {
					errs <- err
					return
				}
				if res.Status != expected[i].Status {
					errs <- fmt.Errorf("status mismatch for request %d: %v vs %v",
						i, res.Status, expected[i].Status)
					return
				}
				for j := range res.NewWeights {
					if res.NewWeights[j] != expected[i].NewWeights[j] {
						errs <- fmt.Errorf("weight contamination for request %d: %v vs %v",
							i, res.NewWeights, expected[i].NewWeights)
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// Same input must produce bit-identical output across repeated calls.
func TestRebalanceBitIdentical(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		NoTradeBand: 0.03,
		MaxTurnover: 0.2,
		Costs:       CostModel{BuyCost: 0.002, SellCost: 0.001},
	}
	r1, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 10; k++ {
		r2, err := Rebalance(context.Background(), req, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range r1.Trades {
			if r2.Trades[i] != r1.Trades[i] || r2.NewWeights[i] != r1.NewWeights[i] {
				t.Fatalf("output not bit-identical on call %d", k)
			}
		}
	}
}
