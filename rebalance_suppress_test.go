package portfolio

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
)

func rbWeightSum(w []float64) float64 {
	var s float64
	for _, v := range w {
		s += v
	}
	return s
}

// assertClosedAndValid verifies the hard guarantees of every weight-bearing
// Rebalance result: lengths, budget closure, threshold/turnover compliance,
// toward-target sign and consistent turnover field.
func assertClosedAndValid(t *testing.T, req RebalanceRequest, res RebalanceResult) {
	t.Helper()
	n := len(req.Current)
	tol := 1e-9 * float64(n)
	if len(res.Trades) != n || len(res.NewWeights) != n {
		t.Fatalf("length mismatch: trades=%v new=%v", res.Trades, res.NewWeights)
	}
	if s := rbWeightSum(res.NewWeights); math.Abs(s-1) > tol {
		t.Fatalf("new weights %v sum to %v, want 1", res.NewWeights, s)
	}
	var to float64
	for i, d := range res.Trades {
		to += math.Abs(d)
		if math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatalf("non-finite trade %d", i)
		}
		desired := req.Target[i] - req.Current[i]
		if d != 0 && (d > 0) != (desired > 0) {
			t.Fatalf("asset %d trade %v points away from target", i, d)
		}
		if d != 0 && math.Abs(d) < req.NoTradeBand {
			t.Fatalf("asset %d executed trade %v below band %v", i, d, req.NoTradeBand)
		}
		if math.Abs(res.NewWeights[i]-(req.Current[i]+d)) > tol {
			t.Fatalf("asset %d new weight inconsistent with trade", i)
		}
	}
	if math.Abs(to-res.Turnover) > tol {
		t.Fatalf("turnover field %v != computed %v", res.Turnover, to)
	}
	if req.MaxTurnover > 0 && to > req.MaxTurnover+tol {
		t.Fatalf("turnover %v exceeds cap %v", to, req.MaxTurnover)
	}
	if res.Reason == "" {
		t.Fatalf("empty reason")
	}
}

func hasDiag(res RebalanceResult, flag DiagFlag) bool {
	for _, d := range res.Diagnostics {
		if d.Flag == flag {
			return true
		}
	}
	return false
}

// 正文示例：单个资产抑制，买入侧重闭合，权重和严格为 1。
func TestRebalanceSuppressionExample(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal partial execution, got %v", res.Status)
	}
	assertClosedAndValid(t, req, res)
	if res.Trades[1] != 0 || res.NewWeights[1] != 0.3 {
		t.Fatalf("suppressed asset must keep current weight: trades=%v new=%v", res.Trades, res.NewWeights)
	}
	if math.Abs(res.Trades[0]-0.03) > 1e-12 || math.Abs(res.Trades[2]+0.03) > 1e-12 {
		t.Fatalf("buy side must be reclosed to 0.03: %v", res.Trades)
	}
	if !hasDiag(res, DiagNoTradeZone) || !hasDiag(res, DiagTradesReclosed) {
		t.Fatalf("expected hold + reclose diagnostics, got %+v", res.Diagnostics)
	}
}

// 单边低于阈值：小额卖出被抑制，买入侧重闭合。
func TestRebalanceSingleSideSuppressed(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.3, 0.5, 0.2},
		Target:      []float64{0.4, 0.48, 0.12},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClosedAndValid(t, req, res)
	if res.Trades[1] != 0 || res.NewWeights[1] != 0.5 {
		t.Fatalf("asset1 must be held: %v", res.Trades)
	}
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal, got %v", res.Status)
	}
}

// 多边低于阈值且方向相反：买卖两侧各有一个小额交易被抑制。
func TestRebalanceMultiSuppressedOppositeDirections(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.25, 0.25, 0.25, 0.25},
		Target:      []float64{0.26, 0.34, 0.23, 0.17},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClosedAndValid(t, req, res)
	if res.Trades[0] != 0 || res.Trades[2] != 0 {
		t.Fatalf("assets 0 and 2 must be held: %v", res.Trades)
	}
	if res.NewWeights[0] != 0.25 || res.NewWeights[2] != 0.25 {
		t.Fatalf("held assets keep current weight: %v", res.NewWeights)
	}
}

// 多边低于阈值且方向相同：被抑制资产全在买入侧，卖出侧无法闭合，
// 完全放弃并返回当前权重。
func TestRebalanceMultiSuppressedSameDirection(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.2, 0.2, 0.4, 0.2},
		Target:      []float64{0.21, 0.22, 0.37, 0.2},
		NoTradeBand: 0.03,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClosedAndValid(t, req, res)
	if res.Status != StatusAbandoned {
		t.Fatalf("want abandoned when one side vanishes, got %v", res.Status)
	}
	for i := range req.Current {
		if res.Trades[i] != 0 || res.NewWeights[i] != req.Current[i] {
			t.Fatalf("abandonment must return current weights: %v %v", res.Trades, res.NewWeights)
		}
	}
}

// 全部低于阈值：完全放弃，返回当前权重。
func TestRebalanceAllBelowBand(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.34, 0.33, 0.33},
		Target:      []float64{0.35, 0.325, 0.325},
		NoTradeBand: 0.02,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusAbandoned {
		t.Fatalf("want abandoned, got %v", res.Status)
	}
	assertClosedAndValid(t, req, res)
	for i := range req.Current {
		if res.Trades[i] != 0 || res.NewWeights[i] != req.Current[i] {
			t.Fatalf("all-below-band must hold current: trades=%v new=%v", res.Trades, res.NewWeights)
		}
	}
	if !hasDiag(res, DiagRebalanceAbandoned) {
		t.Fatalf("expected abandoned diagnostic: %+v", res.Diagnostics)
	}
}

// 恰好处于目标时为 optimal 且零交易。
func TestRebalanceAtTargetOptimal(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.5, 0.5},
		Target:      []float64{0.5, 0.5},
		NoTradeBand: 0.01,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal {
		t.Fatalf("want optimal, got %v", res.Status)
	}
	assertClosedAndValid(t, req, res)
}

// 阈值与换手上限叠加：同时闭合权重和、满足阈值与换手约束。
func TestRebalanceBandAndTurnoverCapBinding(t *testing.T) {
	// 四资产：deltas +0.10, +0.01(supp), -0.05, -0.06。抑制后
	// buys=0.10, sells=0.11，买入侧重闭合到 0.10（turnover 0.20）；
	// cap 0.10 缩放后每笔为 0.05/0.05，仍 >= band 0.03，可行且绑定。
	req := RebalanceRequest{
		Current:     []float64{0.2, 0.3, 0.3, 0.2},
		Target:      []float64{0.3, 0.31, 0.25, 0.14},
		NoTradeBand: 0.03,
		MaxTurnover: 0.10,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClosedAndValid(t, req, res)
	if res.Status != StatusSuboptimal {
		t.Fatalf("want suboptimal, got %v", res.Status)
	}
	if res.Trades[1] != 0 || res.NewWeights[1] != 0.3 {
		t.Fatalf("asset1 must remain suppressed: %v", res.Trades)
	}
	if math.Abs(res.Turnover-0.10) > 1e-9 {
		t.Fatalf("turnover cap violated: %v", res.Turnover)
	}
	for i, d := range res.Trades {
		if d != 0 && math.Abs(d) < req.NoTradeBand-1e-12 {
			t.Fatalf("cap scaling created below-band trade at %d: %v", i, res.Trades)
		}
	}
	if !hasDiag(res, DiagTurnoverCapped) || !hasDiag(res, DiagTradesReclosed) {
		t.Fatalf("expected cap+reclose diagnostics: %+v", res.Diagnostics)
	}
}

// 换手上限极小，叠加阈值后无任何可执行交易：abandoned 且返回当前权重。
func TestRebalanceCapForcesAbandonment(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		NoTradeBand: 0.03,
		MaxTurnover: 0.01,
	}
	res, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClosedAndValid(t, req, res)
	if res.Status != StatusAbandoned {
		t.Fatalf("want abandoned, got %v", res.Status)
	}
	for i := range req.Current {
		if res.Trades[i] != 0 || res.NewWeights[i] != req.Current[i] {
			t.Fatalf("must hold current: %v %v", res.Trades, res.NewWeights)
		}
	}
}

// 成本收益闸门放弃：当前权重、零交易、零换手与成本。
func TestRebalanceGateAbandonmentReturnsCurrent(t *testing.T) {
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
	if res.Status != StatusAbandoned {
		t.Fatalf("want abandoned on gate, got %v", res.Status)
	}
	assertClosedAndValid(t, req, res)
	for i := range req.Current {
		if res.Trades[i] != 0 || res.NewWeights[i] != req.Current[i] {
			t.Fatalf("gate abandonment returns current weights: %v", res.Trades)
		}
	}
	if res.Turnover != 0 || res.Cost != 0 {
		t.Fatalf("abandonment reports zero turnover/cost, got %v/%v", res.Turnover, res.Cost)
	}
	if !hasDiag(res, DiagRebalanceAbandoned) {
		t.Fatalf("expected abandoned diagnostic: %+v", res.Diagnostics)
	}
}

// 非法输入：每类错误返回可区分原因，状态 invalid-input。
func TestRebalanceInvalidInputs(t *testing.T) {
	cases := []struct {
		name    string
		req     RebalanceRequest
		wantSub string
	}{
		{
			name:    "length mismatch",
			req:     RebalanceRequest{Current: []float64{0.6, 0.4}, Target: []float64{0.5, 0.3, 0.2}},
			wantSub: "same non-zero length",
		},
		{
			name:    "current non-finite",
			req:     RebalanceRequest{Current: []float64{math.NaN(), 0.4, 0.6}, Target: []float64{0.4, 0.3, 0.3}},
			wantSub: "current weight 0 is not finite",
		},
		{
			name:    "target infinite",
			req:     RebalanceRequest{Current: []float64{0.6, 0.4}, Target: []float64{math.Inf(1), -math.Inf(1)}},
			wantSub: "target weight 0 is not finite",
		},
		{
			name:    "current sum not one",
			req:     RebalanceRequest{Current: []float64{0.6, 0.5}, Target: []float64{0.5, 0.5}},
			wantSub: "current weights sum",
		},
		{
			name:    "target sum not one",
			req:     RebalanceRequest{Current: []float64{0.5, 0.5}, Target: []float64{0.6, 0.5}},
			wantSub: "target weights sum",
		},
		{
			name:    "negative band",
			req:     RebalanceRequest{Current: []float64{0.5, 0.5}, Target: []float64{0.6, 0.4}, NoTradeBand: -0.01},
			wantSub: "NoTradeBand must not be negative",
		},
		{
			name:    "negative turnover",
			req:     RebalanceRequest{Current: []float64{0.5, 0.5}, Target: []float64{0.6, 0.4}, MaxTurnover: -1},
			wantSub: "MaxTurnover must not be negative",
		},
		{
			name:    "negative min benefit",
			req:     RebalanceRequest{Current: []float64{0.5, 0.5}, Target: []float64{0.6, 0.4}, MinBenefit: -1},
			wantSub: "MinBenefit must not be negative",
		},
		{
			name:    "negative cost",
			req:     RebalanceRequest{Current: []float64{0.5, 0.5}, Target: []float64{0.6, 0.4}, Costs: CostModel{BuyCost: -0.1}},
			wantSub: "BuyCost must be a finite non-negative",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := Rebalance(context.Background(), tc.req, nil)
			if err == nil {
				t.Fatalf("expected error")
			}
			if res.Status != StatusInvalidInput {
				t.Fatalf("want invalid-input status, got %v", res.Status)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// 请求不被修改：输入切片在调用后逐位保持。
func TestRebalanceDoesNotMutateRequest(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		Costs:       CostModel{BuyCost: 0.001, SellCost: 0.002},
		NoTradeBand: 0.03,
		MaxTurnover: 0.05,
	}
	cur0 := cloneVec(req.Current)
	tar0 := cloneVec(req.Target)
	if _, err := Rebalance(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	for i := range cur0 {
		if req.Current[i] != cur0[i] || req.Target[i] != tar0[i] {
			t.Fatalf("request mutated: current=%v target=%v", req.Current, req.Target)
		}
	}
}

// 相同输入逐位一致（含被抑制/重闭合/换手叠加路径）。
func TestRebalanceBitIdentical(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		Costs:       CostModel{BuyCost: 0.001, SellCost: 0.002},
		NoTradeBand: 0.03,
		MaxTurnover: 0.05,
	}
	r1, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 20; k++ {
		r2, err := Rebalance(context.Background(), req, nil)
		if err != nil {
			t.Fatal(err)
		}
		if r1.Status != r2.Status || r1.Turnover != r2.Turnover || r1.Cost != r2.Cost || r1.Benefit != r2.Benefit {
			t.Fatalf("non-deterministic scalars at iteration %d", k)
		}
		for i := range r1.Trades {
			if r1.Trades[i] != r2.Trades[i] || r1.NewWeights[i] != r2.NewWeights[i] {
				t.Fatalf("non-bit-identical output at iteration %d", k)
			}
		}
		if r1.Reason != r2.Reason || len(r1.Diagnostics) != len(r2.Diagnostics) {
			t.Fatalf("non-deterministic explanation at iteration %d", k)
		}
	}
}

// 并发再平衡：结果与单次执行一致，-race 下互不污染。
func TestRebalanceConcurrent(t *testing.T) {
	req := RebalanceRequest{
		Current:     []float64{0.6, 0.3, 0.1},
		Target:      []float64{0.65, 0.28, 0.07},
		Costs:       CostModel{BuyCost: 0.001, SellCost: 0.002},
		NoTradeBand: 0.03,
		MaxTurnover: 0.05,
	}
	want, err := Rebalance(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}

	other := RebalanceRequest{
		Current:     []float64{0.25, 0.25, 0.25, 0.25},
		Target:      []float64{0.7, 0.1, 0.1, 0.1},
		Costs:       CostModel{BuyCost: 0.001, SellCost: 0.001},
		NoTradeBand: 0.02,
		MaxTurnover: 0.3,
	}

	const workers = 64
	const iterations = 40
	var wg sync.WaitGroup
	errCh := make(chan error, workers*iterations)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for k := 0; k < iterations; k++ {
				r := req
				if (id+k)%2 == 0 {
					r = other
				}
				got, gerr := Rebalance(context.Background(), r, nil)
				if gerr != nil {
					errCh <- gerr
					return
				}
				ref := want
				if (id+k)%2 == 0 {
					// other request: check closure instead of equality
					assertConcurrentValid(r, got, errCh)
					continue
				}
				if got.Status != ref.Status {
					errCh <- errors.New("status mismatch under concurrency")
					return
				}
				for i := range got.Trades {
					if got.Trades[i] != ref.Trades[i] || got.NewWeights[i] != ref.NewWeights[i] {
						errCh <- errors.New("output mismatch under concurrency")
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	// 输入未被污染
	if req.Current[0] != 0.6 || other.Target[0] != 0.7 {
		t.Fatalf("shared requests mutated")
	}
}

func assertConcurrentValid(req RebalanceRequest, res RebalanceResult, errCh chan<- error) {
	var sum, to float64
	for i, d := range res.Trades {
		sum += res.NewWeights[i]
		to += math.Abs(d)
		if math.Abs(res.NewWeights[i]-(req.Current[i]+d)) > 1e-9 {
			errCh <- errors.New("concurrent weight inconsistency")
			return
		}
	}
	if math.Abs(sum-1) > 1e-9*float64(len(req.Current)) {
		errCh <- errors.New("concurrent result not closed")
	}
	if req.MaxTurnover > 0 && to > req.MaxTurnover+1e-9*float64(len(req.Current)) {
		errCh <- errors.New("concurrent turnover cap violated")
	}
}
