package portfolio

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
)

// CostModel describes proportional transaction costs.
type CostModel struct {
	// BuyCost/SellCost are proportional rates applied to traded notional.
	BuyCost  float64
	SellCost float64
}

// RebalanceRequest describes a rebalancing problem.
type RebalanceRequest struct {
	Current []float64 // current weights
	Target  []float64 // desired target weights
	Costs   CostModel
	// MaxTurnover caps sum(|trades|). Zero means unconstrained.
	MaxTurnover float64
	// NoTradeBand: trades with |delta| below this per-asset band are
	// suppressed to avoid meaningless churn.
	NoTradeBand float64
	// MinBenefit: rebalancing is skipped entirely when the estimated
	// benefit does not exceed MinBenefit * estimated cost.
	// Zero => trade whenever benefit is positive.
	MinBenefit float64
}

// RebalanceResult describes the trades to execute.
type RebalanceResult struct {
	Status     Status
	Trades     []float64 // signed weight deltas; + buy, - sell
	NewWeights []float64
	Turnover   float64
	Cost       float64
	Benefit    float64 // estimated improvement before costs
	// Reason explains the cost/benefit trade-off decision in plain words.
	Reason      string
	Diagnostics []Diagnostic
}

// Rebalance computes feasible trades moving Current toward Target within
// cost and turnover budgets. Fully deterministic and concurrency-safe: it
// operates on per-call copies and never mutates the request.
//
// Trades below NoTradeBand are suppressed (the asset keeps its current
// weight). Suppression breaks the buy/sell balance, so the remaining trades
// are reclosed: the larger side is scaled down uniformly until buys equal
// sells, which keeps the new weights summing to 1. Reclosure and
// turnover-cap scaling can push additional trades below the band; those are
// suppressed too and reclosure is re-applied (the cascade removes at least
// one asset per round, so it terminates after at most n rounds). Every
// returned weight vector is validated for budget closure, per-asset
// threshold compliance and turnover compliance before return.
func Rebalance(ctx context.Context, req RebalanceRequest, est *Estimates) (RebalanceResult, error) {
	n := len(req.Current)
	if n == 0 || len(req.Target) != n {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "current/target dimension mismatch"},
			errors.New("portfolio: rebalance: current and target must have the same non-zero length")
	}
	if err := validateRebalanceInputs(req, est, n); err != nil {
		return RebalanceResult{Status: StatusInvalidInput, Reason: err.Error()}, err
	}
	if err := ctx.Err(); err != nil {
		return RebalanceResult{Status: StatusTimeout, Reason: "cancelled"}, nil
	}

	res := rebalanceImpl(req, est)

	// Mandatory final validation on every path that carries weights.
	if msg := validateRebalanceResult(req, res); msg != "" {
		res.Status = StatusInfeasible
		res.Trades = nil
		res.NewWeights = nil
		res.Reason = msg
		res.Diagnostics = append(res.Diagnostics, Diagnostic{
			Flag:    DiagInvalidResult,
			Message: msg,
		})
		return res, fmt.Errorf("%w: %s", ErrInvalidRebalanceResult, msg)
	}
	return res, nil
}

// ErrInvalidRebalanceResult is returned when an internally produced
// rebalance result fails the mandatory weight-sum/constraint validation.
// It must never be observed: its presence signals an internal bug.
var ErrInvalidRebalanceResult = errors.New("portfolio: rebalance produced an invalid result")

// rebalanceTol is the numerical tolerance for weight-sum, threshold and
// turnover checks in Rebalance.
const rebalanceTol = 1e-9

// suppressionInfo records one asset whose trade was removed by the
// threshold cascade.
type suppressionInfo struct {
	index           int
	desired         float64 // Target - Current (signed desired trade)
	suppressedTrade float64 // signed effective trade at suppression time
}

// rebalanceImpl contains the unchecked algorithmic core; Rebalance
// validates its output before returning.
func rebalanceImpl(req RebalanceRequest, est *Estimates) RebalanceResult {
	n := len(req.Current)
	desired := make([]float64, n)
	anyMismatch := false
	for i := range desired {
		desired[i] = req.Target[i] - req.Current[i]
		if desired[i] != 0 {
			anyMismatch = true
		}
	}
	if !anyMismatch {
		return RebalanceResult{
			Status:     StatusOptimal,
			Trades:     make([]float64, n),
			NewWeights: cloneVec(req.Current),
			Reason:     "portfolio already exactly at target; no trades needed",
		}
	}

	active := make([]bool, n)
	for i := range desired {
		active[i] = desired[i] != 0
	}

	var suppressed []suppressionInfo
	var buyScale, sellScale, capScale = 1.0, 1.0, 1.0
	capped := false

	// Fixed point: build trades for the current active set, reclose the
	// buy/sell balance, apply the turnover cap, then remove any active
	// trade that fell below the band and repeat. At least one asset is
	// removed per iteration, so at most n iterations occur.
	for {
		trades := make([]float64, n)
		var buyTotal, sellTotal float64
		for i := 0; i < n; i++ {
			if !active[i] {
				continue
			}
			trades[i] = desired[i]
			if desired[i] > 0 {
				buyTotal += desired[i]
			} else {
				sellTotal -= desired[i]
			}
		}

		// Reclose: shrink the larger side uniformly so buys == sells.
		// Sign and the toward-target direction of every remaining trade
		// are preserved; magnitudes never exceed the desired ones.
		buyScale, sellScale = 1.0, 1.0
		switch {
		case buyTotal > sellTotal:
			buyScale = sellTotal / buyTotal
			for i := 0; i < n; i++ {
				if active[i] && trades[i] > 0 {
					trades[i] *= buyScale
				}
			}
		case sellTotal > buyTotal:
			sellScale = buyTotal / sellTotal
			for i := 0; i < n; i++ {
				if active[i] && trades[i] < 0 {
					trades[i] *= sellScale
				}
			}
		}

		var turnover float64
		for _, d := range trades {
			turnover += math.Abs(d)
		}
		capScale = 1.0
		capped = false
		if req.MaxTurnover > 0 && turnover > req.MaxTurnover {
			capScale = req.MaxTurnover / turnover
			for i := range trades {
				trades[i] *= capScale
			}
			capped = true
		}

		victim := -1
		if req.NoTradeBand > 0 {
			for i := 0; i < n; i++ {
				if active[i] && math.Abs(trades[i]) < req.NoTradeBand {
					victim = i
					break
				}
			}
		}
		if victim < 0 {
			break
		}
		active[victim] = false
		suppressed = append(suppressed, suppressionInfo{
			index:           victim,
			desired:         desired[victim],
			suppressedTrade: trades[victim],
		})
	}

	trades := make([]float64, n)
	var buyTotal, sellTotal float64
	for i := 0; i < n; i++ {
		if !active[i] {
			continue
		}
		trades[i] = desired[i]
		if desired[i] > 0 {
			buyTotal += desired[i]
		} else {
			sellTotal -= desired[i]
		}
	}
	if buyTotal > sellTotal {
		for i := 0; i < n; i++ {
			if active[i] && trades[i] > 0 {
				trades[i] *= buyScale
			}
		}
	} else if sellTotal > buyTotal {
		for i := 0; i < n; i++ {
			if active[i] && trades[i] < 0 {
				trades[i] *= sellScale
			}
		}
	}
	if capped {
		for i := range trades {
			trades[i] *= capScale
		}
	}
	// Remove the FP residual of buy==sell closure on the largest trade so
	// the new weights close the budget as exactly as float64 permits.
	snapResidualToLargest(trades)

	var diags []Diagnostic
	if len(suppressed) > 0 {
		for _, s := range suppressed {
			diags = append(diags, Diagnostic{
				Flag: DiagNoTradeZone,
				Message: fmt.Sprintf("asset %d: target trade %+.6g reduced to effective %+.6g below no-trade band %.4g; held at weight %.6g",
					s.index, s.desired, s.suppressedTrade, req.NoTradeBand, req.Current[s.index]),
			})
		}
	}
	finalReclosed := buyScale < 1-1e-15 || sellScale < 1-1e-15
	if finalReclosed {
		diags = append(diags, Diagnostic{
			Flag: DiagTradesReclosed,
			Message: fmt.Sprintf(
				"suppressed trades break buy/sell balance; remaining buys scaled by %.6f and sells by %.6f to reclose the budget",
				buyScale, sellScale),
		})
	}
	if capped {
		diags = append(diags, Diagnostic{
			Flag:    DiagTurnoverCapped,
			Message: fmt.Sprintf("all trades scaled by %.6f to respect turnover cap %.4g", capScale, req.MaxTurnover),
		})
	}

	hasTrade := false
	for _, d := range trades {
		if d != 0 {
			hasTrade = true
			break
		}
	}
	if !hasTrade {
		// Every desired trade ended below the band: full abandonment.
		msg := fmt.Sprintf("all %d desired trades below no-trade band %.4g after reclosure; no executable trade exists, holding current allocation",
			len(suppressed), req.NoTradeBand)
		diags = append(diags, Diagnostic{Flag: DiagRebalanceAbandoned, Message: msg})
		return RebalanceResult{
			Status:      StatusAbandoned,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Reason:      msg,
			Diagnostics: diags,
		}
	}

	newWeights := make([]float64, n)
	for i := range newWeights {
		newWeights[i] = req.Current[i] + trades[i]
	}

	var turnover, cost float64
	for _, d := range trades {
		turnover += math.Abs(d)
		if d > 0 {
			cost += d * req.Costs.BuyCost
		} else {
			cost += -d * req.Costs.SellCost
		}
	}

	benefit, benefitDesc := rebalanceBenefit(req, est, trades)
	threshold := req.MinBenefit * cost

	// Cost/benefit gate: abandonment returns the current weights.
	if benefit <= 0 {
		msg := fmt.Sprintf("no benefit in the feasible trades (%s); holding current allocation and executing no trades", benefitDesc)
		diags = append(diags,
			Diagnostic{Flag: DiagNoTradeZone, Message: "no positive benefit, trades skipped"},
			Diagnostic{Flag: DiagRebalanceAbandoned, Message: msg},
		)
		return RebalanceResult{
			Status:      StatusAbandoned,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Benefit:     benefit,
			Reason:      msg,
			Diagnostics: diags,
		}
	}
	if req.MinBenefit > 0 && benefit < threshold {
		msg := fmt.Sprintf("benefit %.6g below %.3g x cost %.6g; executing no trades and holding current allocation to avoid churn",
			benefit, req.MinBenefit, cost)
		diags = append(diags,
			Diagnostic{Flag: DiagNoTradeZone, Message: "benefit below cost threshold, trades skipped"},
			Diagnostic{Flag: DiagRebalanceAbandoned, Message: msg},
		)
		return RebalanceResult{
			Status:      StatusAbandoned,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Benefit:     benefit,
			Cost:        0,
			Turnover:    0,
			Reason:      msg,
			Diagnostics: diags,
		}
	}

	// Partial execution: some assets were suppressed or the cap binds.
	status := StatusOptimal
	if len(suppressed) > 0 || capped {
		status = StatusSuboptimal
	}

	parts := []string{
		fmt.Sprintf("executing turnover %.6g with proportional cost %.6g (buy %.4g/sell %.4g)",
			turnover, cost, req.Costs.BuyCost, req.Costs.SellCost),
		fmt.Sprintf("%s exceeds cost threshold %.6g", benefitDesc, threshold),
	}
	if len(suppressed) > 0 {
		idx := make([]int, len(suppressed))
		for k, s := range suppressed {
			idx[k] = s.index
		}
		parts = append(parts, fmt.Sprintf("partial execution: %d asset(s) %v held at current weight due to the no-trade band",
			len(suppressed), idx))
	}
	if finalReclosed {
		parts = append(parts, fmt.Sprintf("remaining buys/sells reclosed by factors %.4f/%.4f so new weights sum to 1",
			buyScale, sellScale))
	}
	if capped {
		parts = append(parts, fmt.Sprintf("turnover cap binding: trades scaled to %.1f%% of the feasible plan", capScale*100))
	}

	return RebalanceResult{
		Status:      status,
		Trades:      trades,
		NewWeights:  newWeights,
		Turnover:    turnover,
		Cost:        cost,
		Benefit:     benefit,
		Reason:      strings.Join(parts, "; "),
		Diagnostics: diags,
	}
}

// rebalanceBenefit estimates the improvement of current+trades over
// current: tracking-error reduction when estimates are available, L1
// distance-to-target reduction otherwise.
func rebalanceBenefit(req RebalanceRequest, est *Estimates, trades []float64) (float64, string) {
	n := len(req.Current)
	newWeights := make([]float64, n)
	for i := range newWeights {
		newWeights[i] = req.Current[i] + trades[i]
	}
	if est != nil {
		teCur := trackingError(est.Cov, req.Current, req.Target)
		teNew := trackingError(est.Cov, newWeights, req.Target)
		benefit := teCur - teNew
		return benefit, fmt.Sprintf("tracking-error-to-target reduction %.6g", benefit)
	}
	var dCur, dNew float64
	for i := 0; i < n; i++ {
		dCur += math.Abs(req.Current[i] - req.Target[i])
		dNew += math.Abs(newWeights[i] - req.Target[i])
	}
	benefit := dCur - dNew
	return benefit, fmt.Sprintf("distance-to-target reduction %.6g", benefit)
}

// snapResidualToLargest adjusts the single trade with the largest
// magnitude by -(sum of trades), removing floating-point closure residual
// while changing only one ULP-sized value at most.
func snapResidualToLargest(trades []float64) {
	var sum float64
	largest := -1
	for i, d := range trades {
		sum += d
		if d != 0 && (largest < 0 || math.Abs(d) > math.Abs(trades[largest])) {
			largest = i
		}
	}
	if largest >= 0 && sum != 0 {
		trades[largest] -= sum
	}
}

// isFinite reports whether x is neither NaN nor an infinity.
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// validateRebalanceInputs rejects malformed requests with distinguishable
// error messages.
func validateRebalanceInputs(req RebalanceRequest, est *Estimates, n int) error {
	for i, w := range req.Current {
		if !isFinite(w) {
			return fmt.Errorf("portfolio: rebalance: current weight %d is not finite", i)
		}
	}
	for i, w := range req.Target {
		if !isFinite(w) {
			return fmt.Errorf("portfolio: rebalance: target weight %d is not finite", i)
		}
	}
	sumTol := rebalanceTol * float64(n)
	var sumCur, sumTar float64
	for i := 0; i < n; i++ {
		sumCur += req.Current[i]
		sumTar += req.Target[i]
	}
	if math.Abs(sumCur-1) > sumTol {
		return fmt.Errorf("portfolio: rebalance: current weights sum to %.12g, want 1", sumCur)
	}
	if math.Abs(sumTar-1) > sumTol {
		return fmt.Errorf("portfolio: rebalance: target weights sum to %.12g, want 1", sumTar)
	}
	switch {
	case req.NoTradeBand < 0:
		return errors.New("portfolio: rebalance: NoTradeBand must not be negative")
	case req.MaxTurnover < 0:
		return errors.New("portfolio: rebalance: MaxTurnover must not be negative")
	case req.MinBenefit < 0:
		return errors.New("portfolio: rebalance: MinBenefit must not be negative")
	case !isFinite(req.Costs.BuyCost) || req.Costs.BuyCost < 0:
		return errors.New("portfolio: rebalance: BuyCost must be a finite non-negative number")
	case !isFinite(req.Costs.SellCost) || req.Costs.SellCost < 0:
		return errors.New("portfolio: rebalance: SellCost must be a finite non-negative number")
	}
	if est != nil {
		if len(est.Mu) != n {
			return fmt.Errorf("portfolio: rebalance: estimates Mu has %d entries, want %d", len(est.Mu), n)
		}
		if len(est.Cov) != n {
			return fmt.Errorf("portfolio: rebalance: estimates Cov has %d rows, want %d", len(est.Cov), n)
		}
		for i, mu := range est.Mu {
			if !isFinite(mu) {
				return fmt.Errorf("portfolio: rebalance: estimates Mu[%d] is not finite", i)
			}
		}
		for i, row := range est.Cov {
			if len(row) != n {
				return fmt.Errorf("portfolio: rebalance: estimates Cov row %d has %d columns, want %d", i, len(row), n)
			}
			for j, v := range row {
				if !isFinite(v) {
					return fmt.Errorf("portfolio: rebalance: estimates Cov[%d][%d] is not finite", i, j)
				}
			}
		}
	}
	return nil
}

// validateRebalanceResult checks every weight-bearing return path. It
// returns an empty string when the result is valid, otherwise a
// distinguishable diagnostic message.
func validateRebalanceResult(req RebalanceRequest, res RebalanceResult) string {
	n := len(req.Current)
	if res.Status == StatusInvalidInput || res.Status == StatusTimeout {
		return ""
	}
	if len(res.Trades) != n {
		return fmt.Sprintf("internal validation: trades length %d, want %d", len(res.Trades), n)
	}
	if len(res.NewWeights) != n {
		return fmt.Sprintf("internal validation: new weights length %d, want %d", len(res.NewWeights), n)
	}
	for i := 0; i < n; i++ {
		if !isFinite(res.Trades[i]) || !isFinite(res.NewWeights[i]) {
			return fmt.Sprintf("internal validation: non-finite value at asset %d", i)
		}
		if math.Abs(res.NewWeights[i]-(req.Current[i]+res.Trades[i])) > rebalanceTol {
			return fmt.Sprintf("internal validation: new weight %d inconsistent with current+trade", i)
		}
	}

	// Weight sum must close to 1 within tolerance on every path, including
	// full abandonment (current weights).
	var sum float64
	for _, w := range res.NewWeights {
		sum += w
	}
	sumTol := rebalanceTol * float64(n)
	if math.Abs(sum-1) > sumTol {
		return fmt.Sprintf("internal validation: new weights sum to %.12g, want 1", sum)
	}

	var turnover, cost float64
	for i, d := range res.Trades {
		turnover += math.Abs(d)
		if d > 0 {
			cost += d * req.Costs.BuyCost
		} else {
			cost += -d * req.Costs.SellCost
		}
		// Trades must point toward the target (or be zero).
		desired := req.Target[i] - req.Current[i]
		if d != 0 && (d > 0) != (desired > 0) {
			return fmt.Sprintf("internal validation: trade %d (%+.6g) points away from target", i, d)
		}
		// Non-zero trades must meet the no-trade band (no reclosure bypass).
		if d != 0 && math.Abs(d) < req.NoTradeBand {
			return fmt.Sprintf("internal validation: trade %d of %.6g below no-trade band %.6g", i, d, req.NoTradeBand)
		}
	}
	if math.Abs(turnover-res.Turnover) > sumTol {
		return fmt.Sprintf("internal validation: reported turnover %.12g inconsistent with trades %.12g", res.Turnover, turnover)
	}
	if math.Abs(cost-res.Cost) > sumTol {
		return fmt.Sprintf("internal validation: reported cost %.12g inconsistent with trades %.12g", res.Cost, cost)
	}
	if req.MaxTurnover > 0 && turnover > req.MaxTurnover+sumTol {
		return fmt.Sprintf("internal validation: turnover %.12g exceeds cap %.12g", turnover, req.MaxTurnover)
	}

	// Abandonment and "no trades" paths must return the current weights.
	if res.Status == StatusAbandoned || turnover == 0 {
		for i := 0; i < n; i++ {
			if res.Trades[i] != 0 || math.Abs(res.NewWeights[i]-req.Current[i]) > sumTol {
				return "internal validation: no-trade result differs from current weights"
			}
		}
	}
	return ""
}

// trackingError computes (a-b)'Cov(a-b).
func trackingError(cov [][]float64, a, b []float64) float64 {
	d := make([]float64, len(a))
	for i := range a {
		d[i] = a[i] - b[i]
	}
	v := quadForm(cov, d)
	if v < 0 {
		return 0
	}
	return v
}
