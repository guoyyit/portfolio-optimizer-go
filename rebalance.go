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
// cost and turnover budgets. Fully deterministic.
func Rebalance(ctx context.Context, req RebalanceRequest, est *Estimates) (RebalanceResult, error) {
	n := len(req.Current)
	if n == 0 || len(req.Target) != n {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "current/target dimension mismatch"},
			errors.New("portfolio: rebalance dimension mismatch")
	}
	if hasNonFinite(req.Current) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "current weights contain non-finite values"},
			errors.New("portfolio: current weights contain NaN or +/-Inf")
	}
	if hasNonFinite(req.Target) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "target weights contain non-finite values"},
			errors.New("portfolio: target weights contain NaN or +/-Inf")
	}
	if s := sumVec(req.Current); math.Abs(s-1) > rebalanceWeightTol*float64(n) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: fmt.Sprintf("current weights sum %.12g is not 1", s)},
			fmt.Errorf("portfolio: current weights sum %.12g != 1", s)
	}
	if s := sumVec(req.Target); math.Abs(s-1) > rebalanceWeightTol*float64(n) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: fmt.Sprintf("target weights sum %.12g is not 1", s)},
			fmt.Errorf("portfolio: target weights sum %.12g != 1", s)
	}
	switch {
	case req.MaxTurnover < 0:
		return RebalanceResult{Status: StatusInvalidInput, Reason: "negative MaxTurnover"},
			errors.New("portfolio: negative MaxTurnover")
	case req.NoTradeBand < 0:
		return RebalanceResult{Status: StatusInvalidInput, Reason: "negative NoTradeBand"},
			errors.New("portfolio: negative NoTradeBand")
	case req.MinBenefit < 0:
		return RebalanceResult{Status: StatusInvalidInput, Reason: "negative MinBenefit"},
			errors.New("portfolio: negative MinBenefit")
	case req.Costs.BuyCost < 0 || req.Costs.SellCost < 0:
		return RebalanceResult{Status: StatusInvalidInput, Reason: "negative transaction cost rate"},
			errors.New("portfolio: negative transaction cost rate")
	}
	if math.IsNaN(req.MaxTurnover) || math.IsInf(req.MaxTurnover, 0) ||
		math.IsNaN(req.NoTradeBand) || math.IsInf(req.NoTradeBand, 0) ||
		math.IsNaN(req.MinBenefit) || math.IsInf(req.MinBenefit, 0) ||
		math.IsNaN(req.Costs.BuyCost) || math.IsInf(req.Costs.BuyCost, 0) ||
		math.IsNaN(req.Costs.SellCost) || math.IsInf(req.Costs.SellCost, 0) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "parameters contain non-finite values"},
			errors.New("portfolio: rebalance parameters contain NaN or +/-Inf")
	}
	if est != nil && (len(est.Mu) != n || len(est.Cov) != n) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "estimates dimension mismatch"},
			errors.New("portfolio: estimates dimension mismatch")
	}
	if est != nil {
		for i := range est.Cov {
			if len(est.Cov[i]) != n || hasNonFinite(est.Cov[i]) {
				return RebalanceResult{Status: StatusInvalidInput, Reason: "covariance malformed or contains non-finite values"},
					errors.New("portfolio: covariance malformed or contains NaN or +/-Inf")
			}
		}
		if hasNonFinite(est.Mu) {
			return RebalanceResult{Status: StatusInvalidInput, Reason: "mu contains non-finite values"},
				errors.New("portfolio: mu contains NaN or +/-Inf")
		}
	}
	if err := ctx.Err(); err != nil {
		return RebalanceResult{Status: StatusTimeout, Reason: "cancelled"}, nil
	}

	delta := make([]float64, n)
	for i := range delta {
		delta[i] = req.Target[i] - req.Current[i]
	}

	var diags []Diagnostic

	trades, resolveStatus, resolveDiags, err := resolveRebalanceTrades(delta, req.NoTradeBand, req.MaxTurnover)
	diags = append(diags, resolveDiags...)
	if err != nil {
		return RebalanceResult{Status: StatusInfeasible, Reason: err.Error(), Diagnostics: diags}, err
	}
	var turnover, cost float64
	for _, d := range trades {
		turnover += math.Abs(d)
		if d > 0 {
			cost += d * req.Costs.BuyCost
		} else if d < 0 {
			cost += -d * req.Costs.SellCost
		}
	}
	if turnover == 0 {
		reason := "portfolio already at target; no trades needed"
		if len(resolveStatus.suppressed) > 0 {
			diags = append(diags, Diagnostic{
				Flag: DiagNoTradeZone,
				Message: fmt.Sprintf("%d trade(s) below no-trade band %.4g suppressed at assets %v",
					len(resolveStatus.suppressed), req.NoTradeBand, resolveStatus.suppressed),
			})
			switch {
			case resolveStatus.turnoverBlocked:
				diags = append(diags, Diagnostic{
					Flag:    DiagTurnoverCapped,
					Message: fmt.Sprintf("turnover cap %.4g too small for any band-compliant round trip", req.MaxTurnover),
				})
				reason = fmt.Sprintf(
					"rebalance abandoned: turnover cap %.4g is below the minimum round-trip turnover compatible with the no-trade band %.4g (suppressed assets %v); holding current allocation",
					req.MaxTurnover, req.NoTradeBand, resolveStatus.suppressed)
			case resolveStatus.oneSided:
				reason = fmt.Sprintf(
					"rebalance abandoned: %d asset(s) %v below the no-trade band %.4g left trades on only one side, so the budget could not be re-closed to 1 without violating the band; holding current allocation",
					len(resolveStatus.suppressed), resolveStatus.suppressed, req.NoTradeBand)
			default:
				reason = fmt.Sprintf(
					"rebalance abandoned: %d asset(s) %v had trades below the no-trade band %.4g and no offsetting trades could be re-closed within band and turnover limits; holding current allocation",
					len(resolveStatus.suppressed), resolveStatus.suppressed, req.NoTradeBand)
			}
			diags = append(diags, Diagnostic{Flag: DiagRebalanceAbandoned, Message: "no trades executed; current allocation retained"})
		}
		return validatedRebalance(RebalanceResult{
			Status:      StatusOptimal,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Turnover:    0,
			Cost:        0,
			Reason:      reason,
			Diagnostics: diags,
		}, req)
	}

	// benefit: reduction in tracking error vs target (or L1 distance when
	// no estimates are available)
	var benefit float64
	benefitDesc := ""
	if est != nil {
		teCur := trackingError(est.Cov, req.Current, req.Target)
		newW := make([]float64, n)
		for i := range newW {
			newW[i] = req.Current[i] + trades[i]
		}
		teNew := trackingError(est.Cov, newW, req.Target)
		benefit = teCur - teNew
		benefitDesc = fmt.Sprintf("tracking-error-to-target reduction %.6g", benefit)
	} else {
		var dCur, dNew float64
		for i := 0; i < n; i++ {
			dCur += math.Abs(req.Current[i] - req.Target[i])
			dNew += math.Abs(req.Current[i] + trades[i] - req.Target[i])
		}
		benefit = dCur - dNew
		benefitDesc = fmt.Sprintf("distance-to-target reduction %.6g", benefit)
	}

	// cost/benefit gate
	newWeights := make([]float64, n)
	for i := range newWeights {
		newWeights[i] = req.Current[i] + trades[i]
	}
	threshold := req.MinBenefit * cost
	if benefit <= 0 {
		gateDiags := append(diags,
			Diagnostic{Flag: DiagNoTradeZone, Message: "no positive benefit, trades skipped"},
			Diagnostic{Flag: DiagRebalanceAbandoned, Message: "no trades executed; current allocation retained"})
		return validatedRebalance(RebalanceResult{
			Status:      StatusOptimal,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Turnover:    0,
			Cost:        0,
			Benefit:     benefit,
			Reason:      fmt.Sprintf("no benefit in moving toward target (%s); holding current allocation", benefitDesc),
			Diagnostics: gateDiags,
		}, req)
	}
	if req.MinBenefit > 0 && benefit < threshold {
		gateDiags := append(diags,
			Diagnostic{Flag: DiagNoTradeZone, Message: "benefit below cost threshold, trades skipped"},
			Diagnostic{Flag: DiagRebalanceAbandoned, Message: "no trades executed; current allocation retained"})
		return validatedRebalance(RebalanceResult{
			Status:     StatusOptimal,
			Trades:     make([]float64, n),
			NewWeights: cloneVec(req.Current),
			Turnover:   0,
			Cost:       0,
			Benefit:    benefit,
			Reason: fmt.Sprintf("benefit %.6g below %.3g x cost %.6g; staying in no-trade zone to avoid churn",
				benefit, req.MinBenefit, cost),
			Diagnostics: gateDiags,
		}, req)
	}

	reason := fmt.Sprintf("executing turnover %.4f (cost %.6g): %s exceeds cost threshold %.6g",
		turnover, cost, benefitDesc, threshold)
	reason += "; " + resolveStatus.explanation
	if resolveStatus.partial {
		if resolveStatus.capped {
			reason += "; partial rebalance due to turnover cap"
		} else {
			reason += "; partial rebalance due to no-trade-band suppressions"
		}
	}
	status := StatusOptimal
	if resolveStatus.partial {
		status = StatusSuboptimal
	}
	return validatedRebalance(RebalanceResult{
		Status:      status,
		Trades:      trades,
		NewWeights:  newWeights,
		Turnover:    turnover,
		Cost:        cost,
		Benefit:     benefit,
		Reason:      reason,
		Diagnostics: diags,
	}, req)
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

// rebalanceWeightTol is the per-element floating-point tolerance for
// weight-sum and constraint validation (matching the hard-constraint
// convention used by the optimizer).
const rebalanceWeightTol = 1e-12

func sumVec(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s
}

func hasNonFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return true
		}
	}
	return false
}

// rebalanceResolveInfo explains how desired deltas were turned into
// executable, budget-closing trades.
type rebalanceResolveInfo struct {
	suppressed      []int // assets held at current weight (band or cascade)
	oneSided        bool  // abandoned: trades remained on only one side
	turnoverBlocked bool  // abandoned: cap too small for a compliant round trip
	capped          bool  // executed trades were scaled to the turnover cap
	partial         bool  // trades executed but target not fully reached
	explanation     string
}

// resolveRebalanceTrades converts desired target-current deltas into trades
// that (a) keep every executed trade at least band in magnitude, (b) have
// total buys exactly equal to total sells so weights re-close to budget 1,
// and (c) respect the turnover cap. Suppressing a trade breaks closure: the
// imbalance is removed by scaling the larger trade side down (never up,
// never past target), which can cascade into further suppressions.
func resolveRebalanceTrades(desired []float64, band, maxTurnover float64) ([]float64, rebalanceResolveInfo, []Diagnostic, error) {
	n := len(desired)
	info := rebalanceResolveInfo{}
	// active[i] is the current proposed trade; 0 means inactive. Indexed
	// iteration is deterministic (map iteration in Go is not).
	active := make([]float64, n)
	suppressed := make([]bool, n)
	activeCount := 0
	for i, d := range desired {
		if band > 0 && math.Abs(d) < band {
			suppressed[i] = true
			continue
		}
		if d != 0 {
			active[i] = d
			activeCount++
		}
	}

	capScaledLastIter := false
	var trades []float64
	for {
		var buySum, sellSum float64
		for i := 0; i < n; i++ {
			d := active[i]
			if d > 0 {
				buySum += d
			} else if d < 0 {
				sellSum += -d
			}
		}

		if activeCount == 0 || buySum == 0 || sellSum == 0 {
			// Cannot close the budget without violating the band: give up.
			trades = make([]float64, n)
			info.suppressed = indicesWhere(suppressed)
			info.oneSided = activeCount > 0
			info.turnoverBlocked = capScaledLastIter
			return trades, info, nil, nil
		}

		// Close the buy/sell imbalance by trimming the larger side.
		sideScale := 1.0
		scaledSide := 0 // +1 buys trimmed, -1 sells trimmed
		if buySum > sellSum {
			sideScale = sellSum / buySum
			scaledSide = +1
		} else if sellSum > buySum {
			sideScale = buySum / sellSum
			scaledSide = -1
		}
		if math.Abs(sideScale-1) < 1e-12 {
			sideScale = 1 // only floating-point noise: keep deltas untouched
		}
		if sideScale < 1 {
			for i := 0; i < n; i++ {
				if scaledSide == +1 && active[i] > 0 {
					active[i] *= sideScale
				}
				if scaledSide == -1 && active[i] < 0 {
					active[i] *= sideScale
				}
			}
		}

		// Respect the turnover cap: closed turnover is buys+sells = 2*min.
		var turnover float64
		for _, d := range active {
			turnover += math.Abs(d)
		}
		capScaledLastIter = false
		if maxTurnover > 0 && turnover > maxTurnover+1e-12 {
			s := maxTurnover / turnover
			for i := range active {
				active[i] *= s
			}
			capScaledLastIter = true
			info.capped = true
		}

		// Scaling can push trades back inside the band: suppress and retry.
		dropped := 0
		for i, d := range active {
			if d != 0 && band > 0 && math.Abs(d) < band {
				active[i] = 0
				suppressed[i] = true
				dropped++
			}
		}
		if dropped > 0 {
			activeCount -= dropped
			continue
		}

		trades = cloneVec(active)
		info.suppressed = indicesWhere(suppressed)
		info.partial = len(info.suppressed) > 0 || info.capped || sideScale < 1

		var parts []string
		if len(info.suppressed) > 0 {
			parts = append(parts, fmt.Sprintf(
				"asset(s) %v kept at current weight because trades were below no-trade band %.4g",
				info.suppressed, band))
		}
		if sideScale < 1 {
			side := "buy"
			if scaledSide == -1 {
				side = "sell"
			}
			parts = append(parts, fmt.Sprintf(
				"the %s side was trimmed by %.4f so total buys equal total sells and weights re-close to 1",
				side, sideScale))
		}
		if info.capped {
			parts = append(parts, fmt.Sprintf("trades were scaled to turnover cap %.4f", maxTurnover))
		}
		if len(parts) == 0 {
			parts = append(parts, "all assets trade directly to target; weights sum to 1")
		}
		info.explanation = strings.Join(parts, "; ")

		var diags []Diagnostic
		if len(info.suppressed) > 0 {
			diags = append(diags, Diagnostic{
				Flag: DiagNoTradeZone,
				Message: fmt.Sprintf("%d trade(s) below no-trade band %.4g suppressed at assets %v; remaining trades re-closed to budget 1",
					len(info.suppressed), band, info.suppressed),
			})
		}
		if info.capped {
			diags = append(diags, Diagnostic{
				Flag:    DiagTurnoverCapped,
				Message: fmt.Sprintf("trades scaled down to respect turnover cap %.4f while keeping the budget closed", maxTurnover),
			})
		}
		return trades, info, diags, nil
	}
}

func indicesWhere(mark []bool) []int {
	out := make([]int, 0, len(mark))
	for i, v := range mark {
		if v {
			out = append(out, i)
		}
	}
	return out
}

// validatedRebalance is the single exit validator: every returned weight
// vector is checked for budget closure and hard constraints before leaving
// Rebalance. Violations are reported as StatusInfeasible with an error,
// never returned as a successful result.
func validatedRebalance(res RebalanceResult, req RebalanceRequest) (RebalanceResult, error) {
	n := len(req.Current)
	tol := rebalanceWeightTol * float64(n)
	fail := func(msg string) (RebalanceResult, error) {
		return RebalanceResult{
			Status:     StatusInfeasible,
			Trades:     make([]float64, n),
			NewWeights: cloneVec(req.Current),
			Reason:     "internal validation failure: " + msg,
			Diagnostics: []Diagnostic{{
				Flag:    DiagUnstable,
				Message: "rebalance result failed validation: " + msg,
			}},
		}, errors.New("portfolio: internal rebalance validation failure: " + msg)
	}
	if len(res.Trades) != n || len(res.NewWeights) != n {
		return fail("result dimension mismatch")
	}
	var sumWeights, sumTrades, turnover float64
	for i := 0; i < n; i++ {
		tr, w := res.Trades[i], res.NewWeights[i]
		if math.IsNaN(tr) || math.IsInf(tr, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			return fail("non-finite result values")
		}
		if math.Abs(w-(req.Current[i]+tr)) > tol {
			return fail(fmt.Sprintf("new weight %d inconsistent with current+trade", i))
		}
		if w < -tol || w > 1+tol {
			return fail(fmt.Sprintf("new weight %d outside [0,1]: %v", i, w))
		}
		if req.NoTradeBand > 0 && math.Abs(tr) > 0 && math.Abs(tr) < req.NoTradeBand-tol {
			return fail(fmt.Sprintf("trade %d below no-trade band: %v", i, tr))
		}
		if tr != 0 {
			desired := req.Target[i] - req.Current[i]
			if tr*desired < 0 {
				return fail(fmt.Sprintf("trade %d points away from target", i))
			}
			if math.Abs(tr) > math.Abs(desired)+tol {
				return fail(fmt.Sprintf("trade %d overshoots target", i))
			}
		}
		sumWeights += w
		sumTrades += tr
		turnover += math.Abs(tr)
	}
	if math.Abs(sumWeights-1) > tol {
		return fail(fmt.Sprintf("new weights sum %.12g != 1", sumWeights))
	}
	if math.Abs(sumTrades) > tol {
		return fail(fmt.Sprintf("trades sum %.12g != 0", sumTrades))
	}
	if math.Abs(turnover-res.Turnover) > tol {
		return fail(fmt.Sprintf("reported turnover %.12g != computed %.12g", res.Turnover, turnover))
	}
	if req.MaxTurnover > 0 && turnover > req.MaxTurnover+tol {
		return fail(fmt.Sprintf("turnover %.12g exceeds cap %.12g", turnover, req.MaxTurnover))
	}
	return res, nil
}
