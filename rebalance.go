package portfolio

import (
	"context"
	"errors"
	"fmt"
	"math"
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
	if req.MaxTurnover < 0 || req.NoTradeBand < 0 {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "negative turnover/band"},
			errors.New("portfolio: negative turnover or no-trade band")
	}
	if est != nil && (len(est.Mu) != n || len(est.Cov) != n) {
		return RebalanceResult{Status: StatusInvalidInput, Reason: "estimates dimension mismatch"},
			errors.New("portfolio: estimates dimension mismatch")
	}
	if err := ctx.Err(); err != nil {
		return RebalanceResult{Status: StatusTimeout, Reason: "cancelled"}, nil
	}

	delta := make([]float64, n)
	for i := range delta {
		delta[i] = req.Target[i] - req.Current[i]
	}

	var diags []Diagnostic

	// suppress noise trades inside the no-trade band
	suppressed := 0
	if req.NoTradeBand > 0 {
		for i := range delta {
			if math.Abs(delta[i]) < req.NoTradeBand {
				delta[i] = 0
				suppressed++
			}
		}
		if suppressed > 0 {
			diags = append(diags, Diagnostic{
				Flag:    DiagNoTradeZone,
				Message: fmt.Sprintf("%d trades below no-trade band %.4g suppressed", suppressed, req.NoTradeBand),
			})
		}
	}

	var fullTurnover float64
	for _, d := range delta {
		fullTurnover += math.Abs(d)
	}
	if fullTurnover == 0 {
		return RebalanceResult{
			Status:      StatusOptimal,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Reason:      "portfolio already at target (within no-trade band); no trades needed",
			Diagnostics: diags,
		}, nil
	}

	// scale to turnover budget
	status := StatusOptimal
	scale := 1.0
	if req.MaxTurnover > 0 && fullTurnover > req.MaxTurnover {
		scale = req.MaxTurnover / fullTurnover
		status = StatusSuboptimal
		diags = append(diags, Diagnostic{
			Flag:    DiagTurnoverCapped,
			Message: fmt.Sprintf("trades scaled by %.4f to respect turnover cap %.4f", scale, req.MaxTurnover),
		})
	}
	trades := make([]float64, n)
	for i := range trades {
		trades[i] = delta[i] * scale
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
		return RebalanceResult{
			Status:      StatusOptimal,
			Trades:      make([]float64, n),
			NewWeights:  cloneVec(req.Current),
			Benefit:     benefit,
			Reason:      fmt.Sprintf("no benefit in moving toward target (%s); holding current allocation", benefitDesc),
			Diagnostics: append(diags, Diagnostic{Flag: DiagNoTradeZone, Message: "no positive benefit, trades skipped"}),
		}, nil
	}
	if req.MinBenefit > 0 && benefit < threshold {
		return RebalanceResult{
			Status:     StatusOptimal,
			Trades:     make([]float64, n),
			NewWeights: cloneVec(req.Current),
			Benefit:    benefit,
			Reason: fmt.Sprintf("benefit %.6g below %.3g x cost %.6g; staying in no-trade zone to avoid churn",
				benefit, req.MinBenefit, cost),
			Diagnostics: append(diags, Diagnostic{Flag: DiagNoTradeZone, Message: "benefit below cost threshold, trades skipped"}),
		}, nil
	}

	reason := fmt.Sprintf("executing turnover %.4f (cost %.6g): %s exceeds cost threshold %.6g",
		turnover, cost, benefitDesc, threshold)
	if status == StatusSuboptimal {
		reason += fmt.Sprintf("; partial rebalance (%.1f%% of full) due to turnover cap", scale*100)
	}
	return RebalanceResult{
		Status:      status,
		Trades:      trades,
		NewWeights:  newWeights,
		Turnover:    turnover,
		Cost:        cost,
		Benefit:     benefit,
		Reason:      reason,
		Diagnostics: diags,
	}, nil
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
