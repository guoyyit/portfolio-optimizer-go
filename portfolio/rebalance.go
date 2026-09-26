package portfolio

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// rebalance.go: trade generation under transaction costs and turnover caps.
//
// The procedure is a deterministic, fully explainable heuristic:
//  1. desired trades = target - current.
//  2. Project onto {sum t = 0, sum |t| <= MaxTurnover} so the portfolio
//     stays fully invested and the turnover cap is a hard constraint.
//  3. Suppress trades below the no-trade band (avoids meaningless churn)
//     and renormalize so trades still sum to zero.
//  4. Cost gate: drop trades whose proportional cost exceeds the
//     weight-distance benefit (costRate * CostAversion >= 1), then
//     renormalize again.
// Every dropped or scaled trade is accounted for in the Explanation.

// RebalanceRequest describes a rebalancing task.
type RebalanceRequest struct {
	Current      []float64 // current weights (should sum to 1)
	Target       []float64 // desired target weights (should sum to 1)
	CostRate     []float64 // per-asset proportional cost; nil => UniformCost
	UniformCost  float64   // used when CostRate is nil
	MaxTurnover  float64   // sum |trade| cap; 0 = no cap
	NoTradeBand  float64   // suppress per-asset trades smaller than this
	CostAversion float64   // benefit-vs-cost trade-off; 0 => 1
}

// RebalanceResult is the trade list plus cost accounting.
type RebalanceResult struct {
	Trades      []float64 // signed weight changes; Current+Trades = NewWeights
	NewWeights  []float64
	Turnover    float64 // sum |trade|
	Cost        float64 // sum |trade_i| * rate_i
	Benefit     float64 // reduction in L1 distance to target
	NetBenefit  float64 // Benefit - CostAversion*Cost
	Skipped     []int   // indices suppressed by band or cost gate
	Explanation string
}

// projectTradeSet projects t onto {sum t = 0, sum |t| <= cap} via Dykstra.
// Convergence is judged on the full state (point and increments).
func projectTradeSet(t []float64, cap float64) {
	n := len(t)
	if cap < 0 {
		cap = 0
	}
	zero := make([]float64, n)
	p1 := make([]float64, n) // increment for sum-to-zero set
	p2 := make([]float64, n) // increment for L1 ball set
	cur := make([]float64, n)
	copy(cur, t)
	prev := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for cycle := 0; cycle < projMaxCycle; cycle++ {
		copy(prev, cur)
		maxInc := 0.0
		// Set 1: sum t = 0 (subtract the mean).
		for i := 0; i < n; i++ {
			y[i] = cur[i] + p1[i]
		}
		copy(z, y)
		mean := 0.0
		for i := 0; i < n; i++ {
			mean += y[i]
		}
		mean /= float64(n)
		for i := 0; i < n; i++ {
			y[i] -= mean
			ni := z[i] - y[i]
			if d := math.Abs(ni - p1[i]); d > maxInc {
				maxInc = d
			}
			p1[i] = ni
		}
		copy(cur, y)
		// Set 2: L1 ball around 0 with radius cap.
		for i := 0; i < n; i++ {
			y[i] = cur[i] + p2[i]
		}
		copy(z, y)
		projectL1Ball(y, zero, cap)
		for i := 0; i < n; i++ {
			ni := z[i] - y[i]
			if d := math.Abs(ni - p2[i]); d > maxInc {
				maxInc = d
			}
			p2[i] = ni
		}
		copy(cur, y)
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			if d := math.Abs(cur[i] - prev[i]); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff < projTol && maxInc < projTol {
			break
		}
	}
	copy(t, cur)
}

// l1Distance returns sum |a - b|.
func l1Distance(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += math.Abs(a[i] - b[i])
	}
	return s
}

// zeroSumRestore adjusts trades to sum to zero by scaling the surviving
// (non-suppressed) trades proportionally to their magnitude.
func zeroSumRestore(trades []float64, alive []bool) {
	var sum, totalAbs float64
	for i, t := range trades {
		if alive[i] {
			sum += t
			totalAbs += math.Abs(t)
		}
	}
	if totalAbs == 0 || sum == 0 {
		return
	}
	for i, t := range trades {
		if alive[i] {
			trades[i] = t - sum*math.Abs(t)/totalAbs
		}
	}
}

// Rebalance computes a feasible trade list. Pure and deterministic.
func Rebalance(req RebalanceRequest) (RebalanceResult, error) {
	var res RebalanceResult
	n := len(req.Current)
	if n == 0 {
		return res, errors.New("portfolio: empty current weights")
	}
	if len(req.Target) != n {
		return res, errors.New("portfolio: target/current length mismatch")
	}
	if req.CostRate != nil && len(req.CostRate) != n {
		return res, errors.New("portfolio: cost rate length mismatch")
	}
	if req.MaxTurnover < 0 || req.NoTradeBand < 0 {
		return res, errors.New("portfolio: negative turnover cap or band")
	}
	for i := 0; i < n; i++ {
		if math.IsNaN(req.Current[i]) || math.IsNaN(req.Target[i]) ||
			math.IsInf(req.Current[i], 0) || math.IsInf(req.Target[i], 0) {
			return res, errors.New("portfolio: non-finite weight in request")
		}
	}
	rate := req.CostRate
	if rate == nil {
		rate = make([]float64, n)
		for i := range rate {
			rate[i] = req.UniformCost
		}
	}
	costAversion := req.CostAversion
	if costAversion == 0 {
		costAversion = 1
	}

	var notes []string
	skipped := make([]bool, n)

	// 1-2. Desired trades projected onto {sum 0, L1 <= cap}.
	trades := make([]float64, n)
	for i := 0; i < n; i++ {
		trades[i] = req.Target[i] - req.Current[i]
	}
	if req.MaxTurnover > 0 {
		before := l1Distance(trades, make([]float64, n))
		projectTradeSet(trades, req.MaxTurnover)
		if after := l1Distance(trades, make([]float64, n)); after < before-1e-15 {
			notes = append(notes, fmt.Sprintf(
				"trades scaled to respect turnover cap %.4f (was %.4f)", req.MaxTurnover, before))
		}
	} else {
		// Still enforce sum-to-zero exactly.
		projectTradeSet(trades, math.MaxFloat64)
	}

	// 3. No-trade band.
	if req.NoTradeBand > 0 {
		alive := make([]bool, n)
		anySuppressed := false
		for i, t := range trades {
			if math.Abs(t) < req.NoTradeBand {
				if t != 0 {
					skipped[i] = true
					anySuppressed = true
				}
				trades[i] = 0
			} else {
				alive[i] = true
			}
		}
		if anySuppressed {
			zeroSumRestore(trades, alive)
			notes = append(notes, fmt.Sprintf(
				"%d dust trades below band %.4f suppressed", countTrue(skipped), req.NoTradeBand))
		}
	}

	// 4. Cost gate: drop trades whose cost outweighs the distance benefit.
	{
		alive := make([]bool, n)
		dropped := 0
		for i, t := range trades {
			if t == 0 {
				continue
			}
			if rate[i]*costAversion >= 1 {
				trades[i] = 0
				skipped[i] = true
				dropped++
			} else {
				alive[i] = true
			}
		}
		if dropped > 0 {
			zeroSumRestore(trades, alive)
			notes = append(notes, fmt.Sprintf(
				"%d trades dropped: proportional cost exceeds distance benefit", dropped))
		}
	}

	// Assemble result.
	newW := make([]float64, n)
	var turnover, cost float64
	for i := 0; i < n; i++ {
		newW[i] = req.Current[i] + trades[i]
		turnover += math.Abs(trades[i])
		cost += math.Abs(trades[i]) * rate[i]
	}
	benefit := l1Distance(req.Current, req.Target) - l1Distance(newW, req.Target)
	res.Trades = trades
	res.NewWeights = newW
	res.Turnover = turnover
	res.Cost = cost
	res.Benefit = benefit
	res.NetBenefit = benefit - costAversion*cost
	for i := range skipped {
		if skipped[i] {
			res.Skipped = append(res.Skipped, i)
		}
	}
	if len(notes) == 0 {
		res.Explanation = fmt.Sprintf(
			"full rebalance toward target: turnover %.4f, cost %.6f, distance-to-target reduced by %.4f",
			turnover, cost, benefit)
	} else {
		res.Explanation = strings.Join(notes, "; ") + fmt.Sprintf(
			"; final turnover %.4f, cost %.6f, benefit %.4f", turnover, cost, benefit)
	}
	return res, nil
}

func countTrue(b []bool) int {
	c := 0
	for _, v := range b {
		if v {
			c++
		}
	}
	return c
}
