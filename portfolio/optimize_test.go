package portfolio

import (
	"math"
	"testing"
)

// mkReturns builds a deterministic pseudo-random T x N return matrix.
func mkReturns(t, n int, seed int64) [][]float64 {
	x := seed
	next := func() float64 {
		x = x*6364136223846793005 + 1442695040888963407
		v := float64((x>>33)&0x7fffffff) / float64(0x7fffffff)
		return (v - 0.5) * 0.04
	}
	r := make([][]float64, t)
	for i := range r {
		r[i] = make([]float64, n)
		for j := range r[i] {
			r[i][j] = next() + 0.0005*float64(j)
		}
	}
	return r
}

func checkHardConstraints(t *testing.T, res Result, c Constraints, n int) {
	t.Helper()
	if len(res.Weights) != n {
		t.Fatalf("expected %d weights, got %d", n, len(res.Weights))
	}
	sum, sumAbs := 0.0, 0.0
	minW := c.MinWeight
	if !c.AllowShort && minW < 0 {
		minW = 0
	}
	maxW := c.MaxWeight
	if maxW == 0 {
		maxW = 1
	}
	for i, w := range res.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("non-finite weight at %d", i)
		}
		if w < minW-1e-8 || w > maxW+1e-8 {
			t.Fatalf("box constraint violated at %d: w=%v not in [%v,%v]", i, w, minW, maxW)
		}
		sum += w
		sumAbs += math.Abs(w)
	}
	if math.Abs(sum-1) > 1e-8 {
		t.Fatalf("weights do not sum to 1: %v", sum)
	}
	lev := c.MaxLeverage
	if lev == 0 {
		lev = 1
	}
	if sumAbs > lev+1e-8 {
		t.Fatalf("leverage violated: sum|w|=%v > %v", sumAbs, lev)
	}
	for _, g := range c.Groups {
		gs := 0.0
		for _, m := range g.Members {
			gs += res.Weights[m]
		}
		if gs > g.Max+1e-8 {
			t.Fatalf("group %q cap violated: %v > %v", g.Name, gs, g.Max)
		}
	}
}

func TestMeanVarianceBasic(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(250, 5, 7),
		Objective: ObjectiveMeanVariance,
		Estimator: EstimatorShrinkage,
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 5)
	if !res.Diagnostics.Converged {
		t.Fatal("expected convergence")
	}
}

func TestRiskParityEqualRiskContributions(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(300, 4, 11),
		Objective: ObjectiveRiskParity,
		Estimator: EstimatorSample,
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 4)
	// Verify risk contributions are approximately equal.
	_, sigma, _, err := estimate(p.Returns, p.Estimator)
	if err != nil {
		t.Fatal(err)
	}
	sw := sigma.matVec(res.Weights)
	total := 0.0
	for i := range sw {
		total += res.Weights[i] * sw[i]
	}
	target := total / 4
	for i := range sw {
		rc := res.Weights[i] * sw[i]
		if math.Abs(rc-target) > 0.05*target {
			t.Fatalf("risk contribution %d = %v, target %v", i, rc, target)
		}
	}
}

func TestBoxConstraintBoundary(t *testing.T) {
	// Tight max weight forces diversification.
	p := Problem{
		Returns:   mkReturns(200, 4, 3),
		Objective: ObjectiveMeanVariance,
		Constraints: Constraints{
			MaxWeight: 0.3,
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 4)
	for i, w := range res.Weights {
		if w > 0.3+1e-9 {
			t.Fatalf("weight %d=%v exceeds cap", i, w)
		}
	}
}

func TestGroupCapConstraint(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 5, 5),
		Objective: ObjectiveMeanVariance,
		Constraints: Constraints{
			Groups: []GroupCap{
				{Name: "equity", Members: []int{0, 1, 2}, Max: 0.4},
				{Name: "bond", Members: []int{3, 4}, Max: 0.8},
			},
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 5)
}

func TestTurnoverConstraint(t *testing.T) {
	current := []float64{0.25, 0.25, 0.25, 0.25}
	p := Problem{
		Returns:   mkReturns(200, 4, 9),
		Objective: ObjectiveMeanVariance,
		Current:   current,
		Constraints: Constraints{
			MaxTurnover: 0.1,
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal && res.Status != StatusFeasibleSub {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 4)
	to := 0.0
	for i := range res.Weights {
		to += math.Abs(res.Weights[i] - current[i])
	}
	if to > 0.1+1e-8 {
		t.Fatalf("turnover %v exceeds cap 0.1", to)
	}
	if res.Risk.Turnover > 0.1+1e-8 {
		t.Fatalf("reported turnover %v exceeds cap", res.Risk.Turnover)
	}
}

func TestLeverageWithShorts(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 4, 13),
		Objective: ObjectiveMeanVariance,
		Constraints: Constraints{
			AllowShort:  true,
			MinWeight:   -0.5,
			MaxWeight:   1.0,
			MaxLeverage: 1.5,
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusOptimal && res.Status != StatusFeasibleSub {
		t.Fatalf("status=%v msg=%s", res.Status, res.Diagnostics.Message)
	}
	checkHardConstraints(t, res, p.Constraints, 4)
}

func TestInfeasibleConstraints(t *testing.T) {
	// Group cap makes full investment impossible.
	p := Problem{
		Returns:   mkReturns(100, 3, 17),
		Objective: ObjectiveMeanVariance,
		Constraints: Constraints{
			Groups: []GroupCap{
				{Name: "all", Members: []int{0, 1, 2}, Max: 0.5},
			},
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusInfeasible {
		t.Fatalf("expected infeasible, got %v", res.Status)
	}
	if res.Weights != nil {
		t.Fatal("infeasible result must not carry weights")
	}
}

func TestInfeasibleBox(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(100, 3, 19),
		Objective: ObjectiveMeanVariance,
		Constraints: Constraints{
			MaxWeight: 0.2, // 3 * 0.2 < 1
		},
	}
	res := ConstructOnce(p)
	if res.Status != StatusInfeasible {
		t.Fatalf("expected infeasible, got %v", res.Status)
	}
}

func TestBudgetExhaustedReturnsFeasibleSuboptimal(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 6, 23),
		Objective: ObjectiveMeanVariance,
		MaxIters:  1, // deliberately starve the solver
	}
	res := ConstructOnce(p)
	if res.Status != StatusFeasibleSub {
		t.Fatalf("expected feasible-suboptimal, got %v", res.Status)
	}
	// Even starved, the returned point must satisfy all hard constraints.
	checkHardConstraints(t, res, p.Constraints, 6)
	if res.Diagnostics.Converged {
		t.Fatal("should not report convergence with 1 iteration")
	}
}

func TestDeterminism(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 5, 29),
		Objective: ObjectiveMeanVariance,
		Estimator: EstimatorShrinkage,
		Seed:      99,
	}
	r1 := ConstructOnce(p)
	r2 := ConstructOnce(p)
	if len(r1.Weights) != len(r2.Weights) {
		t.Fatal("weight length mismatch")
	}
	for i := range r1.Weights {
		if math.Float64bits(r1.Weights[i]) != math.Float64bits(r2.Weights[i]) {
			t.Fatalf("non-deterministic weight at %d: %v vs %v", i, r1.Weights[i], r2.Weights[i])
		}
	}
	if r1.Objective != r2.Objective || r1.Status != r2.Status {
		t.Fatal("non-deterministic objective/status")
	}
}

func TestInvalidInput(t *testing.T) {
	// Ragged matrix.
	res := ConstructOnce(Problem{
		Returns: [][]float64{{0.01, 0.02}, {0.01}},
	})
	if res.Status != StatusInvalidInput {
		t.Fatalf("expected invalid-input, got %v", res.Status)
	}
	// Empty.
	res = ConstructOnce(Problem{})
	if res.Status != StatusInvalidInput {
		t.Fatalf("expected invalid-input, got %v", res.Status)
	}
	// Turnover constraint without current weights.
	res = ConstructOnce(Problem{
		Returns:     mkReturns(50, 2, 31),
		Constraints: Constraints{MaxTurnover: 0.1},
	})
	if res.Status != StatusInvalidInput {
		t.Fatalf("expected invalid-input, got %v", res.Status)
	}
}

func TestTimeBudgetReturnsFeasible(t *testing.T) {
	// A tiny time budget forces early termination; the result must still be
	// feasible and explicitly flagged as suboptimal (or optimal if the solve
	// finished within the budget on a fast machine).
	p := Problem{
		Returns:    mkReturns(300, 8, 37),
		Objective:  ObjectiveMeanVariance,
		TimeBudget: 1, // 1 nanosecond: guaranteed to expire immediately
	}
	res := ConstructOnce(p)
	if res.Status != StatusFeasibleSub && res.Status != StatusOptimal {
		t.Fatalf("unexpected status %v", res.Status)
	}
	if res.Status == StatusFeasibleSub {
		checkHardConstraints(t, res, p.Constraints, 8)
	}
}

func TestRiskParityBudgetStarved(t *testing.T) {
	p := Problem{
		Returns:   mkReturns(200, 5, 41),
		Objective: ObjectiveRiskParity,
		MaxIters:  1,
	}
	res := ConstructOnce(p)
	if res.Status != StatusFeasibleSub && res.Status != StatusOptimal {
		t.Fatalf("unexpected status %v", res.Status)
	}
	checkHardConstraints(t, res, p.Constraints, 5)
}
