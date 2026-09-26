package portfolio

import (
	"math"
	"testing"
)

func TestProjectBoxBudget(t *testing.T) {
	c := &Constraints{
		MinWeight: []float64{0.1, 0, 0},
		MaxWeight: []float64{0.5, 1, 1},
	}
	w := []float64{2, -1, 0.3}
	p, ok := project(w, c, 3, 0)
	if !ok {
		t.Fatal("projection failed")
	}
	if !satisfies(p, c, 3, 1e-9) {
		t.Fatalf("projected point violates constraints: %v", p)
	}
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("budget violated: %v", sum)
	}
}

func TestProjectGroupAndLeverage(t *testing.T) {
	c := &Constraints{
		MaxLeverage: 1.5,
		GroupCaps: []GroupCap{
			{Name: "eq", Members: []int{0, 1}, Cap: 0.6},
		},
	}
	w := []float64{0.9, 0.9, -0.8}
	p, ok := project(w, c, 3, 0)
	if !ok {
		t.Fatal("projection failed")
	}
	if !satisfies(p, c, 3, 1e-9) {
		t.Fatalf("violates constraints: %v", p)
	}
	if p[0]+p[1] > 0.6+1e-9 {
		t.Fatalf("group cap violated: %v", p)
	}
	var l1 float64
	for _, v := range p {
		l1 += math.Abs(v)
	}
	if l1 > 1.5+1e-9 {
		t.Fatalf("leverage violated: %v", l1)
	}
}

func TestProjectTurnover(t *testing.T) {
	c := &Constraints{
		MaxTurnover:    0.2,
		CurrentWeights: []float64{0.5, 0.5, 0},
	}
	w := []float64{0, 0, 1}
	p, ok := project(w, c, 3, 0)
	if !ok {
		t.Fatal("projection failed")
	}
	var to float64
	for i := range p {
		to += math.Abs(p[i] - c.CurrentWeights[i])
	}
	if to > 0.2+1e-9 {
		t.Fatalf("turnover violated: %v", to)
	}
	if !satisfies(p, c, 3, 1e-9) {
		t.Fatalf("violates constraints: %v", p)
	}
}

func TestInfeasibleDetection(t *testing.T) {
	// sum of min weights > budget
	c := &Constraints{MinWeight: []float64{0.6, 0.6}}
	if err := validateConstraints(c, 2); err == nil || !err.infeasible {
		t.Fatalf("expected infeasible, got %v", err)
	}
	// min > max
	c2 := &Constraints{MinWeight: []float64{0.9}, MaxWeight: []float64{0.1}}
	if err := validateConstraints(c2, 1); err == nil || !err.infeasible {
		t.Fatalf("expected infeasible, got %v", err)
	}
	// leverage incompatible with budget
	c3 := &Constraints{MaxLeverage: 0.5}
	if err := validateConstraints(c3, 2); err == nil || !err.infeasible {
		t.Fatalf("expected infeasible, got %v", err)
	}
	// malformed dims
	c4 := &Constraints{MinWeight: []float64{0.1}}
	if err := validateConstraints(c4, 3); err == nil || err.infeasible {
		t.Fatalf("expected invalid input, got %v", err)
	}
	// turnover without current weights
	c5 := &Constraints{MaxTurnover: 0.1}
	if err := validateConstraints(c5, 2); err == nil || err.infeasible {
		t.Fatalf("expected invalid input, got %v", err)
	}
}
