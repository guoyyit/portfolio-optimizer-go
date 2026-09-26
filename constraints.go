package portfolio

import (
	"fmt"
	"math"
)

// constraintError distinguishes malformed input from infeasible constraint
// combinations.
type constraintError struct {
	infeasible bool
	msg        string
}

func (e *constraintError) Error() string { return e.msg }

// resolved holds constraints with defaults filled for n assets.
type resolved struct {
	lo, hi   []float64
	budget   float64
	leverage float64 // 0 => unconstrained
	groups   []GroupCap
	turnover float64 // 0 => unconstrained
	current  []float64
}

// resolve fills defaults: lo=0, hi=1, Budget 0 => 1 (fully invested).
func resolve(c *Constraints, n int) *resolved {
	r := &resolved{budget: 1}
	if c == nil {
		r.lo = make([]float64, n)
		r.hi = make([]float64, n)
		for i := range r.hi {
			r.hi[i] = 1
		}
		return r
	}
	r.lo = make([]float64, n)
	r.hi = make([]float64, n)
	for i := range r.hi {
		r.hi[i] = 1
	}
	if c.MinWeight != nil {
		copy(r.lo, c.MinWeight)
	}
	if c.MaxWeight != nil {
		copy(r.hi, c.MaxWeight)
	}
	r.budget = c.Budget
	if r.budget == 0 {
		r.budget = 1
	}
	r.leverage = c.MaxLeverage
	r.groups = c.GroupCaps
	r.turnover = c.MaxTurnover
	r.current = c.CurrentWeights
	return r
}

// validateConstraints checks dimensional and logical consistency of the
// constraint set for n assets. Returns nil when usable.
func validateConstraints(c *Constraints, n int) *constraintError {
	if n <= 0 {
		return &constraintError{msg: "portfolio: zero assets"}
	}
	if c == nil {
		return nil
	}
	if c.MinWeight != nil && len(c.MinWeight) != n {
		return &constraintError{msg: fmt.Sprintf("portfolio: MinWeight has %d entries, want %d", len(c.MinWeight), n)}
	}
	if c.MaxWeight != nil && len(c.MaxWeight) != n {
		return &constraintError{msg: fmt.Sprintf("portfolio: MaxWeight has %d entries, want %d", len(c.MaxWeight), n)}
	}
	if c.MaxTurnover > 0 {
		if len(c.CurrentWeights) != n {
			return &constraintError{msg: "portfolio: MaxTurnover requires CurrentWeights of matching length"}
		}
	}
	if c.MaxTurnover < 0 || c.MaxLeverage < 0 {
		return &constraintError{msg: "portfolio: negative turnover/leverage limit"}
	}
	r := resolve(c, n)
	for i := 0; i < n; i++ {
		if r.lo[i] > r.hi[i] {
			return &constraintError{infeasible: true, msg: fmt.Sprintf("portfolio: asset %d min weight %v exceeds max %v", i, r.lo[i], r.hi[i])}
		}
	}
	var sumLo, sumHi float64
	for i := 0; i < n; i++ {
		sumLo += r.lo[i]
		sumHi += r.hi[i]
	}
	if sumLo > r.budget+1e-12 {
		return &constraintError{infeasible: true, msg: fmt.Sprintf("portfolio: sum of min weights %.6f exceeds budget %.6f", sumLo, r.budget)}
	}
	if sumHi < r.budget-1e-12 {
		return &constraintError{infeasible: true, msg: fmt.Sprintf("portfolio: sum of max weights %.6f below budget %.6f", sumHi, r.budget)}
	}
	if r.leverage > 0 && r.leverage < math.Abs(r.budget)-1e-12 {
		return &constraintError{infeasible: true, msg: fmt.Sprintf("portfolio: max leverage %.6f incompatible with budget %.6f", r.leverage, r.budget)}
	}
	for gi, g := range r.groups {
		if len(g.Members) == 0 {
			return &constraintError{msg: fmt.Sprintf("portfolio: group cap %q has no members", g.Name)}
		}
		var minSum float64
		for _, m := range g.Members {
			if m < 0 || m >= n {
				return &constraintError{msg: fmt.Sprintf("portfolio: group cap %d member index %d out of range", gi, m)}
			}
			minSum += r.lo[m]
		}
		if minSum > g.Cap+1e-12 {
			return &constraintError{infeasible: true, msg: fmt.Sprintf("portfolio: group %q cap %.6f below sum of member min weights %.6f", g.Name, g.Cap, minSum)}
		}
	}
	return nil
}

// ---- individual set projections ----

func projectBox(w, lo, hi []float64) {
	for i := range w {
		if w[i] < lo[i] {
			w[i] = lo[i]
		} else if w[i] > hi[i] {
			w[i] = hi[i]
		}
	}
}

func projectBudget(w []float64, budget float64) {
	var s float64
	for _, v := range w {
		s += v
	}
	d := (s - budget) / float64(len(w))
	for i := range w {
		w[i] -= d
	}
}

// projectL1Ball projects w onto {x : ||x||_1 <= radius} in place.
// Deterministic bisection on the soft-threshold parameter.
func projectL1Ball(w []float64, radius float64) {
	var norm float64
	for _, v := range w {
		norm += math.Abs(v)
	}
	if norm <= radius {
		return
	}
	var maxAbs float64
	for _, v := range w {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	lo, hi := 0.0, maxAbs
	for it := 0; it < 200; it++ {
		mid := 0.5 * (lo + hi)
		var s float64
		for _, v := range w {
			if a := math.Abs(v) - mid; a > 0 {
				s += a
			}
		}
		if s > radius {
			lo = mid
		} else {
			hi = mid
		}
	}
	for i, v := range w {
		a := math.Abs(v) - hi
		if a < 0 {
			a = 0
		}
		if v < 0 {
			w[i] = -a
		} else {
			w[i] = a
		}
	}
}

func projectGroup(w []float64, members []int, cap float64) {
	var s float64
	for _, m := range members {
		s += w[m]
	}
	if s <= cap {
		return
	}
	d := (s - cap) / float64(len(members))
	for _, m := range members {
		w[m] -= d
	}
}

// project returns the closest point (in Euclidean norm, Dykstra's method) to
// w that satisfies all hard constraints, and whether projection converged.
// iters <= 0 selects a default.
func project(w []float64, c *Constraints, n, iters int) ([]float64, bool) {
	r := resolve(c, n)
	if iters <= 0 {
		iters = 2000
	}
	// Build projection operators (in-place on a working copy).
	type projOp func(x []float64)
	ops := []projOp{
		func(x []float64) { projectBox(x, r.lo, r.hi) },
		func(x []float64) { projectBudget(x, r.budget) },
	}
	if r.leverage > 0 {
		ops = append(ops, func(x []float64) { projectL1Ball(x, r.leverage) })
	}
	for _, g := range r.groups {
		g := g
		ops = append(ops, func(x []float64) { projectGroup(x, g.Members, g.Cap) })
	}
	if r.turnover > 0 {
		cur := r.current
		tau := r.turnover
		ops = append(ops, func(x []float64) {
			d := make([]float64, n)
			for i := range x {
				d[i] = x[i] - cur[i]
			}
			projectL1Ball(d, tau)
			for i := range x {
				x[i] = cur[i] + d[i]
			}
		})
	}

	k := len(ops)
	corr := make([][]float64, k)
	for i := range corr {
		corr[i] = make([]float64, n)
	}
	x := cloneVec(w)
	prevX := make([]float64, n)
	prevCorr := make([][]float64, k)
	for j := range prevCorr {
		prevCorr[j] = make([]float64, n)
	}
	// Dykstra's x-sequence can plateau for many sweeps while the
	// corrections still realign, so convergence requires BOTH the
	// iterates and the corrections to be stationary.
	for it := 0; it < iters; it++ {
		copy(prevX, x)
		for j := range prevCorr {
			copy(prevCorr[j], corr[j])
		}
		for j, op := range ops {
			y := make([]float64, n)
			for i := range x {
				y[i] = x[i] + corr[j][i]
			}
			op(y) // y becomes P_j(x + p_j)
			for i := range x {
				corr[j][i] = (x[i] + corr[j][i]) - y[i]
				x[i] = y[i]
			}
		}
		var dx, dc float64
		for i := range x {
			if d := math.Abs(x[i] - prevX[i]); d > dx {
				dx = d
			}
		}
		for j := range corr {
			for i := range corr[j] {
				if d := math.Abs(corr[j][i] - prevCorr[j][i]); d > dc {
					dc = d
				}
			}
		}
		if dx < 1e-13 && dc < 1e-13 {
			break
		}
	}
	return x, satisfiesResolved(x, r, 1e-8)
}

// satisfiesResolved checks w against resolved constraints within tol.
func satisfiesResolved(w []float64, r *resolved, tol float64) bool {
	var sum, l1 float64
	for i, v := range w {
		if v < r.lo[i]-tol || v > r.hi[i]+tol {
			return false
		}
		sum += v
		l1 += math.Abs(v)
	}
	if math.Abs(sum-r.budget) > tol*math.Max(1, math.Abs(r.budget))*float64(len(w)) {
		return false
	}
	if r.leverage > 0 && l1 > r.leverage+tol*float64(len(w)) {
		return false
	}
	for _, g := range r.groups {
		var s float64
		for _, m := range g.Members {
			s += w[m]
		}
		if s > g.Cap+tol*float64(len(g.Members)) {
			return false
		}
	}
	if r.turnover > 0 {
		var t float64
		for i, v := range w {
			t += math.Abs(v - r.current[i])
		}
		if t > r.turnover+tol*float64(len(w)) {
			return false
		}
	}
	return true
}

// satisfies reports whether w satisfies all hard constraints within tol.
func satisfies(w []float64, c *Constraints, n int, tol float64) bool {
	return satisfiesResolved(w, resolve(c, n), tol)
}
