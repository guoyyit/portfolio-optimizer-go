package portfolio

import (
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync"
	"time"
)

// engine.go: problem validation, orchestration, and a thread-safe engine
// with deterministic result caching.

// Default solver budgets.
const (
	DefaultMaxIters = 2000
	DefaultSeed     = 42
)

// normalizeConstraints fills defaults and returns per-asset bounds.
func normalizeConstraints(c *Constraints, n int) (lo, hi []float64, err error) {
	if c.MaxWeight == 0 {
		c.MaxWeight = 1
	}
	if c.MaxLeverage == 0 {
		c.MaxLeverage = 1
	}
	if c.MaxWeight < 0 {
		return nil, nil, errors.New("portfolio: MaxWeight must be >= 0")
	}
	if c.MaxLeverage < 0 {
		return nil, nil, errors.New("portfolio: MaxLeverage must be >= 0")
	}
	if c.MaxTurnover < 0 {
		return nil, nil, errors.New("portfolio: MaxTurnover must be >= 0")
	}
	if c.MinWeight > c.MaxWeight {
		return nil, nil, errors.New("portfolio: MinWeight exceeds MaxWeight")
	}
	minW := c.MinWeight
	if !c.AllowShort && minW < 0 {
		minW = 0
	}
	lo = make([]float64, n)
	hi = make([]float64, n)
	for i := 0; i < n; i++ {
		lo[i] = minW
		hi[i] = c.MaxWeight
	}
	seen := make(map[string]bool)
	for _, g := range c.Groups {
		if g.Name == "" {
			return nil, nil, errors.New("portfolio: group cap with empty name")
		}
		if seen[g.Name] {
			return nil, nil, fmt.Errorf("portfolio: duplicate group %q", g.Name)
		}
		seen[g.Name] = true
		if len(g.Members) == 0 {
			return nil, nil, fmt.Errorf("portfolio: group %q has no members", g.Name)
		}
		for _, m := range g.Members {
			if m < 0 || m >= n {
				return nil, nil, fmt.Errorf("portfolio: group %q member index %d out of range", g.Name, m)
			}
		}
	}
	return lo, hi, nil
}

// validateProblem checks structural properties of the request.
func validateProblem(p *Problem) error {
	if len(p.Returns) == 0 {
		return errors.New("portfolio: empty return series")
	}
	n := len(p.Returns[0])
	if n == 0 {
		return errors.New("portfolio: zero assets")
	}
	for i := range p.Returns {
		if len(p.Returns[i]) != n {
			return errors.New("portfolio: ragged return matrix")
		}
	}
	if p.RiskAversion < 0 {
		return errors.New("portfolio: RiskAversion must be >= 0")
	}
	if p.MaxIters < 0 {
		return errors.New("portfolio: MaxIters must be >= 0")
	}
	if p.TimeBudget < 0 {
		return errors.New("portfolio: TimeBudget must be >= 0")
	}
	if p.Constraints.MaxTurnover > 0 && len(p.Current) != n {
		return errors.New("portfolio: turnover constraint requires Current weights of length N")
	}
	if len(p.Current) != 0 && len(p.Current) != n {
		return errors.New("portfolio: Current weights length mismatch")
	}
	return nil
}

// quickInfeasible reports a reason when the constraints are trivially
// unsatisfiable, or "" when the cheap checks pass.
func quickInfeasible(c *Constraints, lo, hi []float64) string {
	var slo, shi float64
	for i := range lo {
		slo += lo[i]
		shi += hi[i]
	}
	if slo > 1+1e-12 {
		return "sum of minimum weights exceeds 1"
	}
	if shi < 1-1e-12 {
		return "sum of maximum weights below 1"
	}
	if c.MaxLeverage < 1-1e-12 {
		return "MaxLeverage below 1 cannot fund a fully invested portfolio"
	}
	for _, g := range c.Groups {
		var glo float64
		for _, m := range g.Members {
			glo += lo[m]
		}
		if glo > g.Max+1e-12 {
			return fmt.Sprintf("group %q cap %.6g below sum of member minimums %.6g", g.Name, g.Max, glo)
		}
	}
	return ""
}

// riskMetrics computes ex-ante metrics for w.
func riskMetrics(w, mu []float64, sigma *mat, current []float64) RiskMetrics {
	sw := sigma.matVec(w)
	var ret, variance, hhi, maxW, turnover float64
	for i := range w {
		ret += mu[i] * w[i]
		variance += w[i] * sw[i]
		hhi += w[i] * w[i]
		if w[i] > maxW {
			maxW = w[i]
		}
		if current != nil {
			turnover += math.Abs(w[i] - current[i])
		}
	}
	if variance < 0 {
		variance = 0
	}
	vol := math.Sqrt(variance)
	rm := RiskMetrics{
		ExpectedReturn: ret,
		Volatility:     vol,
		Herfindahl:     hhi,
		MaxWeight:      maxW,
		Turnover:       turnover,
	}
	if hhi > 0 {
		rm.EffectiveN = 1 / hhi
	}
	if vol > 0 {
		rm.Sharpe = ret / vol
	}
	return rm
}

// solve runs the full pipeline: estimate -> solve -> metrics.
func solve(p Problem) Result {
	res := Result{Status: StatusOptimal}
	if err := validateProblem(&p); err != nil {
		res.Status = StatusInvalidInput
		res.Diagnostics.Message = err.Error()
		return res
	}
	n := len(p.Returns[0])
	c := p.Constraints
	lo, hi, err := normalizeConstraints(&c, n)
	if err != nil {
		res.Status = StatusInvalidInput
		res.Diagnostics.Message = err.Error()
		return res
	}
	if reason := quickInfeasible(&c, lo, hi); reason != "" {
		res.Status = StatusInfeasible
		res.Diagnostics.Message = "constraints unsatisfiable: " + reason
		return res
	}

	mu, sigma, diag, err := estimate(p.Returns, p.Estimator)
	if err != nil {
		res.Status = StatusInvalidInput
		res.Diagnostics = diag
		res.Diagnostics.Message = err.Error()
		return res
	}
	res.Diagnostics = diag

	pp := &projParams{
		lo:        lo,
		hi:        hi,
		leverage:  c.MaxLeverage,
		groups:    c.Groups,
		turnover:  c.MaxTurnover,
		targetSum: 1,
	}
	if c.MaxTurnover > 0 {
		pp.w0 = make([]float64, n)
		copy(pp.w0, p.Current)
	}

	maxIters := p.MaxIters
	if maxIters == 0 {
		maxIters = DefaultMaxIters
	}
	budget := solveBudget{maxIters: maxIters}
	if p.TimeBudget > 0 {
		budget.deadline = time.Now().Add(p.TimeBudget).UnixNano()
	}

	var out solveOut
	switch p.Objective {
	case ObjectiveMeanVariance:
		ra := p.RiskAversion
		if ra == 0 {
			ra = 1
		}
		out = solveMeanVariance(mu, sigma, ra, pp, budget)
	case ObjectiveRiskParity:
		out = solveRiskParity(sigma, pp, budget)
	default:
		res.Status = StatusInvalidInput
		res.Diagnostics.Message = fmt.Sprintf("portfolio: unknown objective %d", int(p.Objective))
		return res
	}

	res.Diagnostics.Iterations = out.iters
	res.Diagnostics.Converged = out.converged

	if !out.feasible {
		res.Status = StatusInfeasible
		res.Diagnostics.Message = "no weight vector satisfies all hard constraints"
		return res
	}

	res.Weights = out.w
	res.Objective = out.obj
	res.Risk = riskMetrics(out.w, mu, sigma, p.Current)

	if !out.converged {
		res.Status = StatusFeasibleSub
		res.Diagnostics.Message = fmt.Sprintf(
			"iteration/time budget exhausted after %d iterations; returning best feasible (suboptimal) solution",
			out.iters)
	} else {
		res.Diagnostics.Message = "converged"
	}
	return res
}

// ConstructOnce is the stateless, pure entry point: it validates the
// problem, estimates moments, solves, and assembles the result.
func ConstructOnce(p Problem) Result {
	return solve(p)
}

// cacheKey derives a deterministic key from the full problem content.
func cacheKey(p Problem) string {
	h := fnv.New128a()
	var buf [8]byte
	putF := func(v float64) {
		u := math.Float64bits(v)
		for i := 0; i < 8; i++ {
			buf[i] = byte(u >> (8 * i))
		}
		h.Write(buf[:])
	}
	putI := func(v int64) {
		for i := 0; i < 8; i++ {
			buf[i] = byte(v >> (8 * i))
		}
		h.Write(buf[:])
	}
	for _, row := range p.Returns {
		putI(int64(len(row)))
		for _, v := range row {
			putF(v)
		}
	}
	putI(int64(p.Objective))
	putI(int64(p.Estimator))
	putF(p.RiskAversion)
	putF(p.Constraints.MinWeight)
	putF(p.Constraints.MaxWeight)
	putF(p.Constraints.MaxLeverage)
	putF(p.Constraints.MaxTurnover)
	if p.Constraints.AllowShort {
		putI(1)
	} else {
		putI(0)
	}
	// Groups in canonical (name-sorted) order.
	gs := make([]GroupCap, len(p.Constraints.Groups))
	copy(gs, p.Constraints.Groups)
	sort.Slice(gs, func(i, j int) bool { return gs[i].Name < gs[j].Name })
	for _, g := range gs {
		h.Write([]byte(g.Name))
		putF(g.Max)
		mem := make([]int, len(g.Members))
		copy(mem, g.Members)
		sort.Ints(mem)
		for _, m := range mem {
			putI(int64(m))
		}
	}
	for _, v := range p.Current {
		putF(v)
	}
	putI(p.Seed)
	putI(int64(p.MaxIters))
	putI(int64(p.TimeBudget))
	return string(h.Sum(nil))
}

// cloneResult deep-copies a result so callers can never mutate cached state.
func cloneResult(r Result) Result {
	out := r
	if r.Weights != nil {
		out.Weights = make([]float64, len(r.Weights))
		copy(out.Weights, r.Weights)
	}
	return out
}

// Engine serializes construction and caches results for concurrent queries.
// The zero value is not ready; use NewEngine.
type Engine struct {
	mu    sync.Mutex
	cache map[string]Result
}

// NewEngine creates an empty engine.
func NewEngine() *Engine {
	return &Engine{cache: make(map[string]Result)}
}

// Construct solves the problem and returns the result. It is safe for
// concurrent use: identical problems are computed exactly once, the cache
// is updated atomically under the mutex (a panic mid-solve cannot leave a
// partial entry), and every caller receives a private copy of the result.
func (e *Engine) Construct(p Problem) (res Result, err error) {
	key := cacheKey(p)
	e.mu.Lock()
	defer e.mu.Unlock()
	if r, ok := e.cache[key]; ok {
		return cloneResult(r), nil
	}
	defer func() {
		if rec := recover(); rec != nil {
			res = Result{Status: StatusInvalidInput}
			res.Diagnostics.Message = fmt.Sprintf("portfolio: internal error: %v", rec)
			err = fmt.Errorf("portfolio: panic during construction: %v", rec)
		}
	}()
	r := solve(p)
	e.cache[key] = cloneResult(r)
	return r, nil
}

// Cached reports whether an identical problem was already solved.
func (e *Engine) Cached(p Problem) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.cache[cacheKey(p)]
	return ok
}
