package portfolio

import (
	"context"
	"errors"
	"sort"
	"sync"
)

// Engine provides thread-safe portfolio construction, estimation,
// rebalancing and diagnostics. Concurrent calls never observe partially
// written state; failed or cancelled calls leave prior results intact.
type Engine struct {
	mu      sync.RWMutex
	results map[string]Result
}

// NewEngine constructs an empty Engine.
func NewEngine() *Engine {
	return &Engine{results: make(map[string]Result)}
}

// ErrNotFound is returned when querying an unknown result key.
var ErrNotFound = errors.New("portfolio: result not found")

// BuildRequest bundles everything needed to construct a portfolio from raw
// return data.
type BuildRequest struct {
	Key         string // identifier under which the result is stored
	Returns     [][]float64
	Estimator   Estimator // nil => default auto-shrinkage estimator
	Estimation  EstimationConfig
	Constraints *Constraints
	Options     Options
}

// Build estimates from returns, optimizes and atomically stores the result
// under req.Key. Safe for concurrent use: the store is updated in a single
// critical section, and a cancelled/interrupted build stores nothing.
func (e *Engine) Build(ctx context.Context, req BuildRequest) (Result, error) {
	est, err := EstimateWith(req.Returns, req.Estimator, req.Estimation)
	if err != nil {
		return Result{Status: StatusInvalidInput, Message: err.Error()}, err
	}
	res, err := Optimize(ctx, est, req.Constraints, req.Options)
	if err != nil {
		return res, err
	}
	// Interrupted builds must not leave partial writes behind.
	if ctx.Err() != nil {
		return res, ctx.Err()
	}
	e.mu.Lock()
	e.results[req.Key] = res
	e.mu.Unlock()
	return res, nil
}

// Get returns a deep copy of the stored result for key, so callers can
// never mutate shared state.
func (e *Engine) Get(key string) (Result, error) {
	e.mu.RLock()
	res, ok := e.results[key]
	e.mu.RUnlock()
	if !ok {
		return Result{}, ErrNotFound
	}
	return cloneResult(res), nil
}

// Keys lists stored result keys in deterministic (sorted) order.
func (e *Engine) Keys() []string {
	e.mu.RLock()
	keys := make([]string, 0, len(e.results))
	for k := range e.results {
		keys = append(keys, k)
	}
	e.mu.RUnlock()
	sort.Strings(keys)
	return keys
}

// Delete removes a stored result; missing keys are ignored.
func (e *Engine) Delete(key string) {
	e.mu.Lock()
	delete(e.results, key)
	e.mu.Unlock()
}

// cloneResult deep-copies a Result.
func cloneResult(r Result) Result {
	out := r
	out.Weights = cloneVec(r.Weights)
	out.Metrics.RiskContributions = cloneVec(r.Metrics.RiskContributions)
	out.Diagnostics = append([]Diagnostic(nil), r.Diagnostics...)
	return out
}
