// Package portfolio provides local portfolio construction, estimation,
// rebalancing and robustness diagnostics with deterministic, thread-safe
// behavior.
package portfolio

import "time"

// Objective selects the portfolio construction objective.
type Objective int

const (
	// ObjectiveMeanVariance maximizes mu'w - 0.5*riskAversion*w'Σw.
	ObjectiveMeanVariance Objective = iota
	// ObjectiveRiskParity equalizes per-asset risk contributions.
	ObjectiveRiskParity
)

// String names the objective for diagnostics.
func (o Objective) String() string {
	switch o {
	case ObjectiveMeanVariance:
		return "mean-variance"
	case ObjectiveRiskParity:
		return "risk-parity"
	default:
		return "unknown"
	}
}

// Estimator selects the covariance estimation method.
type Estimator int

const (
	// EstimatorSample uses the plain sample covariance.
	EstimatorSample Estimator = iota
	// EstimatorShrinkage uses Ledoit-Wolf style shrinkage toward a
	// constant-correlation target, guaranteeing a well-conditioned PSD matrix.
	EstimatorShrinkage
)

// GroupCap constrains the total weight of a named group of assets.
type GroupCap struct {
	Name    string
	Members []int // asset indices
	Max     float64
}

// Constraints describes hard constraints every returned weight vector
// must satisfy.
type Constraints struct {
	MinWeight   float64    // per-asset lower bound (default 0)
	MaxWeight   float64    // per-asset upper bound (default 1)
	MaxLeverage float64    // sum of |w| must be <= MaxLeverage (default 1)
	MaxTurnover float64    // sum |w - w0| <= MaxTurnover; 0 disables
	Groups      []GroupCap // per-group sum-of-weights caps
	AllowShort  bool       // if false, MinWeight is clamped to >= 0
}

// Problem is a full construction request.
type Problem struct {
	Returns      [][]float64 // T x N matrix of periodic simple returns
	Objective    Objective
	Estimator    Estimator
	RiskAversion float64 // mean-variance only; default 1
	Constraints  Constraints
	Current      []float64 // current weights for turnover constraint (len N or nil)
	Seed         int64     // random seed for stochastic components
	MaxIters     int       // solver iteration budget; 0 => default
	TimeBudget   time.Duration
}

// Status distinguishes solver outcomes.
type Status int

const (
	StatusOptimal      Status = iota // converged within budget
	StatusFeasibleSub                // budget exhausted; best feasible solution returned
	StatusInfeasible                 // no feasible solution found
	StatusInvalidInput               // malformed input
)

func (s Status) String() string {
	switch s {
	case StatusOptimal:
		return "optimal"
	case StatusFeasibleSub:
		return "feasible-suboptimal"
	case StatusInfeasible:
		return "infeasible"
	case StatusInvalidInput:
		return "invalid-input"
	default:
		return "unknown"
	}
}

// Diagnostics reports numerical-health information about the estimation
// and solve.
type Diagnostics struct {
	MissingValues      int     // count of NaN/Inf cells repaired in returns
	OutliersClipped    int     // count of cells winsorized
	CondNumber         float64 // condition number of the covariance used
	ShrunkToPSD        bool    // true if the raw covariance was not PSD
	ShrinkageIntensity float64 // shrinkage weight actually applied
	Iterations         int     // solver iterations used
	Converged          bool    // solver met its tolerance
	Message            string  // human-readable explanation
}

// RiskMetrics summarizes a portfolio's ex-ante risk/return.
type RiskMetrics struct {
	ExpectedReturn float64
	Volatility     float64
	Sharpe         float64 // 0 if volatility == 0
	Herfindahl     float64 // sum w_i^2 concentration
	EffectiveN     float64 // 1 / Herfindahl
	MaxWeight      float64
	Turnover       float64 // sum |w - current|
}

// Result is the output of a construction.
type Result struct {
	Weights     []float64
	Status      Status
	Diagnostics Diagnostics
	Risk        RiskMetrics
	Objective   float64 // achieved objective value
}
