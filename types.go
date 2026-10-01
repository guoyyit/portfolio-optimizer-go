// Package portfolio provides local portfolio construction, estimation,
// rebalancing and robustness diagnostics with hard-constraint guarantees.
package portfolio

import "time"

// Objective selects the portfolio construction target.
type Objective int

const (
	// ObjectiveMeanVariance maximizes mu'w - riskAversion/2 * w'Sigma*w.
	ObjectiveMeanVariance Objective = iota
	// ObjectiveRiskParity equalizes per-asset risk contributions.
	ObjectiveRiskParity
)

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

// GroupCap bounds the total weight of a named group of assets.
type GroupCap struct {
	Name    string  // group identifier (for diagnostics)
	Members []int   // asset indices belonging to the group
	Cap     float64 // maximum sum of weights across Members
}

// Constraints describes hard constraints that any returned weight vector
// must satisfy exactly (within numerical tolerance).
type Constraints struct {
	MinWeight []float64 // per-asset lower bounds (nil => 0)
	MaxWeight []float64 // per-asset upper bounds (nil => 1)

	// Budget is the required sum of weights; the zero value selects the
	// default 1.0 (fully invested).
	Budget float64

	// MaxLeverage bounds the L1 norm of weights (sum of absolute values).
	// Zero means no explicit leverage limit beyond the box constraints.
	MaxLeverage float64

	// GroupCaps limit the summed weight of asset groups.
	GroupCaps []GroupCap

	// MaxTurnover bounds sum(|w - current|) relative to CurrentWeights.
	// Zero means unconstrained; requires CurrentWeights when > 0.
	MaxTurnover    float64
	CurrentWeights []float64
}

// EstimationConfig selects how expected returns and covariance are built.
type EstimationConfig struct {
	// WinsorizeLimit, when > 0, clips each asset's returns to
	// median +- WinsorizeLimit * MAD before estimation (outlier control).
	WinsorizeLimit float64
}

// Options controls solver behaviour.
type Options struct {
	Objective       Objective
	RiskAversion    float64 // mean-variance risk aversion (default 1 when <= 0)
	MaxIterations   int     // default 2000 when <= 0
	TimeBudget      time.Duration
	Tolerance       float64 // convergence tolerance (default 1e-9)
	Seed            int64   // seed for any randomized components
	ProjectionIters int     // max iterations for constraint projection (default 200)
}

// Status describes how a solve terminated.
type Status int

const (
	StatusOptimal      Status = iota // converged within tolerance
	StatusSuboptimal                 // hit iteration/time budget; best feasible point returned
	StatusInfeasible                 // constraints are mutually inconsistent
	StatusTimeout                    // time budget exhausted; best feasible point returned
	StatusInvalidInput               // malformed inputs
)

func (s Status) String() string {
	switch s {
	case StatusOptimal:
		return "optimal"
	case StatusSuboptimal:
		return "suboptimal"
	case StatusInfeasible:
		return "infeasible"
	case StatusTimeout:
		return "timeout"
	case StatusInvalidInput:
		return "invalid-input"
	default:
		return "unknown"
	}
}

// DiagFlag identifies data/numerical conditions detected during estimation
// or optimization.
type DiagFlag string

const (
	DiagMissingData        DiagFlag = "missing-data"
	DiagOutliersWinsorized DiagFlag = "outliers-winsorized"
	DiagHighDimensional    DiagFlag = "assets-exceed-samples"
	DiagShrinkageApplied   DiagFlag = "shrinkage-applied"
	DiagEigenvalueClipped  DiagFlag = "eigenvalues-clipped"
	DiagNearSingular       DiagFlag = "near-singular-covariance"
	DiagTurnoverCapped     DiagFlag = "turnover-capped"
	DiagNoTradeZone        DiagFlag = "no-trade-zone"
	DiagUnstable           DiagFlag = "unstable-allocation"
	// DiagRebalanceAbandoned marks a rebalance request that executed no
	// trades at all (all trades suppressed, the budget could not be
	// re-closed within the no-trade band, or the cost/benefit gate fired);
	// NewWeights then equal CurrentWeights.
	DiagRebalanceAbandoned DiagFlag = "rebalance-abandoned"
)

// Diagnostic is a machine-readable, human-explainable finding.
type Diagnostic struct {
	Flag    DiagFlag
	Message string
}

// RiskMetrics summarizes a portfolio.
type RiskMetrics struct {
	ExpectedReturn float64 // mu'w
	Volatility     float64 // sqrt(w'Sigma w)
	Variance       float64
	// RiskContributions[i] = w_i * (Sigma w)_i / (w'Sigma w); sums to 1.
	RiskContributions []float64
	// Herfindahl concentration of absolute weights.
	Herfindahl float64
	// EffectiveBets = 1 / sum(rc_i^2), a diversification measure.
	EffectiveBets float64
	Turnover      float64 // vs. current weights when provided, else 0
}

// Result is the output of a portfolio construction.
type Result struct {
	Status      Status
	Weights     []float64
	Metrics     RiskMetrics
	Diagnostics []Diagnostic
	Iterations  int
	Objective   float64 // achieved objective value
	Message     string  // human-readable explanation of status/trade-offs
}

// Estimates bundles the estimated expected returns and covariance.
type Estimates struct {
	Mu          []float64
	Cov         [][]float64 // symmetric PSD, n x n
	NSamples    int
	Diagnostics []Diagnostic
}
