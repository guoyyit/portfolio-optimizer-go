package portfolio

import (
	"context"
	"fmt"
	"math"
	"math/rand"
)

// RobustnessConfig controls perturbation-based stability diagnostics.
type RobustnessConfig struct {
	// NumPerturbations random perturbation trials (default 16 when <= 0).
	NumPerturbations int
	// ReturnShock is the relative stddev of gaussian shocks applied to mu
	// (default 0.10 when <= 0).
	ReturnShock float64
	// CovShock is the relative stddev of multiplicative shocks applied to
	// covariance entries (default 0.10 when <= 0).
	CovShock float64
	// Seed fixes the pseudo-random generator for reproducibility.
	Seed int64
}

// Stability thresholds: an allocation is flagged unstable when perturbations
// move any single weight by more than 0.10 on average-max, or average
// solution turnover exceeds 0.20.
const (
	stableMaxShiftThreshold = 0.10
	stableAvgTurnoverThresh = 0.20
	defaultPerturbations    = 16
	defaultShock            = 0.10
)

// RobustnessReport summarizes allocation stability under perturbations.
type RobustnessReport struct {
	// WeightSensitivity[i] = mean |w*_perturbed - w*_base| for asset i.
	WeightSensitivity []float64
	// MaxWeightShift is the largest mean single-asset absolute shift.
	MaxWeightShift float64
	// AvgTurnover is the mean turnover between perturbed and base solutions.
	AvgTurnover float64
	// Concentration (Herfindahl) of the base allocation.
	Concentration float64
	// EffectiveBets of the base allocation.
	EffectiveBets float64
	// Stable is false when sensitivity exceeds built-in thresholds.
	Stable bool
	// BaseWeights is the unperturbed solution.
	BaseWeights []float64
	Findings    []Diagnostic
}

// AssessRobustness perturbs the estimates, re-optimizes and reports
// allocation stability. Deterministic for a fixed Seed: the same inputs and
// seed always produce bit-identical reports.
func AssessRobustness(ctx context.Context, est Estimates, c *Constraints, opts Options, cfg RobustnessConfig) (RobustnessReport, error) {
	n := len(est.Mu)
	if n == 0 {
		return RobustnessReport{}, fmt.Errorf("portfolio: empty estimates")
	}
	numPert := cfg.NumPerturbations
	if numPert <= 0 {
		numPert = defaultPerturbations
	}
	retShock := cfg.ReturnShock
	if retShock <= 0 {
		retShock = defaultShock
	}
	covShock := cfg.CovShock
	if covShock <= 0 {
		covShock = defaultShock
	}

	base, err := Optimize(ctx, est, c, opts)
	if err != nil {
		return RobustnessReport{}, err
	}
	if base.Status == StatusInfeasible || base.Status == StatusInvalidInput {
		return RobustnessReport{}, fmt.Errorf("portfolio: base problem %s: %s", base.Status, base.Message)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	sens := make([]float64, n)
	var avgTurnover float64
	var findings []Diagnostic
	solved := 0
	for trial := 0; trial < numPert; trial++ {
		if err := ctx.Err(); err != nil {
			findings = append(findings, Diagnostic{
				Flag:    DiagUnstable,
				Message: fmt.Sprintf("robustness assessment interrupted after %d/%d trials", trial, numPert),
			})
			break
		}
		pert := Estimates{
			Mu:  make([]float64, n),
			Cov: make([][]float64, n),
		}
		for i := 0; i < n; i++ {
			pert.Mu[i] = est.Mu[i] * (1 + retShock*rng.NormFloat64())
		}
		for i := 0; i < n; i++ {
			pert.Cov[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				pert.Cov[i][j] = est.Cov[i][j] * (1 + covShock*rng.NormFloat64())
			}
		}
		// restore symmetry + PSD after multiplicative noise
		pert.Cov = symmetrize(pert.Cov)
		vals, _ := jacobiEigen(pert.Cov)
		lmax := vals[len(vals)-1]
		floor := 1e-10 * math.Max(lmax, 1)
		pert.Cov, _ = nearestPSD(pert.Cov, floor)

		res, err := Optimize(ctx, pert, c, opts)
		if err != nil || res.Weights == nil {
			findings = append(findings, Diagnostic{
				Flag:    DiagUnstable,
				Message: fmt.Sprintf("perturbation trial %d failed to solve: %v", trial, err),
			})
			continue
		}
		solved++
		var to float64
		for i := 0; i < n; i++ {
			d := math.Abs(res.Weights[i] - base.Weights[i])
			sens[i] += d
			to += d
		}
		avgTurnover += to
	}
	if solved == 0 {
		return RobustnessReport{}, fmt.Errorf("portfolio: no perturbation trial solved")
	}
	inv := 1 / float64(solved)
	var maxShift float64
	for i := range sens {
		sens[i] *= inv
		if sens[i] > maxShift {
			maxShift = sens[i]
		}
	}
	avgTurnover *= inv

	stable := maxShift < stableMaxShiftThreshold && avgTurnover < stableAvgTurnoverThresh
	if !stable {
		findings = append(findings, Diagnostic{
			Flag: DiagUnstable,
			Message: fmt.Sprintf("allocation unstable under %.0f%% parameter shocks: max mean weight shift %.4f, avg turnover %.4f",
				math.Max(retShock, covShock)*100, maxShift, avgTurnover),
		})
	}
	if base.Metrics.Herfindahl > 0.5 {
		findings = append(findings, Diagnostic{
			Flag:    DiagUnstable,
			Message: fmt.Sprintf("concentrated allocation: Herfindahl %.3f (effective bets %.2f)", base.Metrics.Herfindahl, base.Metrics.EffectiveBets),
		})
	}

	return RobustnessReport{
		WeightSensitivity: sens,
		MaxWeightShift:    maxShift,
		AvgTurnover:       avgTurnover,
		Concentration:     base.Metrics.Herfindahl,
		EffectiveBets:     base.Metrics.EffectiveBets,
		Stable:            stable,
		BaseWeights:       cloneVec(base.Weights),
		Findings:          findings,
	}, nil
}
