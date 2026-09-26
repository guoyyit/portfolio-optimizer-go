package portfolio

import (
	"math"
	"math/rand"
)

// robust.go: perturbation-based sensitivity and robustness diagnostics.
//
// AssessRobustness re-solves the problem on slightly perturbed return
// series (Gaussian noise proportional to each asset's volatility) using a
// seeded RNG, so results are exactly reproducible for a given Seed.

// perturbScale is the noise magnitude relative to each asset's sample
// standard deviation.
const perturbScale = 0.01

// defaultTrials is used when the caller passes trials <= 0.
const defaultTrials = 16

// RobustnessReport summarizes weight stability under input perturbations.
type RobustnessReport struct {
	BaseWeights       []float64
	MaxWeightShift    float64   // max L1 shift across perturbation trials
	MeanWeightShift   float64   // mean L1 shift across trials
	WeightSensitivity []float64 // mean |shift| per asset
	Herfindahl        float64   // concentration of the base portfolio
	EffectiveN        float64
	CondNumber        float64 // conditioning of the covariance
	Stable            bool    // heuristic overall verdict
	Trials            int
}

// stability thresholds for the Stable verdict.
const (
	stableMaxShift = 0.10 // 10% total weight shift
	stableCond     = 1e8
)

// AssessRobustness perturbs the return series slightly, re-solves, and
// measures weight sensitivity. Deterministic given p.Seed.
func AssessRobustness(p Problem, trials int) (RobustnessReport, error) {
	var rep RobustnessReport
	if err := validateProblem(&p); err != nil {
		return rep, err
	}
	if trials <= 0 {
		trials = defaultTrials
	}
	seed := p.Seed
	if seed == 0 {
		seed = DefaultSeed
	}

	base := solve(p)
	if base.Status == StatusInvalidInput || base.Status == StatusInfeasible {
		rep.BaseWeights = base.Weights
		return rep, nil
	}
	rep.BaseWeights = base.Weights
	rep.Herfindahl = base.Risk.Herfindahl
	rep.EffectiveN = base.Risk.EffectiveN

	// Per-asset scales from the cleaned estimate.
	mu, sigma, diag, err := estimate(p.Returns, p.Estimator)
	if err != nil {
		return rep, err
	}
	_ = mu
	rep.CondNumber = diag.CondNumber
	n := len(p.Returns[0])
	t := len(p.Returns)
	scales := make([]float64, n)
	for j := 0; j < n; j++ {
		scales[j] = math.Sqrt(math.Max(sigma.at(j, j), 1e-30)) * perturbScale
	}

	rng := rand.New(rand.NewSource(seed))
	rep.WeightSensitivity = make([]float64, n)
	rep.Trials = trials
	var sumShift, maxShift float64
	for k := 0; k < trials; k++ {
		perturbed := make([][]float64, t)
		for i := 0; i < t; i++ {
			perturbed[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				perturbed[i][j] = p.Returns[i][j] + rng.NormFloat64()*scales[j]
			}
		}
		q := p
		q.Returns = perturbed
		out := solve(q)
		if out.Weights == nil {
			continue // infeasible perturbation: skip, base result stands
		}
		shift := 0.0
		for j := 0; j < n; j++ {
			d := math.Abs(out.Weights[j] - base.Weights[j])
			shift += d
			rep.WeightSensitivity[j] += d
		}
		sumShift += shift
		if shift > maxShift {
			maxShift = shift
		}
	}
	rep.MaxWeightShift = maxShift
	rep.MeanWeightShift = sumShift / float64(trials)
	for j := range rep.WeightSensitivity {
		rep.WeightSensitivity[j] /= float64(trials)
	}
	rep.Stable = maxShift < stableMaxShift && rep.CondNumber < stableCond
	return rep, nil
}
