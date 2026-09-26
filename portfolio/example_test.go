package portfolio_test

import (
	"fmt"

	portfolio "github.com/guoyyit/portfolio-optimizer-go/portfolio"
)

// Example demonstrates a basic mean-variance construction with a group cap.
func Example() {
	// 60 days of returns for 3 assets (illustrative values).
	returns := [][]float64{
		{0.010, 0.005, 0.002},
		{-0.008, 0.004, 0.001},
		{0.012, -0.003, 0.003},
		{0.005, 0.006, -0.001},
		{-0.004, 0.002, 0.002},
		{0.009, -0.001, 0.001},
	}
	res := portfolio.ConstructOnce(portfolio.Problem{
		Returns:   returns,
		Objective: portfolio.ObjectiveMeanVariance,
		Estimator: portfolio.EstimatorShrinkage,
		Constraints: portfolio.Constraints{
			MaxWeight: 0.8,
			Groups: []portfolio.GroupCap{
				{Name: "risky", Members: []int{0, 1}, Max: 0.9},
			},
		},
	})
	fmt.Println("status:", res.Status)
	fmt.Printf("sum of weights: %.6f\n", sum(res.Weights))
	fmt.Println("converged:", res.Diagnostics.Converged)
	// Output:
	// status: optimal
	// sum of weights: 1.000000
	// converged: true
}

func sum(w []float64) float64 {
	s := 0.0
	for _, v := range w {
		s += v
	}
	return s
}
