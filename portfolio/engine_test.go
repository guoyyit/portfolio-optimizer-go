package portfolio

import (
	"math"
	"sync"
	"testing"
)

func TestEngineCachesDeterministically(t *testing.T) {
	e := NewEngine()
	p := Problem{
		Returns:   mkReturns(150, 4, 71),
		Objective: ObjectiveMeanVariance,
		Seed:      5,
	}
	r1, err := e.Construct(p)
	if err != nil {
		t.Fatal(err)
	}
	if !e.Cached(p) {
		t.Fatal("expected problem to be cached")
	}
	r2, err := e.Construct(p)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1.Weights {
		if math.Float64bits(r1.Weights[i]) != math.Float64bits(r2.Weights[i]) {
			t.Fatal("cached result differs")
		}
	}
	// Mutating the returned weights must not pollute the cache.
	r1.Weights[0] = 999
	r3, _ := e.Construct(p)
	if r3.Weights[0] == 999 {
		t.Fatal("cache polluted by caller mutation")
	}
}

func TestEngineConcurrent(t *testing.T) {
	e := NewEngine()
	problems := []Problem{
		{Returns: mkReturns(120, 3, 73), Objective: ObjectiveMeanVariance},
		{Returns: mkReturns(120, 4, 79), Objective: ObjectiveRiskParity},
		{Returns: mkReturns(120, 5, 83), Objective: ObjectiveMeanVariance,
			Constraints: Constraints{MaxWeight: 0.4}},
	}
	// Reference results computed serially.
	want := make([]Result, len(problems))
	for i, p := range problems {
		r, err := e.Construct(p)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = r
	}
	// Concurrent queries must return identical results.
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for k := 0; k < 32; k++ {
		for i := range problems {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				r, err := e.Construct(problems[idx])
				if err != nil {
					errs <- err
					return
				}
				for j := range r.Weights {
					if math.Float64bits(r.Weights[j]) != math.Float64bits(want[idx].Weights[j]) {
						errs <- &mismatchError{}
						return
					}
				}
			}(i)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

type mismatchError struct{}

func (m *mismatchError) Error() string { return "concurrent result mismatch" }

func TestEngineConcurrentMixedProblems(t *testing.T) {
	// Distinct problems racing for first computation must not interfere.
	e := NewEngine()
	var wg sync.WaitGroup
	results := make([]Result, 16)
	for k := 0; k < 16; k++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			p := Problem{
				Returns:   mkReturns(100, 3+idx%3, int64(90+idx)),
				Objective: Objective(idx % 2),
				Seed:      int64(idx),
			}
			r, err := e.Construct(p)
			if err != nil {
				t.Error(err)
				return
			}
			results[idx] = r
		}(k)
	}
	wg.Wait()
	// Every result must satisfy hard constraints.
	for k, r := range results {
		if r.Weights == nil {
			t.Fatalf("result %d missing", k)
		}
		sum := 0.0
		for _, w := range r.Weights {
			if math.IsNaN(w) || math.IsInf(w, 0) {
				t.Fatalf("result %d has non-finite weight", k)
			}
			sum += w
		}
		if math.Abs(sum-1) > 1e-8 {
			t.Fatalf("result %d weights sum to %v", k, sum)
		}
	}
}
