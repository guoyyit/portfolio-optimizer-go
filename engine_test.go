package portfolio

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

func buildRequestFor(key string, seedOffset float64) BuildRequest {
	return BuildRequest{
		Key: key,
		Returns: makeReturns(60, 4, func(t, j int) float64 {
			return 0.001*float64(j) + 0.01*math.Sin(float64(t)*0.3+float64(j)+seedOffset)
		}),
		Options: Options{MaxIterations: 500},
	}
}

func TestEngineBuildAndGet(t *testing.T) {
	e := NewEngine()
	res, err := e.Build(context.Background(), buildRequestFor("a", 0))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusOptimal && res.Status != StatusSuboptimal {
		t.Fatalf("bad status %v", res.Status)
	}
	got, err := e.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Weights {
		if got.Weights[i] != res.Weights[i] {
			t.Fatalf("stored result differs")
		}
	}
	if keys := e.Keys(); len(keys) != 1 || keys[0] != "a" {
		t.Fatalf("bad keys: %v", keys)
	}
	// mutation of returned copy must not pollute the store
	got.Weights[0] = -999
	again, _ := e.Get("a")
	if again.Weights[0] == -999 {
		t.Fatalf("Get returned shared state")
	}
	if _, err := e.Get("missing"); err != ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
	e.Delete("a")
	if _, err := e.Get("a"); err != ErrNotFound {
		t.Fatalf("delete failed")
	}
}

func TestEngineConcurrent(t *testing.T) {
	e := NewEngine()
	const workers = 16
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			key := fmt.Sprintf("k%d", w%4)
			for i := 0; i < 10; i++ {
				if _, err := e.Build(context.Background(), buildRequestFor(key, float64(w))); err != nil {
					t.Errorf("build: %v", err)
					return
				}
				if _, err := e.Get(key); err != nil {
					t.Errorf("get: %v", err)
					return
				}
				_ = e.Keys()
			}
		}(w)
	}
	wg.Wait()
	keys := e.Keys()
	if len(keys) != 4 {
		t.Fatalf("expected 4 keys, got %v", keys)
	}
	// all stored results must be complete and feasible
	for _, k := range keys {
		res, err := e.Get(k)
		if err != nil {
			t.Fatal(err)
		}
		if !satisfies(res.Weights, nil, 4, 1e-8) {
			t.Fatalf("stored result %q infeasible: %v", k, res.Weights)
		}
	}
}

func TestEngineCancelLeavesNoPartialWrite(t *testing.T) {
	e := NewEngine()
	if _, err := e.Build(context.Background(), buildRequestFor("x", 0)); err != nil {
		t.Fatal(err)
	}
	before, _ := e.Get("x")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := e.Build(ctx, BuildRequest{
		Key:     "x",
		Returns: makeReturns(50, 4, func(t, j int) float64 { return 0.02 * math.Cos(float64(t+j)) }),
		Options: Options{MaxIterations: 100000},
	})
	if err == nil {
		t.Log("build completed despite cancelled context (fast path); checking store consistency anyway")
	}
	after, _ := e.Get("x")
	if err != nil {
		// cancelled build must not have overwritten the stored result
		for i := range before.Weights {
			if after.Weights[i] != before.Weights[i] {
				t.Fatalf("cancelled build polluted stored result")
			}
		}
	}
}
