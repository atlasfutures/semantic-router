/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestEpisodeStateAdvancesOnlyOnCommit(t *testing.T) {
	t.Parallel()
	state, err := NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{},
		Workers: []WorkerManifest{
			syntheticWorker("a", 1, 0),
			syntheticWorker("b", 1, 0),
		},
	})
	now := time.Unix(1_000, 0)
	if _, err := policy.Select([]float32{1, 2}, nil, state, 123, now); err != nil {
		t.Fatal(err)
	}
	if state.PreviousArm != nil || state.TurnIndex != 0 {
		t.Fatalf("selection mutated state: %+v", state)
	}
	if err := state.Commit(1, 123, now); err != nil {
		t.Fatal(err)
	}
	if state.PreviousArm == nil || *state.PreviousArm != 1 ||
		state.TurnIndex != 1 ||
		state.Warmth[0] != nil ||
		state.Warmth[1] == nil ||
		state.Warmth[1].LastInputTokens != 123 {
		t.Fatalf("Commit() state = %+v", state)
	}
}

func TestExpectedCacheHitRatioAndSwitchCostMatchRayline(t *testing.T) {
	t.Parallel()
	assertFloat64Close(t, expectedCacheHitRatio(nil), 0, 0)
	for _, test := range []struct {
		seconds float64
		want    float64
	}{
		{0, 0.8},
		{150, 0.8},
		{225, 0.5},
		{300, 0.2},
		{600, 0.2},
	} {
		seconds := test.seconds
		assertFloat64Close(
			t,
			expectedCacheHitRatio(&seconds),
			test.want,
			1e-12,
		)
	}

	now := time.Unix(1_000, 0)
	worker := syntheticWorker("worker", 0.00001, 0.000001)
	miss, cost := switchCostForWorker(
		&worker,
		&WorkerWarmth{
			LastUsed:        now,
			LastInputTokens: 1_000,
		},
		1_200,
		now,
	)
	if miss != 400 {
		t.Fatalf("miss tokens = %d, want 400", miss)
	}
	assertFloat64Close(t, cost, 0.0036, 1e-12)
}

func TestPolicyAppliesColdAndStayMargins(t *testing.T) {
	t.Parallel()
	now := time.Unix(1_000, 0)
	previous := 0
	state := &EpisodeState{
		PreviousArm: &previous,
		Warmth: []*WorkerWarmth{
			{LastUsed: now, LastInputTokens: 1_000},
			nil,
			nil,
		},
	}
	manifest := &Manifest{
		Policy: PolicyManifest{
			PreviousWorkerStayMargin: 0.05,
			ColdSwitchMarginPerUSD:   1,
			ColdSwitchUpgradeExempt:  true,
			StayMarginUpgradeExempt:  false,
		},
		Workers: []WorkerManifest{
			syntheticWorker("previous", 0.00001, 0.000001),
			syntheticWorker("cheaper", 0.000005, 0),
			syntheticWorker("upgrade", 0.00002, 0),
		},
	}
	policy := newPolicy(manifest)
	decision, err := policy.Select(
		[]float32{1, 1.04, 1.03},
		nil,
		state,
		1_000,
		now,
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != 0 || !decision.Stayed {
		t.Fatalf("Select() = %+v, want previous worker stay", decision)
	}
	if decision.CacheMissTokens[1] != 1_000 ||
		decision.SwitchCostUSD[1] != 0.005 {
		t.Fatalf("cold switch accounting = %+v", decision)
	}
	if !decision.ColdSwitchUpgradeExemptions[2] ||
		decision.SwitchCostUSD[2] != 0 {
		t.Fatalf("upgrade exemption = %+v", decision)
	}

	policy.contract.StayMarginUpgradeExempt = true
	decision, err = policy.Select(
		[]float32{1, 0, 1.03},
		nil,
		state,
		1_000,
		now,
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != 2 || decision.Stayed ||
		!decision.StayUpgradeExempted {
		t.Fatalf("Select(upgrade) = %+v", decision)
	}
}

func TestPolicyStayMarginEqualityStays(t *testing.T) {
	t.Parallel()
	previous := 0
	state := &EpisodeState{
		PreviousArm: &previous,
		Warmth:      make([]*WorkerWarmth, 2),
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{
			PreviousWorkerStayMargin: 0.05,
		},
		Workers: []WorkerManifest{
			syntheticWorker("previous", 1, 0),
			syntheticWorker("candidate", 1, 0),
		},
	})
	decision, err := policy.Select(
		[]float32{0, policy.contract.PreviousWorkerStayMargin},
		nil,
		state,
		1,
		time.Unix(1_000, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != previous || !decision.Stayed {
		t.Fatalf("Select(stay-margin equality) = %+v", decision)
	}
}

func TestPolicyRejectsInvalidInputs(t *testing.T) {
	t.Parallel()
	policy := newPolicy(&Manifest{
		Workers: []WorkerManifest{syntheticWorker("worker", 1, 0)},
	})
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1_000, 0)
	tests := []struct {
		name     string
		scores   []float32
		excluded []bool
		state    *EpisodeState
		tokens   int
		now      time.Time
	}{
		{"score count", nil, nil, state, 1, now},
		{"nonfinite score", []float32{float32(math.NaN())}, nil, state, 1, now},
		{"nil state", []float32{1}, nil, nil, 1, now},
		{"negative tokens", []float32{1}, nil, state, -1, now},
		{"zero time", []float32{1}, nil, state, 1, time.Time{}},
		{
			"warmth count",
			[]float32{1},
			nil,
			&EpisodeState{},
			1,
			now,
		},
		{"exclusion count", []float32{1}, []bool{false, true}, state, 1, now},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := policy.Select(
				test.scores,
				test.excluded,
				test.state,
				test.tokens,
				test.now,
			); err == nil {
				t.Fatal("Select() unexpectedly succeeded")
			}
		})
	}
}

func assertFloat64Close(
	t *testing.T,
	actual float64,
	expected float64,
	tolerance float64,
) {
	t.Helper()
	if math.Abs(actual-expected) > tolerance {
		t.Fatalf(
			"actual %.16f, want %.16f (tolerance %.16f)",
			actual,
			expected,
			tolerance,
		)
	}
}

// A hard constraint outranks the score. The excluded arm keeps its score in
// the trace, because masking it would hide why the winner won.
func TestPolicyExclusionOutranksTheHighestScore(t *testing.T) {
	t.Parallel()
	state, err := NewEpisodeState(3)
	if err != nil {
		t.Fatal(err)
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{},
		Workers: []WorkerManifest{
			syntheticWorker("a", 1, 0),
			syntheticWorker("b", 1, 0),
			syntheticWorker("c", 1, 0),
		},
	})
	decision, err := policy.Select(
		[]float32{0.1, 0.9, 0.5},
		[]bool{false, true, false},
		state,
		10,
		time.Unix(1_000, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != 2 || decision.SelectedWorker != "c" {
		t.Fatalf("Select(excluded) = %+v, want the best eligible arm", decision)
	}
	if decision.AdjustedScores[1] != 0.9 {
		t.Fatalf("exclusion rewrote the trace scores: %+v", decision)
	}
	if !reflect.DeepEqual(
		decision.ExcludedArms,
		[]bool{false, true, false},
	) {
		t.Fatalf("Select() lost the exclusion record: %+v", decision)
	}
}

// The stay margin resists churn. It must not resist a constraint the request
// itself imposes, so an excluded previous arm is a forced switch.
func TestPolicyExcludedPreviousArmIsAForcedSwitch(t *testing.T) {
	t.Parallel()
	previous := 0
	state := &EpisodeState{
		PreviousArm: &previous,
		Warmth:      make([]*WorkerWarmth, 2),
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{PreviousWorkerStayMargin: 10},
		Workers: []WorkerManifest{
			syntheticWorker("previous", 1, 0),
			syntheticWorker("candidate", 1, 0),
		},
	})
	decision, err := policy.Select(
		[]float32{0.9, 0.1},
		[]bool{true, false},
		state,
		10,
		time.Unix(1_000, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != 1 || decision.Stayed {
		t.Fatalf(
			"Select(excluded previous) = %+v, want a switch to arm 1",
			decision,
		)
	}
}

func TestPolicyFailsWhenExclusionEmptiesThePool(t *testing.T) {
	t.Parallel()
	state, err := NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{},
		Workers: []WorkerManifest{
			syntheticWorker("a", 1, 0),
			syntheticWorker("b", 1, 0),
		},
	})
	_, err = policy.Select(
		[]float32{0.1, 0.9},
		[]bool{true, true},
		state,
		10,
		time.Unix(1_000, 0),
	)
	if !errors.Is(err, ErrNoEligibleArm) {
		t.Fatalf("Select(all excluded) error = %v, want ErrNoEligibleArm", err)
	}
}

// An empty constraint must leave the artifact's own behaviour byte for byte
// where it was, which is what makes the flag safe to add unmarked.
func TestPolicyEmptyExclusionMatchesTheUnconstrainedChoice(t *testing.T) {
	t.Parallel()
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{},
		Workers: []WorkerManifest{
			syntheticWorker("a", 1, 0),
			syntheticWorker("b", 1, 0),
		},
	})
	now := time.Unix(1_000, 0)
	scores := []float32{0.5, 0.5}
	for _, excluded := range [][]bool{nil, {false, false}} {
		state, err := NewEpisodeState(2)
		if err != nil {
			t.Fatal(err)
		}
		decision, err := policy.Select(scores, excluded, state, 10, now)
		if err != nil {
			t.Fatal(err)
		}
		if decision.SelectedArm != 0 {
			t.Fatalf(
				"Select(%v) = arm %d, want the first-wins tie break",
				excluded,
				decision.SelectedArm,
			)
		}
	}
}

// A standing exclusion meets both defences at once: the arm holds the top
// score and the episode is parked on it. Neither may keep it, or an arm taken
// out of service would go on serving every turn of a settled episode.
func TestPolicyExclusionBeatsTheScoreAndTheStayMarginTogether(t *testing.T) {
	t.Parallel()
	previous := 0
	state := &EpisodeState{
		PreviousArm: &previous,
		Warmth:      make([]*WorkerWarmth, 2),
	}
	policy := newPolicy(&Manifest{
		Policy: PolicyManifest{PreviousWorkerStayMargin: 10},
		Workers: []WorkerManifest{
			syntheticWorker("excluded", 1, 0),
			syntheticWorker("eligible", 1, 0),
		},
	})
	decision, err := policy.Select(
		[]float32{0.9, 0.1},
		[]bool{true, false},
		state,
		10,
		time.Unix(1_000, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.SelectedArm != 1 || decision.Stayed {
		t.Fatalf(
			"Select(excluded top-scoring previous) = %+v, want a switch to arm 1",
			decision,
		)
	}
	if !reflect.DeepEqual(decision.ExcludedArms, []bool{true, false}) {
		t.Fatalf("Select() lost the exclusion record: %+v", decision)
	}
}
