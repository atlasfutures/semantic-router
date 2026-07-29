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

package extproc

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCSelectorMapsArtifactArmWithoutMutatingState(t *testing.T) {
	previous := 0
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	state.PreviousArm = &previous
	state.TurnIndex = 7
	encoder := &fakeARCEncoder{
		result: &raylinearc.EncoderResult{
			Embedding:         make([]float32, 1024),
			SerializedTokens:  120,
			FullHistoryTokens: 140,
			TruncatedTokens:   20,
			ModelRevision:     config.RaylineARCEncoderModelRevision,
			EngineBuildID:     "vllm@build",
			IOPluginVersion:   "rayline-arc-io@0.1.0",
		},
	}
	scorer := &fakeARCScorer{
		workerIDs: []string{"worker-a", "worker-b"},
		decision: raylinearc.Decision{
			SelectedArm:     1,
			SelectedWorker:  "worker-b",
			RawScores:       []float32{0.1, 0.9},
			AdjustedScores:  []float32{0.1, 0.8},
			SwitchCostUSD:   []float64{0, 0.01},
			CacheMissTokens: []int{0, 100},
		},
	}
	selector := newRaylineARCSelector(
		scorer,
		encoder,
		"artifact-revision",
	)
	selectionContext := validARCSelectionContext(state)

	result, err := selector.Select(context.Background(), selectionContext)
	if err != nil {
		t.Fatal(err)
	}
	assertRaylineARCSelectionResult(t, result)
	assertRaylineARCStateUnchanged(t, state)
	if !reflect.DeepEqual(encoder.turns, selectionContext.RaylineARC.Turns) {
		t.Fatalf("encoder turns = %#v", encoder.turns)
	}
}

//nolint:cyclop // One assertion helper audits every field of the bounded ARC trace.
func assertRaylineARCSelectionResult(
	t *testing.T,
	result *selection.SelectionResult,
) {
	t.Helper()
	if result.SelectedModel != "worker-b" ||
		result.Method != selection.MethodRaylineARC ||
		result.Tier != selection.TierExperimental {
		t.Fatalf("unexpected selection result: %#v", result)
	}
	if result.AllScores != nil {
		t.Fatalf(
			"ARC worker ids leaked through generic scores: %#v",
			result.AllScores,
		)
	}
	if result.RaylineARC == nil ||
		result.RaylineARC.SelectedArm != 1 ||
		result.RaylineARC.EpisodeIDHash != strings.Repeat("a", 64) {
		t.Fatalf("missing ARC trace: %#v", result.RaylineARC)
	}
	// The trace must carry deployment-private artifact identity only as
	// fixed-width hashes, never the raw pin values.
	for _, hashed := range []string{
		result.RaylineARC.ArtifactID,
		result.RaylineARC.ArtifactRevision,
	} {
		if len(hashed) != 16 ||
			strings.Contains(hashed, "artifact") ||
			strings.ToLower(hashed) != hashed {
			t.Fatalf("ARC trace leaked raw artifact identity: %#v", result.RaylineARC)
		}
	}
	if result.RaylineARC.ArtifactRevision == result.RaylineARC.ArtifactID {
		t.Fatalf("distinct identities hashed to one value: %#v", result.RaylineARC)
	}
}

func assertRaylineARCStateUnchanged(
	t *testing.T,
	state *raylinearc.EpisodeState,
) {
	t.Helper()
	if state.TurnIndex != 7 || state.PreviousArm == nil ||
		*state.PreviousArm != 0 {
		t.Fatalf("selection mutated episode state: %#v", state)
	}
}

func TestRaylineARCSelectorRejectsCandidateOrderDriftBeforeEncoding(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := &fakeARCEncoder{}
	selector := newRaylineARCSelector(
		&fakeARCScorer{workerIDs: []string{"worker-a", "worker-b"}},
		encoder,
		"artifact-revision",
	)
	selectionContext := validARCSelectionContext(state)
	selectionContext.CandidateModels[0],
		selectionContext.CandidateModels[1] = selectionContext.CandidateModels[1],
		selectionContext.CandidateModels[0]

	if _, err := selector.Select(
		context.Background(),
		selectionContext,
	); err == nil {
		t.Fatal("expected candidate-order failure")
	}
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
}

func TestRaylineARCSelectorRejectsPreparationFailure(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := &fakeARCEncoder{}
	selector := newRaylineARCSelector(
		&fakeARCScorer{workerIDs: []string{"worker-a", "worker-b"}},
		encoder,
		"artifact-revision",
	)
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.PreparationFailure = "invalid_turns"

	if _, err := selector.Select(
		context.Background(),
		selectionContext,
	); err == nil {
		t.Fatal("expected preparation failure")
	}
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
}

func TestCreateRaylineARCSelectorKeepsComponentNotReadyOnArtifactFailure(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "arc",
					ModelRefs: []config.ModelRef{{Model: "a"}, {Model: "b"}},
					Algorithm: &config.AlgorithmConfig{
						Type:    config.RaylineARCAlgorithmType,
						OnError: "fail_closed",
						RaylineARC: &config.RaylineARCAlgorithmConfig{
							ArtifactDir:      "/missing/rayline-arc-artifact",
							ArtifactRevision: "immutable-revision",
						},
					},
				},
			},
		},
	}
	selector, episodeStore, closeStore, failureClass := createRaylineARCSelector(cfg)
	if selector == nil || failureClass != "artifact" {
		t.Fatalf(
			"selector=%#v failure_class=%q, want unavailable artifact selector",
			selector,
			failureClass,
		)
	}
	if episodeStore != nil || closeStore != nil {
		t.Fatal("unready selector retained an episode store")
	}
	if _, err := selector.Select(
		context.Background(),
		&selection.SelectionContext{},
	); err == nil {
		t.Fatal("not-ready selector did not fail closed")
	}
}

func TestRaylineARCOptionalSecretFailsClosedWithoutDisclosingReference(t *testing.T) {
	envName := "RAYLINE_ARC_TEST_MISSING_MODAL_CREDENTIAL"
	secretValue := t.Name() + "-secret"

	if _, err := raylineARCOptionalSecret(envName); err == nil ||
		strings.Contains(err.Error(), envName) {
		t.Fatalf("missing credential error = %v", err)
	}

	t.Setenv(envName, secretValue)
	value, err := raylineARCOptionalSecret(envName)
	if err != nil {
		t.Fatal(err)
	}
	if value != secretValue {
		t.Fatal("credential environment was not resolved")
	}
}

func validARCSelectionContext(
	state *raylinearc.EpisodeState,
) *selection.SelectionContext {
	return &selection.SelectionContext{
		DecisionName: "arc",
		CandidateModels: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
		},
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("a", 64),
			Turns: []raylinearc.Turn{
				{Role: "user", Text: "hello"},
			},
			State: state,
		},
	}
}

type fakeARCEncoder struct {
	result *raylinearc.EncoderResult
	err    error
	calls  int
	turns  []raylinearc.Turn
}

func (encoder *fakeARCEncoder) Encode(
	_ context.Context,
	_ string,
	turns []raylinearc.Turn,
) (*raylinearc.EncoderResult, error) {
	encoder.calls++
	encoder.turns = append([]raylinearc.Turn(nil), turns...)
	if encoder.err != nil {
		return nil, encoder.err
	}
	if encoder.result == nil {
		return nil, errors.New("missing fake encoder result")
	}
	return encoder.result, nil
}

type fakeARCScorer struct {
	workerIDs []string
	decision  raylinearc.Decision
	err       error
}

func (scorer *fakeARCScorer) WorkerIDs() []string {
	return append([]string(nil), scorer.workerIDs...)
}

func (scorer *fakeARCScorer) ArtifactID() string {
	return "artifact-id"
}

func (scorer *fakeARCScorer) EncoderRevision() string {
	return config.RaylineARCEncoderModelRevision
}

func (scorer *fakeARCScorer) Worker(
	index int,
) (raylinearc.WorkerManifest, bool) {
	if index < 0 || index >= len(scorer.workerIDs) {
		return raylinearc.WorkerManifest{}, false
	}
	return raylinearc.WorkerManifest{
		ID:    scorer.workerIDs[index],
		Model: "provider/" + scorer.workerIDs[index],
	}, true
}

func (scorer *fakeARCScorer) Select(
	_ []float32,
	_ *raylinearc.EpisodeState,
	_ int,
	_ time.Time,
) (raylinearc.Decision, error) {
	return scorer.decision, scorer.err
}
