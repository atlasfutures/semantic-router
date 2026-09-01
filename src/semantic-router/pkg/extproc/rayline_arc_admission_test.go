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
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// blockingARCEncoder holds every call inside Encode until the test releases
// it, so a slot stays occupied while a second decision arrives.
type blockingARCEncoder struct {
	entered chan struct{}
	release chan struct{}
	calls   atomic.Int64
	result  *raylinearc.EncoderResult
}

func newBlockingARCEncoder() *blockingARCEncoder {
	return &blockingARCEncoder{
		entered: make(chan struct{}, 8),
		release: make(chan struct{}),
		result: &raylinearc.EncoderResult{
			Embedding:         make([]float32, 1024),
			SerializedTokens:  120,
			FullHistoryTokens: 140,
			ModelRevision:     config.RaylineARCEncoderModelRevision,
			EngineBuildID:     "vllm@build",
			IOPluginVersion:   "rayline-arc-io@0.1.0",
		},
	}
}

func (encoder *blockingARCEncoder) Encode(
	_ context.Context,
	_ string,
	_ []raylinearc.Turn,
) (*raylinearc.EncoderResult, error) {
	encoder.calls.Add(1)
	encoder.entered <- struct{}{}
	<-encoder.release
	return encoder.result, nil
}

func admissionTestScorer() *fakeARCScorer {
	return &fakeARCScorer{
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
}

func admissionTestContext(t *testing.T) *selection.SelectionContext {
	t.Helper()
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatalf("NewEpisodeState() error = %v", err)
	}
	return validARCSelectionContext(state)
}

// A shed must arrive as its own bounded selection class. Anything that folds
// it into an encoder failure class would make the encoder error rate
// unreadable on the first page of the dashboard.
func TestRaylineARCSelectorShedsWithItsOwnFailureClass(t *testing.T) {
	encoder := newBlockingARCEncoder()
	selector := newRaylineARCSelector(
		admissionTestScorer(),
		encoder,
		raylinearc.NewAdmissionGate(1),
		"artifact-revision",
	)

	// Both contexts are built on the test goroutine: t.Fatalf is only legal
	// there.
	held := admissionTestContext(t)
	offered := admissionTestContext(t)

	firstDone := make(chan error, 1)
	go func() {
		_, err := selector.Select(context.Background(), held)
		firstDone <- err
	}()

	select {
	case <-encoder.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("first decision never reached the encoder")
	}

	_, err := selector.Select(context.Background(), offered)
	if err == nil {
		t.Fatal("second decision was admitted while the only slot was held")
	}
	failure, ok := err.(*raylineARCSelectionFailure)
	if !ok {
		t.Fatalf("shed error = %T, want *raylineARCSelectionFailure", err)
	}
	wantClass := "encoder_" + string(raylinearc.EncoderFailureAdmission)
	if failure.class != wantClass {
		t.Fatalf("class = %q, want %q", failure.class, wantClass)
	}
	// The class is a metric label and a log field, so it must stay bounded and
	// must never leak episode identity or request text.
	if strings.ContainsAny(failure.class, " \t\n\"") {
		t.Fatalf("class %q is not a bounded label value", failure.class)
	}

	// The shed request must not have touched the encoder at all: no session
	// opened, no ingress slot spent, no deadline started.
	if calls := encoder.calls.Load(); calls != 1 {
		t.Fatalf("encoder calls = %d, want 1: a shed must never call the encoder", calls)
	}

	close(encoder.release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first decision error = %v", err)
	}
}

// The slot must come back, or the gate would shed permanently after the first
// burst instead of tracking real occupancy.
func TestRaylineARCSelectorReleasesAdmissionSlotAfterEncoding(t *testing.T) {
	encoder := newBlockingARCEncoder()
	close(encoder.release)
	gate := raylinearc.NewAdmissionGate(1)
	selector := newRaylineARCSelector(
		admissionTestScorer(),
		encoder,
		gate,
		"artifact-revision",
	)

	for attempt := range 4 {
		if _, err := selector.Select(
			context.Background(),
			admissionTestContext(t),
		); err != nil {
			t.Fatalf("sequential decision %d error = %v", attempt, err)
		}
		<-encoder.entered
	}
	if gate.Inflight() != 0 {
		t.Fatalf("Inflight() = %d, want 0 after every decision returned", gate.Inflight())
	}
	if gate.HighWater() != 1 {
		t.Fatalf("HighWater() = %d, want 1 for sequential decisions", gate.HighWater())
	}
}

// Existing deployments leave the cap unset. They must keep admitting every
// decision exactly as before.
func TestRaylineARCSelectorWithoutAdmissionGateAdmitsEveryDecision(t *testing.T) {
	encoder := newBlockingARCEncoder()
	close(encoder.release)
	selector := newRaylineARCSelector(
		admissionTestScorer(),
		encoder,
		nil,
		"artifact-revision",
	)

	const decisions = 8
	contexts := make([]*selection.SelectionContext, 0, decisions)
	for range decisions {
		contexts = append(contexts, admissionTestContext(t))
	}
	errs := make(chan error, decisions)
	for _, offered := range contexts {
		go func() {
			_, err := selector.Select(context.Background(), offered)
			errs <- err
		}()
	}
	for range decisions {
		if err := <-errs; err != nil {
			t.Fatalf("decision error = %v with admission control disabled", err)
		}
	}
	if calls := encoder.calls.Load(); calls != decisions {
		t.Fatalf("encoder calls = %d, want %d", calls, decisions)
	}
}
