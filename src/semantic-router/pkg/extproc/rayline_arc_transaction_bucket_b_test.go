//go:build vsr_next_bucket_b

// Parked until Bucket B re-seats the ARC dispatch hooks on upstream's
// prepareProviderDispatch / applyDispatchDecision seam. Build with
// -tags vsr_next_bucket_b once those symbols exist again.

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
	"os"
	"slices"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCEpisodeFinalizerCoversEOFErrorCancelAndPanic(
	t *testing.T,
) {
	tests := []struct {
		name   string
		stream ext_proc.ExternalProcessor_ProcessServer
	}{
		{
			name:   "EOF",
			stream: NewMockStream(nil),
		},
		{
			name: "handler error",
			stream: &MockStream{
				Ctx:       context.Background(),
				RecvError: errors.New("synthetic receive failure"),
			},
		},
		{
			name: "cancel",
			stream: &MockStream{
				Ctx:       context.Background(),
				RecvError: status.Error(codes.Canceled, "synthetic cancel"),
			},
		},
		{
			name: "panic",
			stream: &panicOnRecvStream{
				MockStream: MockStream{Ctx: context.Background()},
				panicVal:   "synthetic panic",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
			requestContext.RaylineARCTransaction.markSelection(1, 123)
			_ = router.processWithContext(test.stream, requestContext)
			assertARCEpisodeNotAdvanced(t, store, episode)
		})
	}
}

func TestRaylineARCTransactionRenewsRedisLease(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	episode := raylinearc.HashEpisodeID(
		t.Name() + time.Now().String(),
	)
	store, err := raylinearc.NewRedisEpisodeStore(
		raylinearc.RedisEpisodeStoreConfig{
			Address: address,
			KeyPrefix: "test:rayline-arc:" +
				episode + ":",
			LeaseTTL: 90 * time.Millisecond,
			IdleTTL:  time.Minute,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	lease, state, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	requestContext := &RequestContext{
		Headers: make(map[string]string),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(
			store,
			lease,
			state,
			episode,
			90*time.Millisecond,
			nil,
		),
	}
	requestContext.RaylineARCTransaction.markSelection(0, 77)
	time.Sleep(240 * time.Millisecond)
	if finalizeErr := finalizeRaylineARCResponseHeaders(
		requestContext,
		true,
	); finalizeErr != nil {
		t.Fatal(finalizeErr)
	}
	nextLease, nextState, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = store.Abort(context.Background(), nextLease)
	}()
	if nextState.TurnIndex != 1 {
		t.Fatalf("renewed transaction state = %#v", nextState)
	}
}

func TestRaylineARCEpisodeCommitsOnceOnFirst2xxHeaders(t *testing.T) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	requestContext.RaylineARCTransaction.markSelectionWithAffinity(
		1,
		123,
		"replica-b",
		[]string{"replica-a", "replica-b"},
	)
	requestContext.VSRRaylineARC = &selection.RaylineARCTrace{
		EpisodeIDHash: episode,
	}
	requestContext.RequestModel = "worker-b"

	if _, err := router.handleResponseHeaders(
		arcResponseHeaders("200"),
		requestContext,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := router.handleResponseHeaders(
		arcResponseHeaders("204"),
		requestContext,
	); err != nil {
		t.Fatal(err)
	}
	router.finalizeRaylineARCAbort(requestContext, "stream_abort_after_2xx")

	lease, state, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = store.Abort(context.Background(), lease)
	}()
	if state.TurnIndex != 1 ||
		state.PreviousArm == nil ||
		*state.PreviousArm != 1 ||
		state.Warmth[1] == nil ||
		state.Warmth[1].LastInputTokens != 123 ||
		state.EncoderOwner != "replica-b" ||
		!slices.Equal(
			state.EncoderVisitedOwners,
			[]string{"replica-a", "replica-b"},
		) {
		t.Fatalf("committed state = %#v", state)
	}
}

func TestRaylineARCEpisodeAbortsOnNon2xx(t *testing.T) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	requestContext.RaylineARCTransaction.markSelection(1, 123)
	if _, err := router.handleResponseHeaders(
		arcResponseHeaders("503"),
		requestContext,
	); err != nil {
		t.Fatal(err)
	}
	assertARCEpisodeNotAdvanced(t, store, episode)
}

func TestRaylineARCEpisodeCloseFansOutAfter2xxAndClearsAffinity(t *testing.T) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	transaction := requestContext.RaylineARCTransaction
	transaction.markSelectionWithAffinity(
		1,
		123,
		"replica-b",
		[]string{"replica-a", "replica-b"},
	)
	transaction.closeRequested = true
	transaction.sessionCloseWait = time.Second
	closeCalls := 0
	transaction.sessionCloser = func(
		_ context.Context,
		gotEpisode string,
		visited []string,
	) (raylinearc.EncoderCloseReport, error) {
		closeCalls++
		if gotEpisode != episode ||
			!slices.Equal(visited, []string{"replica-a", "replica-b"}) {
			t.Fatalf("close input = %q/%v", gotEpisode, visited)
		}
		return raylinearc.EncoderCloseReport{Attempted: 2, Closed: 2}, nil
	}
	if _, err := router.handleResponseHeaders(
		arcResponseHeaders("200"),
		requestContext,
	); err != nil {
		t.Fatal(err)
	}
	if closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", closeCalls)
	}
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Abort(context.Background(), lease) }()
	if state.TurnIndex != 1 || state.EncoderOwner != "" ||
		len(state.EncoderVisitedOwners) != 0 {
		t.Fatalf("post-close state = %#v", state)
	}
}

func TestRaylineARCEpisodeCloseFailurePreservesProviderSuccessAndAffinity(
	t *testing.T,
) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	transaction := requestContext.RaylineARCTransaction
	transaction.markSelectionWithAffinity(
		1,
		123,
		"replica-b",
		[]string{"replica-a", "replica-b"},
	)
	transaction.closeRequested = true
	transaction.sessionCloseWait = time.Second
	transaction.sessionCloser = func(
		context.Context,
		string,
		[]string,
	) (raylinearc.EncoderCloseReport, error) {
		return raylinearc.EncoderCloseReport{Attempted: 2, Closed: 1, Failed: 1},
			&raylinearc.EncoderFailure{
				Class: raylinearc.EncoderFailureTransport,
				Stage: "pre_response",
			}
	}
	response, err := router.handleResponseHeaders(
		arcResponseHeaders("200"),
		requestContext,
	)
	if err != nil || response == nil || response.GetImmediateResponse() != nil {
		t.Fatalf("successful provider response replaced by close failure: response=%#v err=%v", response, err)
	}
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Abort(context.Background(), lease) }()
	if state.EncoderOwner != "replica-b" ||
		!slices.Equal(
			state.EncoderVisitedOwners,
			[]string{"replica-a", "replica-b"},
		) {
		t.Fatalf("failed-close affinity = %#v", state)
	}
}

func TestRaylineARCEpisodeCommitFailsAfterLeaseLoss(t *testing.T) {
	router, requestContext, _, _ := newTestARCEpisodeTransaction(t)
	requestContext.RaylineARCTransaction.markSelection(1, 123)
	requestContext.RaylineARCTransaction.leaseLost.Store(true)
	response, err := router.handleResponseHeaders(
		arcResponseHeaders("200"),
		requestContext,
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.GetImmediateResponse() == nil ||
		int(response.GetImmediateResponse().GetStatus().GetCode()) != 503 ||
		boundedARCEpisodeFailure(
			requestContext.RaylineARCTransaction.finalizeErr,
		) != "lease_lost" {
		t.Fatalf(
			"lease-loss response=%#v finalize_error=%v",
			response,
			requestContext.RaylineARCTransaction.finalizeErr,
		)
	}
}
