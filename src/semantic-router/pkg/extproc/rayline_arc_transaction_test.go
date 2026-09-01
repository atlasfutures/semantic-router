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
	"slices"
	"sync"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

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

func newTestARCEpisodeTransaction(
	t *testing.T,
) (
	*OpenAIRouter,
	*RequestContext,
	*raylinearc.MemoryEpisodeStore,
	string,
) {
	t.Helper()
	store, err := raylinearc.NewMemoryEpisodeStore(
		raylinearc.MemoryEpisodeStoreConfig{
			MaxEpisodes: 8,
			IdleTTL:     time.Minute,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
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
			time.Minute,
			nil,
		),
	}
	return &OpenAIRouter{}, requestContext, store, episode
}

func assertARCEpisodeNotAdvanced(
	t *testing.T,
	store *raylinearc.MemoryEpisodeStore,
	episode string,
) {
	t.Helper()
	timeoutContext, cancel := context.WithTimeout(
		context.Background(),
		time.Second,
	)
	defer cancel()
	lease, state, err := store.Prepare(timeoutContext, episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = store.Abort(context.Background(), lease)
	}()
	if state.TurnIndex != 0 || state.PreviousArm != nil {
		t.Fatalf("aborted episode advanced: %#v", state)
	}
}

func arcResponseHeaders(
	statusCode string,
) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: statusCode},
				},
			},
		},
	}
}

// blockingRenewStore signals when a renewal starts and blocks until its
// context is cancelled, then returns that cancellation error.
type blockingRenewStore struct {
	*raylinearc.MemoryEpisodeStore
	entered chan struct{}
	once    sync.Once
}

func (store *blockingRenewStore) Renew(
	ctx context.Context,
	_ raylinearc.Lease,
) error {
	store.once.Do(func() { close(store.entered) })
	<-ctx.Done()
	return ctx.Err()
}

// TestCommitSucceedsWhileRenewalIsInFlight proves that cancelling an in-flight
// renewal during commit is treated as orderly shutdown, not a lost lease. The
// upstream returned 2xx, so the episode must advance.
func TestCommitSucceedsWhileRenewalIsInFlight(t *testing.T) {
	memory, err := raylinearc.NewMemoryEpisodeStore(
		raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	store := &blockingRenewStore{
		MemoryEpisodeStore: memory,
		entered:            make(chan struct{}),
	}
	episode := raylinearc.HashEpisodeID("renewal-race")
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}

	transaction := newRaylineARCEpisodeTransaction(
		store, lease, state, episode, 60*time.Millisecond, nil,
	)
	transaction.markSelection(1, 42)

	select {
	case <-store.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("renewal never started")
	}

	if commitErr := transaction.commit(
		context.Background(),
		&RequestContext{},
	); commitErr != nil {
		t.Fatalf("a valid 2xx commit was rejected as a lost lease: %v", commitErr)
	}

	// The committed turn must be visible to the next episode preparation.
	_, resumed, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.PreviousArm == nil || *resumed.PreviousArm != 1 {
		t.Fatalf("episode did not advance: %#v", resumed)
	}
}

func TestKnownLeaseLossBlocksRaylineARCDispatch(t *testing.T) {
	transaction := &raylineARCEpisodeTransaction{selectionReady: true}
	ctx := &RequestContext{RaylineARCTransaction: transaction}
	if !raylineARCDispatchAllowed(ctx) {
		t.Fatal("valid prepared transaction was blocked")
	}
	transaction.leaseLost.Store(true)
	if raylineARCDispatchAllowed(ctx) {
		t.Fatal("known-lost lease remained dispatchable")
	}
}
