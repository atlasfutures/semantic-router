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
	"sync"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCEpisodeCommitsOnceOnFirst2xxHeaders(t *testing.T) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	requestContext.RaylineARCTransaction.markSelection(1, 123)
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
		state.Warmth[1].LastInputTokens != 123 {
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

// TestRouterCloseWaitsForInflightARCTransactions proves a hot reload cannot
// close the episode store while a prepared lease is still awaiting its
// terminal commit or abort.
func TestRouterCloseWaitsForInflightARCTransactions(t *testing.T) {
	closed := make(chan struct{})
	store, _ := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 4,
		IdleTTL:     time.Minute,
	})
	router := &OpenAIRouter{
		RaylineARCEpisodeStore: store,
		raylineARCEpisodeStoreClose: func() error {
			close(closed)
			return nil
		},
	}
	// Mirror RouterService.Process: the stream registers its hold on entry.
	if !router.tryHoldRaylineARC() {
		t.Fatal("hold refused on an open router")
	}

	returned := make(chan struct{})
	go func() {
		defer close(returned)
		_ = router.Close()
	}()

	select {
	case <-closed:
		t.Fatal("episode store closed while a transaction was in flight")
	case <-time.After(100 * time.Millisecond):
	}

	router.releaseRaylineARCHold()

	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return after the transaction finalized")
	}
	select {
	case <-closed:
	default:
		t.Fatal("episode store was never closed")
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

// TestHoldRefusedOnceCloseBegins proves a stream that captured a router which
// has started closing is told to retry rather than using a closed store.
func TestHoldRefusedOnceCloseBegins(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(
		raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 2, IdleTTL: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{
		RaylineARCEpisodeStore:      store,
		raylineARCEpisodeStoreClose: func() error { return nil },
	}
	if !router.tryHoldRaylineARC() {
		t.Fatal("hold refused before Close")
	}
	router.releaseRaylineARCHold()

	if err := router.Close(); err != nil {
		t.Fatal(err)
	}
	if router.tryHoldRaylineARC() {
		t.Fatal("hold granted on a closing router")
	}
}

func TestRaylineARCDrainTimeoutCoversConfiguredEncoderBudget(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		{
			Algorithm: &config.AlgorithmConfig{
				Type: config.RaylineARCAlgorithmType,
				RaylineARC: &config.RaylineARCAlgorithmConfig{
					Encoder: config.RaylineARCEncoderConfig{
						TotalTimeoutSeconds: 180,
					},
				},
			},
		},
	}
	got := configuredRaylineARCDrainTimeout(cfg)
	want := 180*time.Second + episodeFinalizeTimeout
	if got != want {
		t.Fatalf("drain timeout = %s, want %s", got, want)
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
