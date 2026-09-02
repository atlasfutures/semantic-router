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

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// blockingEncodeSelector stands in for the artifact-backed selector during a
// cold encode. A cold encode runs for minutes, so it is the whole window this
// gate is about: the episode lease is held and renewed the entire time, and no
// arm has been chosen yet.
type blockingEncodeSelector struct {
	entered chan struct{}
	once    sync.Once
}

func (s *blockingEncodeSelector) Select(
	ctx context.Context,
	_ *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	s.once.Do(func() { close(s.entered) })
	<-ctx.Done()
	return nil, ctx.Err()
}

func (s *blockingEncodeSelector) Method() selection.SelectionMethod {
	return selection.MethodRaylineARC
}

func (s *blockingEncodeSelector) Tier() selection.AlgorithmTier {
	return selection.TierSupported
}

func (s *blockingEncodeSelector) ExternalDependencies() []selection.Dependency {
	return nil
}

func (s *blockingEncodeSelector) UpdateFeedback(
	context.Context,
	*selection.Feedback,
) error {
	return nil
}

func shutdownARCAlgorithm() *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type:    config.RaylineARCAlgorithmType,
		OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{
				IDHeader:              "x-rayline-session",
				AcquireTimeoutSeconds: 10,
				LeaseTTLSeconds:       60,
			},
		},
	}
}

func shutdownARCSelectionContext() *selection.SelectionContext {
	return &selection.SelectionContext{
		DecisionName: "arc-decision",
		CandidateModels: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
		},
	}
}

func arcEpisodeTransactions(t *testing.T, outcome, failureClass string) float64 {
	t.Helper()
	return testutil.ToFloat64(
		metrics.RaylineARCEpisodeTransactions.WithLabelValues(
			outcome,
			failureClass,
		),
	)
}

// runShutdownMidEncode reproduces the shutdown ordering the CP7 gate names.
//
// The lease is already held and the renewal goroutine is running. Selection
// blocks in the encoder. The stream is then torn down the way a shutdown tears
// it down -- the request context is cancelled -- and the ext_proc loop exits,
// which is the only place left that can finalize the episode.
func runShutdownMidEncode(
	t *testing.T,
	router *OpenAIRouter,
	requestContext *RequestContext,
) {
	t.Helper()
	streamContext, cancelStream := context.WithCancel(context.Background())
	defer cancelStream()
	requestContext.TraceContext = streamContext

	selector := &blockingEncodeSelector{entered: make(chan struct{})}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, selector)
	router.ModelSelector = registry

	selectDone := make(chan error, 1)
	go func() {
		_, _, err := router.selectModelFromCandidates(
			shutdownARCSelectionContext(),
			shutdownARCAlgorithm(),
			requestContext,
		)
		selectDone <- err
	}()

	select {
	case <-selector.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the encoder call never started")
	}

	cancelStream()

	select {
	case err := <-selectDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf(
				"mid-encode cancellation returned %v, want context.Canceled",
				err,
			)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("selection never unblocked after the stream was cancelled")
	}

	// The ext_proc loop sees the cancelled stream and returns. Its deferred
	// finalizer is what must release the lease.
	_ = router.processWithContext(
		&MockStream{
			Ctx:       streamContext,
			RecvError: status.Error(codes.Canceled, "shutdown"),
		},
		requestContext,
	)
}

// TestARCShutdownMidEncodeReleasesTheLease closes the CP7 shutdown gate for the
// in-memory episode backend. CP6 only proved that a *completed* transaction is
// finalized on shutdown. Here no arm was ever chosen, so the episode must not
// advance, the lease must be released rather than left to expire, and the
// renewal goroutine must not report a lost lease: the lease was never lost,
// the request was.
func TestARCShutdownMidEncodeReleasesTheLease(t *testing.T) {
	router, requestContext, store, episode := newTestARCEpisodeTransaction(t)

	leaseLostBefore := arcEpisodeTransactions(t, "lease_lost", "renew")
	abortsBefore := arcEpisodeTransactions(t, "abort", "process_terminal")

	runShutdownMidEncode(t, router, requestContext)

	// Prepare is bounded, so a leaked lease fails here instead of hanging.
	assertARCEpisodeNotAdvanced(t, store, episode)

	if got := arcEpisodeTransactions(t, "lease_lost", "renew"); got != leaseLostBefore {
		t.Fatalf(
			"lease_lost transactions = %v, want %v: shutdown is not lease loss",
			got,
			leaseLostBefore,
		)
	}
	if got := arcEpisodeTransactions(t, "abort", "process_terminal"); got != abortsBefore+1 {
		t.Fatalf(
			"abort{process_terminal} transactions = %v, want %v",
			got,
			abortsBefore+1,
		)
	}
}

// TestARCShutdownMidEncodeDeletesTheRedisLeaseKey is the same gate against the
// backend the arc-redis cell actually runs. The in-memory store cannot show the
// difference between a released lease and one that merely expired, because it
// has no TTL on the lease itself. Redis does: the lease key must be gone
// immediately, not sixty seconds later.
func TestARCShutdownMidEncodeDeletesTheRedisLeaseKey(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	episode := raylinearc.HashEpisodeID(t.Name() + time.Now().String())
	prefix := "test:rayline-arc:" + episode + ":"
	store, err := raylinearc.NewRedisEpisodeStore(
		raylinearc.RedisEpisodeStoreConfig{
			Address:   address,
			KeyPrefix: prefix,
			LeaseTTL:  60 * time.Second,
			IdleTTL:   10 * time.Minute,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	lease, state, err := store.Prepare(context.Background(), episode, 2)
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
			60*time.Second,
			nil,
		),
	}

	runShutdownMidEncode(t, &OpenAIRouter{}, requestContext)

	if held := redisEpisodeLeaseHeld(t, address, prefix+episode+":lease"); held {
		t.Fatal("the lease key survived shutdown and will leak until its TTL")
	}
	// The episode is still readable and still on turn zero: the aborted turn
	// must not advance the trajectory the next request scores against.
	nextLease, nextState, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Abort(context.Background(), nextLease) }()
	if nextState.TurnIndex != 0 || nextState.PreviousArm != nil {
		t.Fatalf("aborted episode advanced: %#v", nextState)
	}
}

// redisEpisodeLeaseHeld reads the lease key directly, because the store API
// deliberately cannot tell a caller whether someone else holds a lease.
func redisEpisodeLeaseHeld(t *testing.T, address, key string) bool {
	t.Helper()
	client := redis.NewClient(&redis.Options{Addr: address})
	defer func() { _ = client.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	count, err := client.Exists(ctx, key).Result()
	if err != nil {
		t.Fatal(err)
	}
	return count == 1
}
