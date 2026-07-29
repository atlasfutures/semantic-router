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
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

const episodeFinalizeTimeout = 5 * time.Second

type raylineARCEpisodeTransaction struct {
	store            raylinearc.EpisodeStore
	lease            raylinearc.Lease
	state            *raylinearc.EpisodeState
	episodeIDHash    string
	leaseTTL         time.Duration
	selectedArm      int
	serializedTokens int
	selectionReady   bool
	finalizeOnce     sync.Once
	finalizeErr      error
	renewCancel      context.CancelFunc
	renewDone        chan struct{}
	leaseLost        atomic.Bool
	// onFinalize is an optional terminal-path hook; the stream-level hold in
	// processWithContext is what keeps the episode store open.
	onFinalize func()
}

func newRaylineARCEpisodeTransaction(
	store raylinearc.EpisodeStore,
	lease raylinearc.Lease,
	state *raylinearc.EpisodeState,
	episodeIDHash string,
	leaseTTL time.Duration,
	onFinalize func(),
) *raylineARCEpisodeTransaction {
	transaction := &raylineARCEpisodeTransaction{
		store:         store,
		lease:         lease,
		state:         state,
		episodeIDHash: episodeIDHash,
		leaseTTL:      leaseTTL,
		selectedArm:   -1,
		onFinalize:    onFinalize,
	}
	transaction.startRenewal()
	return transaction
}

func (transaction *raylineARCEpisodeTransaction) releaseHold() {
	if transaction.onFinalize != nil {
		transaction.onFinalize()
		transaction.onFinalize = nil
	}
}

func (transaction *raylineARCEpisodeTransaction) markSelection(
	selectedArm int,
	serializedTokens int,
) {
	if transaction == nil {
		return
	}
	transaction.selectedArm = selectedArm
	transaction.serializedTokens = serializedTokens
	transaction.selectionReady = true
}

// dispatchAllowed is the last pre-upstream fence. The renewal goroutine can
// discover lease loss after selection but before Envoy receives the request
// mutation; a known-lost lease must never dispatch and later masquerade as a
// committable turn.
func (transaction *raylineARCEpisodeTransaction) dispatchAllowed() bool {
	return transaction != nil &&
		transaction.selectionReady &&
		!transaction.leaseLost.Load()
}

func (transaction *raylineARCEpisodeTransaction) commit(
	ctx context.Context,
	requestContext *RequestContext,
) error {
	if transaction == nil {
		return nil
	}
	transaction.finalizeOnce.Do(func() {
		defer transaction.releaseHold()
		transaction.stopRenewal()
		if !transaction.selectionReady || transaction.leaseLost.Load() {
			transaction.finalizeErr = ErrRaylineARCEpisodeLeaseLost
			transaction.abortStore(ctx)
			return
		}
		nextState := cloneARCState(transaction.state)
		if err := nextState.Commit(
			transaction.selectedArm,
			transaction.serializedTokens,
			time.Now().UTC(),
		); err != nil {
			transaction.finalizeErr = err
			transaction.abortStore(ctx)
			return
		}
		transaction.finalizeErr = transaction.store.Commit(
			ctx,
			transaction.lease,
			transaction.lease.Version(),
			nextState,
		)
		if transaction.finalizeErr != nil {
			transaction.abortStore(ctx)
			return
		}
		transaction.state = nextState
		metrics.RecordRaylineARCEpisodeTransaction("commit", "")
		recordCommittedARCEpisodeTelemetry(requestContext, transaction)
	})
	return transaction.finalizeErr
}

func (transaction *raylineARCEpisodeTransaction) abort(
	ctx context.Context,
	class string,
) error {
	if transaction == nil {
		return nil
	}
	transaction.finalizeOnce.Do(func() {
		defer transaction.releaseHold()
		transaction.stopRenewal()
		transaction.finalizeErr = transaction.store.Abort(
			ctx,
			transaction.lease,
		)
		if errors.Is(transaction.finalizeErr, raylinearc.ErrEpisodeLeaseLost) {
			transaction.finalizeErr = nil
		}
		metrics.RecordRaylineARCEpisodeTransaction("abort", class)
	})
	return transaction.finalizeErr
}

func (transaction *raylineARCEpisodeTransaction) abortStore(
	ctx context.Context,
) {
	_ = transaction.store.Abort(ctx, transaction.lease)
	metrics.RecordRaylineARCEpisodeTransaction("abort", "commit_failure")
}

func (transaction *raylineARCEpisodeTransaction) startRenewal() {
	renewer, ok := transaction.store.(raylinearc.EpisodeLeaseRenewer)
	if !ok || transaction.leaseTTL <= 0 {
		return
	}
	renewContext, cancel := context.WithCancel(context.Background())
	transaction.renewCancel = cancel
	transaction.renewDone = make(chan struct{})
	interval := transaction.leaseTTL / 3
	if interval < 50*time.Millisecond {
		interval = 50 * time.Millisecond
	}
	if interval >= transaction.leaseTTL {
		interval = transaction.leaseTTL / 2
	}
	go func() {
		defer close(transaction.renewDone)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-renewContext.Done():
				return
			case <-ticker.C:
				timeoutContext, timeoutCancel := context.WithTimeout(
					renewContext,
					raylineARCDurationMin(
						interval,
						episodeFinalizeTimeout,
					),
				)
				err := renewer.Renew(timeoutContext, transaction.lease)
				timeoutCancel()
				if err != nil {
					if renewContext.Err() != nil {
						// stopRenewal cancelled this attempt mid-flight. That
						// is orderly shutdown, not a lost lease: treating it
						// as loss would abort a valid upstream-2xx commit.
						return
					}
					transaction.leaseLost.Store(true)
					metrics.RecordRaylineARCEpisodeTransaction(
						"lease_lost",
						"renew",
					)
					return
				}
			}
		}
	}()
}

func (transaction *raylineARCEpisodeTransaction) stopRenewal() {
	if transaction.renewCancel == nil {
		return
	}
	transaction.renewCancel()
	<-transaction.renewDone
	transaction.renewCancel = nil
}

func (r *OpenAIRouter) finalizeRaylineARCAbort(
	ctx *RequestContext,
	class string,
) {
	if ctx != nil && ctx.SelectionTransaction != nil {
		finalizeSelectionAbort(ctx, class)
		return
	}
	if ctx == nil || ctx.RaylineARCTransaction == nil {
		return
	}
	finalizeContext, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	if err := ctx.RaylineARCTransaction.abort(finalizeContext, class); err != nil {
		logging.ComponentErrorEvent(
			"extproc",
			"rayline_arc_episode_finalize",
			map[string]interface{}{
				"outcome":       "abort_failed",
				"failure_class": boundedARCEpisodeFailure(err),
			},
		)
	}
}

func boundedARCEpisodeFailure(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, raylinearc.ErrEpisodeLeaseLost),
		errors.Is(err, ErrRaylineARCEpisodeLeaseLost):
		return "lease_lost"
	default:
		return "store"
	}
}

func cloneARCState(
	state *raylinearc.EpisodeState,
) *raylinearc.EpisodeState {
	if state == nil {
		return nil
	}
	cloned := &raylinearc.EpisodeState{
		TurnIndex: state.TurnIndex,
		Warmth:    make([]*raylinearc.WorkerWarmth, len(state.Warmth)),
	}
	if state.PreviousArm != nil {
		value := *state.PreviousArm
		cloned.PreviousArm = &value
	}
	for index, warmth := range state.Warmth {
		if warmth == nil {
			continue
		}
		value := *warmth
		cloned.Warmth[index] = &value
	}
	return cloned
}

func recordCommittedARCEpisodeTelemetry(
	ctx *RequestContext,
	transaction *raylineARCEpisodeTransaction,
) {
	if ctx == nil || ctx.VSRRaylineARC == nil {
		return
	}
	sessiontelemetry.RecordSessionDecision(
		sessiontelemetry.SessionDecisionParams{
			SessionID:     transaction.episodeIDHash,
			SelectedModel: ctx.RequestModel,
			DecisionName:  ctx.VSRSelectedDecisionName,
			TurnIndex:     arcTurnIndexForTelemetry(transaction.state.TurnIndex),
			Timestamp:     time.Now().UTC(),
		},
	)
}

var ErrRaylineARCEpisodeLeaseLost = errors.New(
	"rayline ARC episode lease lost",
)

func raylineARCDurationMin(
	left time.Duration,
	right time.Duration,
) time.Duration {
	if left < right {
		return left
	}
	return right
}

func arcTurnIndexForTelemetry(value uint64) int {
	maximum := uint64(^uint(0) >> 1)
	if value > maximum {
		return int(maximum) // #nosec G115 -- maximum is the platform int bound.
	}
	return int(value) // #nosec G115 -- value is bounded above by maximum.
}
