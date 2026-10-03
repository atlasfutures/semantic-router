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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

const episodeFinalizeTimeout = 5 * time.Second

// relaxedBoundaryStageTimeout bounds storing a relaxed episode's boundary
// decision before dispatch.
const relaxedBoundaryStageTimeout = 250 * time.Millisecond

type raylineARCEpisodeTransaction struct {
	store         raylinearc.EpisodeStore
	lease         raylinearc.Lease
	state         *raylinearc.EpisodeState
	episodeIDHash string
	leaseTTL      time.Duration
	selectedArm   int
	// policyNext is the policy-service ledger and epoch to commit with this
	// turn; nil outside that mode.
	policyNext *raylinearc.PolicyEpisodeState
	// sideCall marks a call outside the main conversation. It commits
	// nothing: the turn count, previous arm, ledger and every per-turn
	// record stay as they were, and a strict lease is released unwritten.
	sideCall         bool
	serializedTokens int
	encoderOwner     string
	encoderVisited   []string
	closeRequested   bool
	sessionCloser    raylineARCSessionCloseFunc
	sessionCloseWait time.Duration
	selectionReady   bool
	// thinkingLedger is the lever ledger this turn staged, committed only
	// with the turn; nil leaves the stored ledger as it was.
	thinkingLedger *thinkinglever.Ledger
	// dispatchWorker is the worker this turn was dispatched to, and
	// opaqueReasoning the opaque reasoning blocks its response held; both
	// record, with the turn, which worker issued each block.
	dispatchWorker  string
	opaqueReasoning []string
	// upstreamPrefix is this turn's body shape, committed only with the turn.
	upstreamPrefix *raylinearc.UpstreamPrefix
	// controlPlacement is the thinking-control placer this turn advanced,
	// committed only with the turn.
	controlPlacement *raylinearc.ControlPlacement
	// reasoningIssuers is the issuer set this turn leaves behind, committed
	// only with the turn; unstaged leaves the stored set as it was.
	reasoningIssuers       []string
	reasoningIssuersStaged bool
	finalizeOnce           sync.Once
	finalizeErr            error
	renewCancel            context.CancelFunc
	renewDone              chan struct{}
	leaseLost              atomic.Bool
	// relaxed marks a turn on a relaxed episode: it read the episode without a
	// lease as read and commits only if the episode is still that. It
	// never fails the request over episode state: a lost race or a store error
	// drops this turn's state update and is counted.
	relaxed   bool
	read      raylinearc.EpisodeReadToken
	snapshots raylinearc.EpisodeSnapshotStore
	// stateless marks a relaxed turn whose read failed; it decided from a fresh
	// state and commits nothing.
	stateless bool
	// borrowed marks a resend that joined an identical in-flight turn: it holds
	// no lease, renews nothing and commits nothing, and its state is a
	// read-only copy of what the first copy prepared.
	borrowed bool
	// leaseless marks a side call on a strict episode: it read the episode
	// without the lease, and commits and releases nothing.
	leaseless bool
	// commitOnCompletion marks a policy-service turn, which commits only once
	// the client has the whole 2xx response. Every other turn commits at the
	// 2xx response headers, and its strict lease is released there.
	commitOnCompletion bool
	// onFinalize is an optional terminal-path hook; the stream-level hold in
	// processWithContext is what keeps the episode store open.
	onFinalize func()
	// inflight is the coalescing entry this turn leads or joined, through
	// which a refused resend hands its refusal to the lease owner.
	inflight *raylineARCInflightEntry
	// handedOver is a refusal the commit took from inflight and then could
	// not commit; the abort that follows stages it.
	handedOver *refusedTurn
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

// newRelaxedRaylineARCEpisodeTransaction is a relaxed episode's turn: no lease,
// no renewal, and a commit conditional on the version read.
func newRelaxedRaylineARCEpisodeTransaction(
	snapshots raylinearc.EpisodeSnapshotStore,
	state *raylinearc.EpisodeState,
	read raylinearc.EpisodeReadToken,
	episodeIDHash string,
	stateless bool,
) *raylineARCEpisodeTransaction {
	return &raylineARCEpisodeTransaction{
		snapshots:     snapshots,
		state:         state,
		read:          read,
		episodeIDHash: episodeIDHash,
		relaxed:       true,
		stateless:     stateless,
	}
}

// newBorrowedRaylineARCEpisodeTransaction is the transaction a coalesced
// resend dispatches under. It reads like the first copy's prepared one, so the
// resend renders the same controls and ledger, and finalizes to nothing.
func newBorrowedRaylineARCEpisodeTransaction(
	store raylinearc.EpisodeStore,
	state *raylinearc.EpisodeState,
	episodeIDHash string,
	inflight *raylineARCInflightEntry,
) *raylineARCEpisodeTransaction {
	return &raylineARCEpisodeTransaction{
		inflight: inflight,
		// The store is held only so a refused resend can clear the boundary
		// decision that chose the refusing arm; a borrowed turn writes nothing
		// else.
		store:         store,
		state:         state,
		episodeIDHash: episodeIDHash,
		borrowed:      true,
	}
}

// newSideCallRaylineARCEpisodeTransaction is a side call's transaction on a
// strict episode: a lease-free read that finalizes to nothing.
func newSideCallRaylineARCEpisodeTransaction(
	state *raylinearc.EpisodeState,
	episodeIDHash string,
) *raylineARCEpisodeTransaction {
	return &raylineARCEpisodeTransaction{
		state:         state,
		episodeIDHash: episodeIDHash,
		selectedArm:   -1,
		sideCall:      true,
		leaseless:     true,
	}
}

// selectionCommitsOnCompletion reports whether this request's turn commits
// at the end of the response rather than at its headers.
func selectionCommitsOnCompletion(ctx *RequestContext) bool {
	return ctx != nil && ctx.RaylineARCTransaction != nil && ctx.RaylineARCTransaction.commitOnCompletion
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
	transaction.markSelectionWithAffinity(
		selectedArm,
		serializedTokens,
		"",
		nil,
	)
}

func (transaction *raylineARCEpisodeTransaction) markSelectionWithAffinity(
	selectedArm int,
	serializedTokens int,
	encoderOwner string,
	encoderVisited []string,
) {
	if transaction == nil {
		return
	}
	transaction.selectedArm = selectedArm
	transaction.serializedTokens = serializedTokens
	if encoderOwner != "" {
		transaction.encoderOwner = encoderOwner
		transaction.encoderVisited = append([]string(nil), encoderVisited...)
	} else if transaction.state != nil {
		transaction.encoderOwner = transaction.state.EncoderOwner
		transaction.encoderVisited = append(
			[]string(nil),
			transaction.state.EncoderVisitedOwners...,
		)
	}
	transaction.selectionReady = true
}

// takesHandovers reports whether this turn, leading a coalesced group, takes a
// refused resend's hand-over as it finishes: only a strict main turn holds the
// lease and stages on commit or abort.
func (transaction *raylineARCEpisodeTransaction) takesHandovers() bool {
	return transaction != nil && !transaction.relaxed && !transaction.sideCall
}

// markPolicyState stages the policy-service ledger and epoch this turn
// commits, or marks the request a side call that commits nothing. It is a
// no-op outside the policy-service mode.
func (transaction *raylineARCEpisodeTransaction) markPolicyState(
	next *raylinearc.PolicyEpisodeState,
	sideCall bool,
) {
	if transaction == nil {
		return
	}
	transaction.sideCall = transaction.sideCall || sideCall
	if transaction.sideCall && transaction.inflight != nil {
		// A side call stages nothing as it finishes, so it takes no
		// hand-over.
		transaction.inflight.refuseHandovers()
	}
	if next != nil {
		transaction.policyNext = next.Clone()
	}
}

// stageRaylineARCPolicySelection stages what a policy-service decision
// commits with the turn and, at a schedule boundary, stores the decision
// before the request is dispatched.
func stageRaylineARCPolicySelection(ctx *RequestContext, trace *selection.RaylineARCTrace) {
	transaction := ctx.RaylineARCTransaction
	transaction.markPolicyState(trace.PolicyNextState, trace.PolicySideCall)
	if trace.PolicyBoundary == nil {
		return
	}
	parent := ctx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	stageContext, cancel := context.WithTimeout(parent, episodeFinalizeTimeout)
	defer cancel()
	transaction.retainPolicyBoundary(stageContext, trace.PolicyBoundary)
}

// retainPolicyBoundary stores the episode as prepared plus this boundary's
// decision, so a retry of the request reuses it if this one fails. The
// turn's commit replaces it. Storing it is best effort: a failure leaves the
// retry to decide again, as before, except that a strict lease found lost
// stops the dispatch, as any lost lease does.
func (transaction *raylineARCEpisodeTransaction) retainPolicyBoundary(
	ctx context.Context,
	boundary *raylinearc.PolicyBoundaryDecision,
) {
	if transaction == nil || transaction.state == nil || transaction.borrowed ||
		transaction.sideCall || transaction.stateless {
		return
	}
	staged := cloneARCState(transaction.state)
	staged.PolicyBoundary = raylinearc.ClonePolicyBoundary(boundary)
	if transaction.relaxed {
		transaction.retainRelaxedPolicyBoundary(ctx, staged)
		return
	}
	stager, ok := transaction.store.(raylinearc.EpisodeStateStager)
	if !ok {
		return
	}
	err := stager.Stage(ctx, transaction.lease, staged)
	switch {
	case errors.Is(err, raylinearc.ErrEpisodeLeaseLost):
		transaction.leaseLost.Store(true)
		metrics.RecordRaylineARCEpisodeTransaction("lease_lost", "stage")
	case err != nil:
		logging.ComponentWarnEvent("extproc", "rayline_arc_boundary_stage_failed", map[string]interface{}{
			"failure_class": boundedARCEpisodeFailure(err),
		})
	default:
		transaction.state = staged
	}
}

// retainRelaxedPolicyBoundary stores a relaxed episode's boundary decision
// with a conditional commit, then reads the episode back so the turn's own
// commit is conditional on the staged version. If anything else wrote in
// between, the turn keeps its original read and its commit loses, as any
// relaxed turn that raced does.
func (transaction *raylineARCEpisodeTransaction) retainRelaxedPolicyBoundary(
	parent context.Context,
	staged *raylinearc.EpisodeState,
) {
	// A relaxed turn never waits on episode state, so the stage gets a short
	// bound of its own; losing it only means a retry decides again.
	ctx, cancel := context.WithTimeout(parent, relaxedBoundaryStageTimeout)
	defer cancel()
	err := transaction.snapshots.CommitIfUnchanged(ctx, transaction.episodeIDHash, transaction.read, staged)
	if err != nil {
		metrics.RecordRaylineARCEpisodeTransaction("relaxed_dropped", "boundary_stage")
		logging.ComponentWarnEvent("extproc", "rayline_arc_boundary_stage_failed", map[string]interface{}{
			"failure_class": boundedARCEpisodeFailure(err), "relaxed": true,
		})
		return
	}
	_, read, err := transaction.snapshots.Snapshot(ctx, transaction.episodeIDHash, len(staged.Warmth))
	if err != nil || read.Version() != transaction.read.Version()+1 {
		return
	}
	transaction.read = read
	transaction.state = staged
}

// stageThinkingLedger records the ledger this turn's lever plan produced.
func (transaction *raylineARCEpisodeTransaction) stageThinkingLedger(ledger thinkinglever.Ledger) {
	if transaction == nil {
		return
	}
	transaction.thinkingLedger = ledger.Clone()
}

// stageUpstreamPrefix records the shape of this turn's provider-bound body.
func (transaction *raylineARCEpisodeTransaction) stageUpstreamPrefix(prefix raylinearc.UpstreamPrefix) {
	if transaction == nil {
		return
	}
	transaction.upstreamPrefix = &prefix
}

// stageControlPlacement records the placer state this turn's render left.
func (transaction *raylineARCEpisodeTransaction) stageControlPlacement(placement raylinearc.ControlPlacement) {
	if transaction == nil {
		return
	}
	cloned := raylinearc.CloneControlPlacements([]raylinearc.ControlPlacement{placement})[0]
	transaction.controlPlacement = &cloned
}

// committedControlPlacement returns the placer state the prepared state
// carries for key, and whether there is an episode at all.
func (transaction *raylineARCEpisodeTransaction) committedControlPlacement(
	key string,
) (thinkingcontrol.PlacerState, bool, bool) {
	if transaction == nil || transaction.state == nil {
		return thinkingcontrol.PlacerState{}, false, false
	}
	state, found := transaction.state.ControlPlacementFor(key)
	return state, found, true
}

// stageReasoningIssuers records the encrypted reasoning issuer set this turn
// leaves behind (raylinearc.NextReasoningIssuers).
func (transaction *raylineARCEpisodeTransaction) stageReasoningIssuers(issuers []string) {
	if transaction == nil {
		return
	}
	transaction.reasoningIssuers = append([]string(nil), issuers...)
	transaction.reasoningIssuersStaged = true
}

// committedThinking returns the ledger the prepared state carries, and the
// committed turn count the planner measures spacing against.
func (transaction *raylineARCEpisodeTransaction) committedThinking() (*thinkinglever.Ledger, uint64, bool) {
	if transaction == nil || transaction.state == nil {
		return nil, 0, false
	}
	return transaction.state.Thinking, transaction.state.TurnIndex, true
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
	transaction.observeOpaqueReasoning(requestContext)
	transaction.finalizeOnce.Do(func() {
		defer transaction.releaseHold()
		if transaction.borrowed {
			metrics.RecordRaylineARCEpisodeTransaction("coalesced", "")
			return
		}
		if transaction.sideCall {
			transaction.commitSideCall(ctx)
			return
		}
		if transaction.relaxed {
			transaction.commitRelaxed(ctx, requestContext)
			return
		}
		transaction.stopRenewal()
		if !transaction.selectionReady || transaction.leaseLost.Load() {
			transaction.finalizeErr = ErrRaylineARCEpisodeLeaseLost
			transaction.abortStore(ctx)
			return
		}
		nextState, err := transaction.nextState()
		if err != nil {
			transaction.finalizeErr = err
			transaction.abortStore(ctx)
			return
		}
		// A coalesced resend refused while this request held the lease
		// handed its refusal here; it is committed with this turn, as an
		// abort would stage it.
		refused := transaction.takeHandedOverRefusal()
		if refused != nil {
			if folded, changed := refused.apply(nextState); changed {
				nextState = folded
			}
		}
		transaction.closeRetainedEncoderSession(
			ctx,
			requestContext,
			nextState,
		)
		transaction.finalizeErr = transaction.store.Commit(
			ctx,
			transaction.lease,
			transaction.lease.Version(),
			nextState,
		)
		if transaction.finalizeErr != nil {
			// The abort stages the refusal instead.
			transaction.handedOver = refused
			transaction.abortStore(ctx)
			return
		}
		transaction.state = nextState
		metrics.RecordRaylineARCEpisodeTransaction("commit", "")
		recordCommittedARCEpisodeTelemetry(requestContext, transaction)
	})
	return transaction.finalizeErr
}

// commitSideCall finishes a side call without touching the episode. A strict
// episode's lease is released unwritten, so a concurrent relaxed turn's
// conditional commit is not invalidated by a version bump either.
func (transaction *raylineARCEpisodeTransaction) commitSideCall(ctx context.Context) {
	if !transaction.relaxed && !transaction.leaseless {
		transaction.stopRenewal()
		// The reply has been delivered and nothing is written, so a lease
		// that cannot be released is logged and left to expire rather than
		// failing a finished side call.
		err := transaction.store.Abort(ctx, transaction.lease)
		if err != nil && !errors.Is(err, raylinearc.ErrEpisodeLeaseLost) {
			logging.ComponentWarnEvent("extproc", "rayline_arc_side_call_release_failed", map[string]interface{}{
				"failure_class": boundedARCEpisodeFailure(err),
			})
		}
	}
	metrics.RecordRaylineARCEpisodeTransaction("commit", "side_call")
}

// commitRelaxed records a relaxed turn if the episode is still at the version
// it read. Nothing here fails the request: the turn has been served, and a
// lost race or a store error only drops its state update.
func (transaction *raylineARCEpisodeTransaction) commitRelaxed(
	ctx context.Context,
	requestContext *RequestContext,
) {
	if !transaction.selectionReady {
		return
	}
	if transaction.stateless {
		metrics.RecordRaylineARCEpisodeTransaction("relaxed_dropped", "stateless")
		return
	}
	nextState, err := transaction.nextState()
	if err != nil {
		metrics.RecordRaylineARCEpisodeTransaction("relaxed_dropped", "state")
		return
	}
	err = transaction.snapshots.CommitIfUnchanged(ctx, transaction.episodeIDHash, transaction.read, nextState)
	switch {
	case errors.Is(err, raylinearc.ErrEpisodeConflict):
		metrics.RecordRaylineARCEpisodeTransaction("relaxed_dropped", "conflict")
		logging.ComponentEvent("extproc", "rayline_arc_relaxed_conflict", map[string]interface{}{})
	case err != nil:
		metrics.RecordRaylineARCEpisodeTransaction("relaxed_dropped", "episode_store")
		logging.ComponentWarnEvent("extproc", "rayline_arc_relaxed_commit_failed", map[string]interface{}{"error": err.Error()})
	default:
		transaction.state = nextState
		metrics.RecordRaylineARCEpisodeTransaction("commit", "relaxed")
		recordCommittedARCEpisodeTelemetry(requestContext, transaction)
	}
}

// nextState is the episode after this turn: the prepared state with the turn's
// staging applied and the selection committed.
func (transaction *raylineARCEpisodeTransaction) nextState() (*raylinearc.EpisodeState, error) {
	nextState := cloneARCState(transaction.state)
	nextState.EncoderOwner = transaction.encoderOwner
	nextState.EncoderVisitedOwners = append(
		[]string(nil),
		transaction.encoderVisited...,
	)
	if transaction.policyNext != nil {
		nextState.Policy = transaction.policyNext.Clone()
	}
	if transaction.thinkingLedger != nil {
		nextState.Thinking = transaction.thinkingLedger.Clone()
	}
	if transaction.upstreamPrefix != nil {
		nextState.Upstream = raylinearc.WithUpstreamPrefix(nextState.Upstream, *transaction.upstreamPrefix)
	}
	if transaction.controlPlacement != nil {
		nextState.Controls = raylinearc.WithControlPlacement(nextState.Controls, *transaction.controlPlacement)
	}
	if len(transaction.opaqueReasoning) > 0 && transaction.dispatchWorker != "" {
		nextState.ReasoningProvenance = raylinearc.WithReasoningProvenance(
			nextState.ReasoningProvenance, transaction.dispatchWorker, transaction.opaqueReasoning,
		)
	}
	if transaction.reasoningIssuersStaged {
		nextState.ReasoningIssuers = append([]string(nil), transaction.reasoningIssuers...)
	}
	if err := nextState.Commit(
		transaction.selectedArm,
		transaction.serializedTokens,
		time.Now().UTC(),
	); err != nil {
		return nil, err
	}
	return nextState, nil
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
		if transaction.borrowed || transaction.relaxed || transaction.leaseless {
			return
		}
		transaction.stopRenewal()
		transaction.applyHandedOverRefusal(ctx)
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

// abortStore releases the lease after a commit that could not be completed.
//
// Detached from the caller's context on purpose, and this is the only place
// that can do it: finalizeOnce has already fired by the time this runs, so a
// later abort() is a no-op that returns the stored commit error without
// touching the store. If the commit failed BECAUSE its context was cancelled
// -- a client that disconnected mid-dispatch, or a route lookup that ran out
// of its deadline -- then aborting on that same cancelled context fails too,
// and the lease is neither committed nor released. It then sits until its TTL
// while every later turn on that session is refused as contended.
func (transaction *raylineARCEpisodeTransaction) abortStore(
	ctx context.Context,
) {
	abortContext, cancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		episodeFinalizeTimeout,
	)
	defer cancel()
	// A commit that failed leaves the turn unrecorded, as an abort does, so
	// a refusal a coalesced resend handed over is cleared here too.
	transaction.applyHandedOverRefusal(abortContext)
	_ = transaction.store.Abort(abortContext, transaction.lease)
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
		TurnIndex:            state.TurnIndex,
		Warmth:               make([]*raylinearc.WorkerWarmth, len(state.Warmth)),
		EncoderOwner:         state.EncoderOwner,
		EncoderVisitedOwners: append([]string(nil), state.EncoderVisitedOwners...),
		Policy:               state.Policy.Clone(),
		Thinking:             state.Thinking.Clone(),
		Upstream:             append([]raylinearc.UpstreamPrefix(nil), state.Upstream...),
		Controls:             raylinearc.CloneControlPlacements(state.Controls),
		ReasoningIssuers:     append([]string(nil), state.ReasoningIssuers...),
		ReasoningProvenance:  append([]raylinearc.ReasoningProvenance(nil), state.ReasoningProvenance...),
		PolicyBoundary:       raylinearc.ClonePolicyBoundary(state.PolicyBoundary),
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

// observeOpaqueReasoning records the opaque reasoning blocks this turn's
// response held, read from the response decoded for the client. A turn that
// commits at its headers has none decoded yet and records nothing.
func (transaction *raylineARCEpisodeTransaction) observeOpaqueReasoning(requestContext *RequestContext) {
	if transaction == nil || requestContext == nil || requestContext.SemanticResponse == nil {
		return
	}
	transaction.opaqueReasoning = protocolcodec.ResponseOpaqueReasoning(requestContext.SemanticResponse)
}
