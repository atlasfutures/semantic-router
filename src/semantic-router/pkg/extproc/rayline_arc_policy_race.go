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

// Held turns race a prepared decide against a deadline (pathfinder ADR 0155,
// proposed; arc_serving_contract.md, "Staged turns"). On a racing cell
// (policy_service.decide_mode: race):
//
//  1. Every strict main turn's decide asks the service to stage the turn.
//  2. When the turn commits -- the client has the whole 2xx reply -- and
//     before its episode lease is released, VSR appends the reply's
//     assistant output to the staged turn. The reply carries the service's
//     pick for the next turn, a fallback lease, and the staged prefix digest.
//  3. The next turn, if it extends that staged prefix, sends only its new
//     messages (turn_delta). If its new messages are tool results only and
//     the lease's action is in this turn's offer, the decide is raced
//     against race_ms: a decide that lands serves its own action, as today;
//     one that does not serves the lease's, and still completes inside the
//     episode lease, its pick logged as provisional.
//
// Nothing here fails a turn. A delta the service cannot extend is sent in
// full; a failed append leaves the next turn to decide synchronously.
//
// The fence is the service's, as for every decide. VSR waits for a call
// until its own timeout and no longer: a sync decide that times out fails
// its turn and releases the lease while the service may still be working.
// What keeps a later call from overlapping a call VSR stopped waiting for is
// the service's per-episode busy lock, which refuses the later call
// session_busy: a decide retries it for the episode's acquire timeout, and
// an append that meets it fails, which leaves nothing staged. So the staging
// bounds its waits inside the lease (policyStagingBudgets) for the next
// turn's sake, and a call it stops waiting for is fenced exactly as a timed
// out sync decide is.
//
// What a turn carries to the next is an allow-list: the staged entry holds
// only the service's digest and count, VSR's own digest of the same
// messages, the lease and the committed turn it belongs to. Every commit
// replaces it or drops it, so nothing outlives the turn that wrote it.

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Decision modes a racing cell records (selection.RaylineARCTrace.PolicyDecisionMode).
const (
	policyDecisionModeSync     = "sync"
	policyDecisionModeRaceWon  = "race_won"
	policyDecisionModeRaceLost = "race_lost"
)

// maxPolicyStagedTurns bounds the episodes a process keeps a staged turn for;
// the least recently staged is dropped first, and its next turn decides in
// full.
const maxPolicyStagedTurns = 4096

// policyStagedTurn is VSR's record of the service's staged turn after an
// append. It belongs to the episode state committed at turnIndex: any other
// turn count (another replica committed a turn, or this one failed to)
// makes it stale.
type policyStagedTurn struct {
	turnIndex     uint64
	contextEpoch  string
	requestFormat string
	// count and prefixDigest are the service's: the staged message count and
	// its digest, which a delta decide echoes. digest is VSR's own digest of
	// the same messages (raylinearc.MessagesDigest), which the next request
	// must reproduce over its first count messages.
	count        int
	prefixDigest string
	digest       string
	// appendedFrom is where the appended reply starts, and historyDigest
	// VSR's digest of the messages before it: a mismatch is then told apart
	// as the history's or the appended reply's.
	appendedFrom  int
	historyDigest string
	hold          *raylinearc.PolicyHold
	holdExpires   time.Time
	// shellDigest is the decided turn's shell (policyShellDigest): a lease
	// serves only a turn whose shell is the same.
	shellDigest string
}

// policyStagedTurns is the process's staged turns, keyed by episode. A
// turn reads and writes its episode's entry only under the episode lease.
type policyStagedTurns struct {
	mu      sync.Mutex
	entries map[string]*list.Element
	order   *list.List
}

type policyStagedEntry struct {
	episode string
	turn    policyStagedTurn
}

func newPolicyStagedTurns() *policyStagedTurns {
	return &policyStagedTurns{entries: map[string]*list.Element{}, order: list.New()}
}

func (turns *policyStagedTurns) get(episode string) (policyStagedTurn, bool) {
	if turns == nil {
		return policyStagedTurn{}, false
	}
	turns.mu.Lock()
	defer turns.mu.Unlock()
	element, ok := turns.entries[episode]
	if !ok {
		return policyStagedTurn{}, false
	}
	return element.Value.(*policyStagedEntry).turn, true
}

func (turns *policyStagedTurns) put(episode string, turn policyStagedTurn) {
	if turns == nil {
		return
	}
	turns.mu.Lock()
	defer turns.mu.Unlock()
	if element, ok := turns.entries[episode]; ok {
		element.Value.(*policyStagedEntry).turn = turn
		turns.order.MoveToFront(element)
		return
	}
	turns.entries[episode] = turns.order.PushFront(&policyStagedEntry{episode: episode, turn: turn})
	for turns.order.Len() > maxPolicyStagedTurns {
		oldest := turns.order.Back()
		turns.order.Remove(oldest)
		delete(turns.entries, oldest.Value.(*policyStagedEntry).episode)
	}
}

// dropIf drops the episode's entry when match holds of it.
func (turns *policyStagedTurns) dropIf(episode string, match func(policyStagedTurn) bool) {
	if turns == nil {
		return
	}
	turns.mu.Lock()
	defer turns.mu.Unlock()
	if element, ok := turns.entries[episode]; ok && match(element.Value.(*policyStagedEntry).turn) {
		turns.order.Remove(element)
		delete(turns.entries, episode)
	}
}

func (turns *policyStagedTurns) drop(episode string) {
	if turns == nil {
		return
	}
	turns.mu.Lock()
	defer turns.mu.Unlock()
	if element, ok := turns.entries[episode]; ok {
		turns.order.Remove(element)
		delete(turns.entries, episode)
	}
}

// policyAppender posts an append to where the turn was decided.
type policyAppender interface {
	Append(context.Context, raylinearc.PolicyAppendRequest) (*raylinearc.PolicyAppendResponse, error)
}

// policyPendingDecide is a decide running on its own context, so it can
// outlive the race and the request's stream. Its call holds its own
// admission slot (decidePolicyThroughBusy) and is bounded by the client's
// timeout and the busy wait.
type policyPendingDecide struct {
	done      chan struct{}
	response  *raylinearc.PolicyDecisionResponse
	err       error
	usedDelta bool
	latency   time.Duration
	mu        sync.Mutex
	after     []func(*policyPendingDecide)
}

func startPolicyPendingDecide(
	run func() (*raylinearc.PolicyDecisionResponse, error, bool),
) *policyPendingDecide {
	pending := &policyPendingDecide{done: make(chan struct{})}
	started := time.Now()
	go func() {
		response, err, usedDelta := run()
		pending.mu.Lock()
		pending.response, pending.err, pending.usedDelta = response, err, usedDelta
		pending.latency = time.Since(started)
		after := pending.after
		pending.after = nil
		close(pending.done)
		pending.mu.Unlock()
		for _, fn := range after {
			fn(pending)
		}
	}()
	return pending
}

// onDone runs fn once the decide has finished: now, if it already has.
func (pending *policyPendingDecide) onDone(fn func(*policyPendingDecide)) {
	pending.mu.Lock()
	select {
	case <-pending.done:
		pending.mu.Unlock()
		fn(pending)
		return
	default:
	}
	pending.after = append(pending.after, fn)
	pending.mu.Unlock()
}

// wait blocks until the decide finishes, bound passes or ctx ends,
// reporting whether the decide finished.
func (pending *policyPendingDecide) wait(ctx context.Context, bound time.Duration) bool {
	if pending == nil {
		return true
	}
	timer := time.NewTimer(bound)
	defer timer.Stop()
	select {
	case <-pending.done:
		return true
	case <-timer.C:
		return false
	case <-ctx.Done():
		return false
	}
}

// policyStaging is what a racing cell's strict main turn hands its episode
// transaction: the late decide to wait for, and what the append needs.
type policyStaging struct {
	turns         *policyStagedTurns
	appender      policyAppender
	session       *raylinearc.PolicySessionBinding
	admit         func() (func(), error)
	pkg           raylinearc.PolicyPackageRef
	episodeIDHash string
	requestID     string
	requestFormat string
	contextEpoch  string
	// messages and roles are the conversation the turn was decided on.
	messages []json.RawMessage
	roles    []string
	// shellDigest is the turn's shell (policyShellDigest).
	shellDigest string
	// pending is a raced decide whose race the lease won; nil otherwise.
	pending *policyPendingDecide
	// clientBody is the reply the client received, once it is complete.
	clientBody []byte
	// appendTimeout bounds the append.
	appendTimeout time.Duration
}

// policyRaceEligible reports whether this cell's turn may stage and race at
// all: a racing cell's strict main turn in a format the append can carry.
func (scorer *policyServiceScorer) policyRaceEligible(request *raylinearc.PolicyDecisionRequest, sideCall bool) bool {
	return scorer.race && !sideCall && request.EpisodeMode == "" &&
		request.Selection.Exploration == nil && policyAppendFormat(request.RequestFormat)
}

func policyAppendFormat(format string) bool {
	switch format {
	case policyFormatAnthropic, policyFormatOpenAI, policyFormatResponses:
		return true
	default:
		return false
	}
}

// policyDeltaFor is the staged turn this request extends, when it extends
// it: the same committed turn, epoch and format, more messages, and its
// first count messages reproducing the staged ones in VSR's prefix form.
func policyDeltaFor(
	scorer *policyServiceScorer,
	episodeIDHash string,
	state *raylinearc.EpisodeState,
	turn *raylinearc.PolicyEpisodeState,
	transition string,
	format string,
	messages []json.RawMessage,
) (policyStagedTurn, string) {
	if scorer.deltaUnsupported.Load() {
		return policyStagedTurn{}, "delta_unsupported"
	}
	staged, ok := scorer.staged.get(episodeIDHash)
	switch {
	case !ok:
		return policyStagedTurn{}, "none"
	case staged.turnIndex != state.TurnIndex:
		return policyStagedTurn{}, "stale_turn"
	case transition != raylinearc.PolicyTransitionExtends || turn == nil || staged.contextEpoch != turn.ContextEpoch():
		return policyStagedTurn{}, "new_context"
	case staged.requestFormat != format:
		return policyStagedTurn{}, "format"
	case len(messages) <= staged.count:
		return policyStagedTurn{}, "no_new_messages"
	case raylinearc.MessagesDigest(messages, staged.count) != staged.digest:
		if raylinearc.MessagesDigest(messages, staged.appendedFrom) != staged.historyDigest {
			return policyStagedTurn{}, "history_mismatch"
		}
		return policyStagedTurn{}, "appended_mismatch"
	}
	return staged, "delta"
}

// policyShellOf is a full request's shell: everything but its conversation.
func policyShellOf(request raylinearc.PolicyClientRequest) raylinearc.PolicyClientRequest {
	if request.Input != nil {
		return raylinearc.PolicyClientRequest{
			Shell: raylinearc.PolicyShellInput, Instructions: request.Instructions, Tools: request.Tools,
		}
	}
	return raylinearc.PolicyClientRequest{Shell: raylinearc.PolicyShellMessages, System: request.System, Tools: request.Tools}
}

// policyDeltaRefused reports a delta decide the service could not serve as
// one, which is sent in full instead: no staged turn it extends, a package
// that takes no delta, a request it read as malformed, a service that
// predates the member, or an answer that did not decode.
func policyDeltaRefused(err error) bool {
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) {
		return false
	}
	switch failure.Class {
	case raylinearc.PolicySessionMismatchClass, raylinearc.PolicyTurnDeltaUnsupportedClass,
		"invalid_request", "decode":
		return true
	case "status":
		return failure.Status == 404 || failure.Status == 405
	default:
		return false
	}
}

// policyLeaseNewMessagesAllowed is the visible half of "not a boundary": the
// messages after the leased prefix are tool results only. Anything else -- a
// typed user message, a system message, an image the user attached -- may be
// a stage-one boundary, which only a decide sees. Allow-list: an unknown
// shape is not a tool result.
//
// Claude Code sends a mid-conversation system message after each tool result
// (testdata/thread_continuation), so a system message of text only is
// allowed beside at least one tool result; one that carries anything else
// (a tool_addition, an image) is not.
func policyLeaseNewMessagesAllowed(format string, messages []json.RawMessage) bool {
	if len(messages) == 0 {
		return false
	}
	toolResults := 0
	for _, message := range messages {
		var envelope struct {
			Role    string          `json:"role"`
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(message, &envelope) != nil {
			return false
		}
		if envelope.Role == "system" && format != policyFormatResponses {
			if !policyTextOnlyContent(envelope.Content) {
				return false
			}
			continue
		}
		toolResults++
		switch format {
		case policyFormatOpenAI:
			if envelope.Role != "tool" {
				return false
			}
		case policyFormatResponses:
			if envelope.Type != "function_call_output" && envelope.Type != "custom_tool_call_output" {
				return false
			}
		case policyFormatAnthropic:
			var blocks []struct {
				Type string `json:"type"`
			}
			if envelope.Role != "user" || json.Unmarshal(envelope.Content, &blocks) != nil || len(blocks) == 0 {
				return false
			}
			for _, block := range blocks {
				if block.Type != "tool_result" {
					return false
				}
			}
		default:
			return false
		}
	}
	return toolResults > 0
}

// policyTextOnlyContent is a message content that is a string or text
// blocks only.
func policyTextOnlyContent(raw json.RawMessage) bool {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return true
	}
	var blocks []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &blocks) != nil || len(blocks) == 0 {
		return false
	}
	for _, block := range blocks {
		if block.Type != "text" {
			return false
		}
	}
	return true
}

// policyLeaseCheck is why a fallback lease may not serve this turn, "" when
// it may: the turn is no boundary VSR can see, extends the leased prefix (the
// delta matched it), and the lease is this package's and epoch's, unexpired,
// and names a bound action this turn offers (bound, healthy, not excluded,
// privacy-allowed, past the hard mask: the offer already removed the rest).
func policyLeaseCheck(
	scorer *policyServiceScorer,
	staged policyStagedTurn,
	signals raylinearc.PolicyTurnSignals,
	format string,
	newMessages []json.RawMessage,
	available []string,
	plainAsk bool,
	scheduleBoundary bool,
	shellDigest string,
	now time.Time,
) string {
	hold := staged.hold
	switch {
	case hold == nil:
		return "no_lease"
	case !plainAsk:
		return "derived_offer"
	case scheduleBoundary || signals.Compaction != nil || signals.Subagent != nil ||
		signals.CallKind == raylinearc.PolicyCallSide:
		return "boundary_signal"
	case !policyLeaseNewMessagesAllowed(format, newMessages):
		return "boundary_messages"
	case hold.ValidWhile.PackageSHA256 != scorer.sha256 || hold.ValidWhile.ContextEpoch != staged.contextEpoch ||
		hold.ValidWhile.PrefixDigest != staged.prefixDigest || hold.ValidWhile.Profile != nil:
		return "lease_unbound"
	case !now.Before(staged.holdExpires):
		return "lease_expired"
	case shellDigest != staged.shellDigest:
		return "shell_changed"
	}
	if _, bound := scorer.bindings[hold.ActionID]; !bound || !slices.Contains(available, hold.ActionID) {
		return "lease_not_offered"
	}
	return ""
}

// policyStagingBudgets bound the work a racing turn does inside its lease
// after the client may already have its reply: the rest of a lost race's
// decide (lateDecide, from the lease serving: a delta, its full resend and
// their busy retries) and the append. A next turn can arrive then and waits
// for the lease up to the episode's acquire timeout, so together they stay
// well inside it: the staging never makes that turn contended where a sync
// cell would have served it. Each is also no longer than a decide may take.
func policyStagingBudgets(total, acquire time.Duration) (appendTimeout, lateDecide time.Duration) {
	if acquire <= 0 {
		acquire = episodeFinalizeTimeout
	}
	if total <= 0 {
		total = episodeFinalizeTimeout
	}
	return raylineARCDurationMin(total, acquire/4), raylineARCDurationMin(2*total, acquire/2)
}

// policyLeaseOutcome names a lease check's result for the staged-turn log.
func policyLeaseOutcome(refusal string) string {
	if refusal == "" {
		return "eligible"
	}
	return refusal
}

// newPolicyStaging is what the turn's transaction needs to wait for its late
// decide and append its reply: the append goes where the decide went, the
// one client or the episode's session.
func (selector *raylineARCSelector) newPolicyStaging(
	armed *raylineARCArmedComponents,
	scorer *policyServiceScorer,
	arcContext *selection.RaylineARCSelectionContext,
	format string,
	shell raylinearc.PolicyClientRequest,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	roles []string,
	decider policyDecider,
	session *raylinearc.PolicySessionBinding,
	pending *policyPendingDecide,
) *policyStaging {
	var appender policyAppender
	if session != nil {
		appender = session
	} else if client, ok := decider.(*raylinearc.PolicyServiceClient); ok && client != nil {
		appender = client
	}
	return &policyStaging{
		turns:         scorer.staged,
		appender:      appender,
		session:       session,
		admit:         scorer.appendAdmit,
		pkg:           raylinearc.PolicyPackageRef{Alias: scorer.alias, PackageSHA256: scorer.sha256},
		episodeIDHash: arcContext.EpisodeIDHash,
		requestID:     arcContext.RequestID,
		requestFormat: format,
		shellDigest:   policyShellDigest(shell),
		contextEpoch:  turn.ContextEpoch(),
		messages:      messages,
		roles:         roles,
		pending:       pending,
		appendTimeout: scorer.appendTimeout,
	}
}

// appendAdmit takes a slot of the appends' own gate.
func (scorer *policyServiceScorer) appendAdmit() (func(), error) {
	if scorer.appendAdmission == nil {
		return func() {}, nil
	}
	return scorer.appendAdmission.Acquire()
}

// policyEstimatedTokens is the character-based estimate the router routes
// with, over the new messages' bytes.
func policyEstimatedTokens(messages []json.RawMessage) int {
	total := 0
	for _, message := range messages {
		total += len(message)
	}
	return estimatedTokensForBytes(total)
}

// policyLeaseResult is the turn served from the fallback lease: the lease's
// action on its worker, recorded as the action that served, with no scores
// (the policy scored nothing yet) and unknown token counts.
func (selector *raylineARCSelector) policyLeaseResult(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	hold *raylinearc.PolicyHold,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	atBoundary, retained bool,
	workerIDs []string,
) *selection.SelectionResult {
	binding := scorer.bindings[hold.ActionID]
	count := len(workerIDs)
	if binding.arm < 0 || binding.arm >= count {
		return nil
	}
	excludedArms := make([]bool, count)
	for arm := range excludedArms {
		excludedArms[arm] = arm != binding.arm
	}
	decision := raylinearc.Decision{
		SelectedArm:                 binding.arm,
		SelectedWorker:              workerIDs[binding.arm],
		RawScores:                   make([]float32, count),
		AdjustedScores:              make([]float32, count),
		SwitchCostUSD:               make([]float64, count),
		CacheMissTokens:             make([]int, count),
		ColdSwitchUpgradeExemptions: make([]bool, count),
		ExcludedArms:                excludedArms,
	}
	if !validARCDecision(decision, workerIDs) {
		return nil
	}
	result := selector.selectionResult(armed, selCtx, arcContext, state, &raylinearc.EncoderResult{}, decision, 0)
	result.RaylineARC.RawScores, result.RaylineARC.AdjustedScores = nil, nil
	result.RaylineARC.EncoderLatencyUnknown = true
	result.RaylineARC.TokenCountsUnknown = true
	result.Reasoning = "policy-service ARC decision (fallback_lease)"
	result.RaylineARC.PolicyActionID = hold.ActionID
	result.RaylineARC.PolicyArmID = hold.ArmID
	result.RaylineARC.ThinkingLevel = binding.level
	result.RaylineARC.PolicyActionModel = binding.model
	result.RaylineARC.WorkerProviderModel = scorer.workers[binding.arm].Model
	result.RaylineARC.WorkerRoute = scorer.routes[binding.arm]
	result.RaylineARC.PolicyPackageSHA256 = scorer.sha256
	result.RaylineARC.OfferKind = "lease"
	// The ledger records the action that served, whatever the late decide
	// picks; the next request's attribution says so to the service.
	result.RaylineARC.PolicyTurnState = turn.Clone()
	result.RaylineARC.PolicyNextState = turn.Next(messages, hold.ActionID, hold.ArmID)
	if atBoundary && !retained {
		result.RaylineARC.PolicyBoundary = raylinearc.NewPolicyBoundaryDecision(binding.arm, state.TurnIndex, turn, messages)
	}
	return result
}

// logPolicyLateDecide records a lost race's decide once it lands: its pick is
// provisional, the lease's action served, and the next append's attribution
// tells the service which.
func logPolicyLateDecide(requestID, episodeIDHash, served string, pending *policyPendingDecide) {
	fields := map[string]interface{}{
		"request_id":         requestID,
		"episode_id_hash":    episodeIDHash,
		"served_action_id":   served,
		"latency_ms":         pending.latency.Milliseconds(),
		"turn_delta":         pending.usedDelta,
		"decided_action_id":  nil,
		"lease_agrees":       nil,
		"provisional":        true,
		"failure_class":      nil,
		"policy_decide_mode": policyDecisionModeRaceLost,
	}
	if pending.err != nil {
		class := "transport"
		var failure *raylinearc.PolicyServiceError
		if errors.As(pending.err, &failure) {
			class = failure.Class
		}
		fields["failure_class"] = class
	} else if pending.response != nil {
		decided := pending.response.Decision.SelectedActionID
		fields["decided_action_id"] = decided
		fields["lease_agrees"] = decided == served
	}
	logging.ComponentEvent("extproc", "rayline_arc_policy_late_decide", fields)
}

// awaitPolicyStaging waits for a lost race's late decide, so the episode
// lease is released only once the service has finished the turn's call and
// the next turn cannot meet it as session_busy. A turn on a sessioned
// Server then persists the session the late decide ended on.
func (transaction *raylineARCEpisodeTransaction) awaitPolicyStaging() {
	staging := transaction.policyStaging
	if staging == nil || staging.pending == nil {
		return
	}
	// The decide runs under lateDecideBudget, so this wait ends.
	<-staging.pending.done
	if staging.session != nil {
		if session, changed := staging.session.Session(); changed {
			transaction.markPolicySession(session, false)
		}
	}
}

// appendPolicyStagedTurn posts this committed turn's assistant output to the
// staged turn and returns what the next turn may build on, or nil. It runs
// inside the episode lease, after the reply was sent, on its own context,
// timeout and admission slot. Every failure is logged and leaves nothing.
func (transaction *raylineARCEpisodeTransaction) appendPolicyStagedTurn(parent context.Context) *policyStagedTurn {
	staging := transaction.policyStaging
	if staging == nil || staging.appender == nil || transaction.policyNext == nil {
		return nil
	}
	fail := func(class string) *policyStagedTurn {
		logging.ComponentWarnEvent("extproc", "rayline_arc_policy_append_failed", map[string]interface{}{
			"request_id": staging.requestID, "episode_id_hash": staging.episodeIDHash, "class": class,
		})
		return nil
	}
	assistant, ok := policyAssistantOutput(staging.requestFormat, staging.clientBody)
	if !ok {
		return fail("no_assistant_output")
	}
	if policyReplyThinks(staging.requestFormat, staging.clientBody) {
		// No captured client shows how a thinking block is echoed (its
		// signature and display), so such a reply is not appended: the next
		// turn decides in full rather than staging a guess.
		return fail("thinking_turn")
	}
	roles := slices.Clone(staging.roles)
	if staging.requestFormat == policyFormatResponses {
		appended, err := policyResponsesRoles(assistant)
		if err != nil {
			return fail("no_assistant_output")
		}
		roles = append(roles, appended...)
	} else {
		roles = append(roles, "assistant")
	}
	messages := append(slices.Clone(staging.messages), assistant...)
	request := raylinearc.PolicyAppendRequest{
		SchemaVersion: raylinearc.PolicyAppendRequestSchema,
		Package:       staging.pkg,
		EpisodeIDHash: staging.episodeIDHash,
		ContextEpoch:  transaction.policyNext.ContextEpoch(),
		Messages:      assistant,
		Attribution:   transaction.policyNext.Attribution(messages, roles),
	}
	release, err := staging.admit()
	if err != nil {
		return fail("shed")
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), staging.appendTimeout)
	started := time.Now()
	response, err := staging.appender.Append(ctx, request)
	cancel()
	release()
	if err != nil {
		class := "transport"
		var failure *raylinearc.PolicyServiceError
		if errors.As(err, &failure) {
			class = failure.Class
		}
		return fail(class)
	}
	if response.Package != staging.pkg || response.Messages != len(messages) {
		return fail("append_mismatch")
	}
	staged := &policyStagedTurn{
		contextEpoch:  request.ContextEpoch,
		requestFormat: staging.requestFormat,
		count:         response.Messages,
		prefixDigest:  response.PrefixDigest,
		digest:        raylinearc.MessagesDigest(messages, len(messages)),
		appendedFrom:  len(staging.messages),
		shellDigest:   staging.shellDigest,
		historyDigest: raylinearc.MessagesDigest(messages, len(staging.messages)),
	}
	if hold := response.Hold; hold != nil {
		cloned := *hold
		staged.hold = &cloned
		staged.holdExpires = time.Now().Add(time.Duration(hold.TTLSeconds) * time.Second)
	}
	fields := map[string]interface{}{
		"request_id": staging.requestID, "episode_id_hash": staging.episodeIDHash,
		"messages": response.Messages, "latency_ms": time.Since(started).Milliseconds(),
		"session_action":  boundedPolicySessionAction(response.Encoding.SessionAction),
		"lease_action_id": nil,
	}
	if staged.hold != nil {
		fields["lease_action_id"] = staged.hold.ActionID
	}
	logging.ComponentEvent("extproc", "rayline_arc_policy_append", fields)
	return staged
}

// policyAssistantOutput is the reply's assistant output as the next request
// will carry it: Messages and Chat one assistant message, Responses the
// reply's output items. It is read from the reply the client received; the
// next turn's digest check is what proves the client echoed it so.
func policyAssistantOutput(format string, body []byte) ([]json.RawMessage, bool) {
	if len(body) == 0 {
		return nil, false
	}
	switch format {
	case policyFormatAnthropic:
		var reply struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(body, &reply) != nil || reply.Role != "assistant" || len(reply.Content) == 0 {
			return nil, false
		}
		content, ok := anthropicEchoContent(reply.Content)
		if !ok {
			return nil, false
		}
		message, err := raylinearc.MarshalUnescaped(struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}{"assistant", content})
		return []json.RawMessage{message}, err == nil
	case policyFormatOpenAI:
		var reply struct {
			Choices []struct {
				Message struct {
					Role      string          `json:"role"`
					Content   json.RawMessage `json:"content"`
					ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(body, &reply) != nil || len(reply.Choices) != 1 || reply.Choices[0].Message.Role != "assistant" {
			return nil, false
		}
		chosen := reply.Choices[0].Message
		if string(chosen.ToolCalls) == "null" {
			chosen.ToolCalls = nil
		}
		if len(chosen.Content) == 0 {
			chosen.Content = json.RawMessage("null")
		}
		message, err := raylinearc.MarshalUnescaped(chosen)
		return []json.RawMessage{message}, err == nil
	case policyFormatResponses:
		// Each item as the next turn materializes it (stored history and
		// resent input alike): through the snapshot type, without the
		// router's retention fields (raylineARCPolicyResponsesInput).
		var reply struct {
			Output []responseapi.OutputItem `json:"output"`
		}
		if json.Unmarshal(body, &reply) != nil || len(reply.Output) == 0 {
			return nil, false
		}
		items := make([]json.RawMessage, 0, len(reply.Output))
		for _, output := range reply.Output {
			output.ID, output.Status = "", ""
			encoded, err := raylinearc.MarshalUnescaped(output)
			if err != nil {
				return nil, false
			}
			items = append(items, raylinearc.UnescapeHTMLEscapes(encoded))
		}
		return items, true
	default:
		return nil, false
	}
}

// anthropicEchoContent is a Messages reply's content as Claude Code echoes
// it in the next request's history (testdata/thread_continuation, Claude
// Code 2.1.309): a text block carries "citations": [] where the reply had
// none, and a tool_use block "caller": {"type":"direct"} where it named
// none. Every other block and member is kept as the reply had it. The next
// turn's digest check is what proves the echo matched; a client that echoes
// otherwise costs only a full resend (delta_mismatch_reason
// appended_mismatch).
func anthropicEchoContent(raw json.RawMessage) (json.RawMessage, bool) {
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(raw, &blocks) != nil {
		return nil, false
	}
	for _, block := range blocks {
		var kind string
		_ = json.Unmarshal(block["type"], &kind)
		switch kind {
		case "text":
			if citations, ok := block["citations"]; !ok || string(citations) == "null" {
				block["citations"] = json.RawMessage(`[]`)
			}
		case "tool_use":
			if _, ok := block["caller"]; !ok {
				block["caller"] = json.RawMessage(`{"type":"direct"}`)
			}
		}
	}
	encoded, err := raylinearc.MarshalUnescaped(blocks)
	return encoded, err == nil
}

// policyReplyThinks reports whether the reply the client received holds
// reasoning, read before any normalization: Messages thinking or
// redacted_thinking blocks, a Chat message's reasoning members, a Responses
// reasoning item. An unreadable reply counts as thinking.
func policyReplyThinks(format string, body []byte) bool {
	switch format {
	case policyFormatAnthropic:
		var reply struct {
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		}
		if json.Unmarshal(body, &reply) != nil {
			return true
		}
		for _, block := range reply.Content {
			if block.Type == "thinking" || block.Type == "redacted_thinking" {
				return true
			}
		}
	case policyFormatOpenAI:
		var reply struct {
			Choices []struct {
				Message map[string]json.RawMessage `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(body, &reply) != nil {
			return true
		}
		for _, choice := range reply.Choices {
			for _, member := range []string{"reasoning", "reasoning_content", "reasoning_details"} {
				if value, ok := choice.Message[member]; ok && string(value) != "null" && string(value) != `""` && string(value) != "[]" {
					return true
				}
			}
		}
	case policyFormatResponses:
		var reply struct {
			Output []struct {
				Type string `json:"type"`
			} `json:"output"`
		}
		if json.Unmarshal(body, &reply) != nil {
			return true
		}
		for _, item := range reply.Output {
			if item.Type == "reasoning" {
				return true
			}
		}
	default:
		return true
	}
	return false
}

// policyShellDigest is the request's shell (system and tools; Responses
// instructions and tools) as the lease check compares it: cache_control
// dropped everywhere, and Claude Code's per-request billing attribution
// line dropped from system text, so only a shell that changed what the
// encoder reads ahead of the conversation differs.
func policyShellDigest(request raylinearc.PolicyClientRequest) string {
	instructions, _ := json.Marshal(request.Instructions)
	parts := []json.RawMessage{request.System, request.Tools, instructions}
	hash := sha256.New()
	for _, part := range parts {
		var value any
		decoder := json.NewDecoder(bytes.NewReader(part))
		decoder.UseNumber()
		if len(part) > 0 && decoder.Decode(&value) == nil {
			value = policyShellCanonical(value)
			if canonical, err := raylinearc.MarshalUnescaped(value); err == nil {
				part = canonical
			}
		}
		hash.Write([]byte(strconv.Itoa(len(part))))
		hash.Write([]byte{':'})
		hash.Write(part)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func policyShellCanonical(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		delete(typed, "cache_control")
		for key, member := range typed {
			typed[key] = policyShellCanonical(member)
		}
		return typed
	case []any:
		for index, element := range typed {
			typed[index] = policyShellCanonical(element)
		}
		return typed
	case string:
		if !strings.Contains(typed, policyBillingHeaderPrefix) {
			return typed
		}
		lines := strings.Split(typed, "\n")
		kept := lines[:0]
		for _, line := range lines {
			if !strings.HasPrefix(strings.TrimSpace(line), policyBillingHeaderPrefix) {
				kept = append(kept, line)
			}
		}
		return strings.Join(kept, "\n")
	default:
		return value
	}
}

// policyBillingHeaderPrefix starts Claude Code's per-request attribution
// line in its system prompt.
const policyBillingHeaderPrefix = "x-anthropic-billing-header:"

// stageRaylineARCAppendSource keeps the reply the client receives for the
// turn's append; a turn that stages nothing keeps nothing.
func stageRaylineARCAppendSource(ctx *RequestContext, clientBody []byte) {
	if ctx == nil || ctx.RaylineARCTransaction == nil || ctx.RaylineARCTransaction.policyStaging == nil {
		return
	}
	ctx.RaylineARCTransaction.policyStaging.clientBody = clientBody
}

// addPolicyDecisionModeFields records a racing cell's decision mode and race
// wait on a dispatch-time record; a sync cell's records are unchanged.
func addPolicyDecisionModeFields(fields map[string]interface{}, trace *selection.RaylineARCTrace) {
	if trace == nil || trace.PolicyDecisionMode == "" {
		return
	}
	fields["decision_mode"] = trace.PolicyDecisionMode
	if trace.PolicyDecisionMode != policyDecisionModeSync {
		fields["race_ms_observed"] = float64(trace.PolicyRaceWait.Microseconds()) / 1000
	}
}

// publishPolicyStagedTurn keeps what this turn's append staged, as of the
// turn count its commit writes, or drops the episode's entry: a turn that
// appended nothing leaves nothing for the next to build on. It runs before
// the commit, under the lease, so the next turn's prepare cannot read the
// committed state beside the previous entry; a commit that then fails takes
// the entry back (retractPolicyStagedTurn).
func (transaction *raylineARCEpisodeTransaction) publishPolicyStagedTurn(turnIndex uint64, staged *policyStagedTurn) {
	staging := transaction.policyStaging
	if staging == nil {
		return
	}
	if staged == nil {
		staging.turns.drop(staging.episodeIDHash)
		return
	}
	staged.turnIndex = turnIndex
	staging.turns.put(staging.episodeIDHash, *staged)
}

// retractPolicyStagedTurn drops the entry publishPolicyStagedTurn wrote
// when its commit failed, unless a later turn has replaced it.
func (transaction *raylineARCEpisodeTransaction) retractPolicyStagedTurn(staged *policyStagedTurn) {
	staging := transaction.policyStaging
	if staging == nil || staged == nil {
		return
	}
	staging.turns.dropIf(staging.episodeIDHash, func(current policyStagedTurn) bool {
		return current.turnIndex == staged.turnIndex && current.prefixDigest == staged.prefixDigest
	})
}

// dropPolicyStagedTurn forgets the episode's staged turn after a commit that
// did not happen: the service may hold an append the episode never recorded.
func (transaction *raylineARCEpisodeTransaction) dropPolicyStagedTurn() {
	if staging := transaction.policyStaging; staging != nil {
		staging.turns.drop(staging.episodeIDHash)
	}
}
