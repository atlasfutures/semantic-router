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
	"github.com/google/uuid"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// arcFailureMissingEpisodeID is the bounded class for a request that names no
// episode. It is the caller's omission rather than a router failure, which is
// what separates it from every other preparation class.
const arcFailureMissingEpisodeID = "missing_episode_id"

// arcFailureNoCapableArm is the bounded class for a turn no arm in the basket
// can serve without changing what the model is shown. It answers 503 like
// every other authoritative selection failure, because the gateway replays 503
// and does not replay a body-level 400: a refusal here would end the turn,
// while a replayable status sends it to the fallback provider intact.
const arcFailureNoCapableArm = "no_capable_arm"

// arcFailureSelectorUnavailable is a request refused by a selector that
// construction left unarmed for good; unlike not_ready, it will not recover.
const arcFailureSelectorUnavailable = "selector_unavailable"

// requestedFault reads the failure this request asked for. It answers empty
// unless the cell opted in, so the header is inert everywhere else and no
// caller can make a serving cell refuse a request. It is read once, here, and
// every later phase reads the answer rather than the header.
func requestedFault(arc *config.RaylineARCAlgorithmConfig, reqCtx *RequestContext) string {
	if arc == nil || !arc.FaultInjection.Enabled {
		return ""
	}
	if strings.TrimSpace(reqCtx.Headers[headers.VSRFault]) == headers.FaultUpstreamDecode {
		return headers.FaultUpstreamDecode
	}
	return ""
}

// raylineARCEpisodeMode says what a request with no episode identity means.
type raylineARCEpisodeMode int

const (
	// raylineARCEpisodeRequired is the routed path. A turn with no episode
	// identity cannot be scored against a trajectory, and inventing one would
	// silently give every turn of a conversation its own history.
	raylineARCEpisodeRequired raylineARCEpisodeMode = iota
	// raylineARCEpisodeEphemeral is a route lookup that joins no trajectory.
	//
	// The episode identity is minted per call and never stored, so the lookup
	// reads and writes nothing: no lease, no episode store round trip, and no
	// key left behind. The selector falls back to a fresh in-memory episode,
	// which is the same state a conversation's first turn sees.
	raylineARCEpisodeEphemeral
)

func (r *OpenAIRouter) buildRaylineARCSelectionContext(
	algorithm *config.AlgorithmConfig,
	reqCtx *RequestContext,
	modelRefs []config.ModelRef,
	episodeMode raylineARCEpisodeMode,
) *selection.RaylineARCSelectionContext {
	if algorithm == nil ||
		algorithm.Type != config.RaylineARCAlgorithmType ||
		algorithm.RaylineARC == nil {
		return nil
	}
	result := &selection.RaylineARCSelectionContext{}
	if reqCtx == nil {
		result.PreparationFailure = "missing_request"
		return result
	}
	result.RequestID = reqCtx.RequestID
	reqCtx.InjectedFault = requestedFault(algorithm.RaylineARC, reqCtx)
	rawEpisodeID := strings.TrimSpace(
		reqCtx.Headers[algorithm.RaylineARC.Episode.IDHeader],
	)
	if rawEpisodeID == "" && episodeMode == raylineARCEpisodeRequired {
		// The refusal names the header, so the header name travels with the
		// request rather than being looked up again from config later.
		reqCtx.RaylineARCEpisodeIDHeader = algorithm.RaylineARC.Episode.IDHeader
		result.PreparationFailure = arcFailureMissingEpisodeID
		return result
	}
	if rawEpisodeID == "" {
		// An ephemeral identity is random rather than derived from anything
		// about the request. A hash derived from the turns would collide
		// across identical prompts and braid unrelated callers into one
		// encoder session; random cannot.
		result.EpisodeIDHash = raylinearc.HashEpisodeID(
			raylineARCEphemeralEpisodePrefix + uuid.NewString(),
		)
	} else {
		result.EpisodeIDHash = raylinearc.HashEpisodeID(rawEpisodeID)
	}
	if failure := parseRaylineARCCloseRequest(
		algorithm.RaylineARC.Episode.CloseHeader,
		reqCtx,
	); failure != "" {
		result.PreparationFailure = failure
		return result
	}
	// The policy service reads only the formats its contract names. Refusing
	// here, before the lease, leaves the episode untouched.
	if algorithm.RaylineARC.PolicyService != nil && policyRequestFormat(reqCtx.SourceFormat) == "" {
		result.PreparationFailure = arcFailurePolicyRequestFormat
		return result
	}
	// An ephemeral lookup skips preparation entirely, which is the whole of
	// what makes it free: no lease to acquire, renew or release, and nothing
	// written to the episode store. prepareSelection builds a fresh episode
	// state when it finds none here.
	// A side call commits nothing, so it is classified before the episode is
	// read and never takes or waits on the episode lease.
	var signalHeaders map[string]string
	if algorithm.RaylineARC.PolicyService != nil {
		signalHeaders = raylineARCTrustedSignalHeaders(algorithm.RaylineARC.PolicyService, reqCtx.Headers)
		result.PolicyCallKind, result.PolicyCallKindSource = raylineARCPolicyCallKindOfBody(
			signalHeaders, policyRequestFormat(reqCtx.SourceFormat), reqCtx.RaylineARCRawBody, result.EpisodeIDHash,
		)
	}
	if rawEpisodeID != "" {
		state, coalesced, failure := r.prepareOrJoinRaylineARCTurn(
			algorithm.RaylineARC,
			reqCtx,
			result.EpisodeIDHash,
			len(modelRefs),
			result.PolicyCallKind == raylinearc.PolicyCallSide,
		)
		if failure != "" {
			result.PreparationFailure = failure
			return result
		}
		result.State = state
		result.Coalesced = coalesced
		// Only an episode with no memory of its model reads the records a
		// two-stage package's refusal is answered from
		// (rayline_arc_derived_hold.go).
		if algorithm.RaylineARC.PolicyService != nil && result.PolicyCallKind != raylinearc.PolicyCallSide &&
			(state == nil || state.PreviousArm == nil) {
			result.ServedWorkers = r.raylineARCServedWorkers(reqCtx, result.EpisodeIDHash, signalHeaders)
		}
		// A policy-service turn records its ledger entry, so it commits only
		// once the client has the whole reply. Artifact-mode turns commit at
		// the response headers, as they always have.
		if algorithm.RaylineARC.PolicyService != nil && reqCtx.RaylineARCTransaction != nil {
			reqCtx.RaylineARCTransaction.commitOnCompletion = true
		}
	}
	turns, imageBearing, err := r.projectRaylineARCTurns(
		reqCtx,
		raylinearc.TurnOptions{
			IncludeSystemText: algorithm.RaylineARC.IncludeSystemText,
			DropMidConversationSystemText: algorithm.RaylineARC.
				DropMidConversationSystemText,
			IncludeToolNames: algorithm.RaylineARC.IncludeToolNames,
		},
	)
	if err != nil {
		code := raylinearc.TurnNormalizationErrorCode(err)
		if code == "" {
			code = "invalid_turns"
		}
		logRaylineARCTurnRejection(err, code)
		result.PreparationFailure = "turns_" + code
		r.finalizeRaylineARCAbort(reqCtx, result.PreparationFailure)
		return result
	}
	result.Turns = turns
	if algorithm.RaylineARC.PolicyService != nil {
		result.RawRequest = reqCtx.RaylineARCRawBody
		result.RequestFormat = policyRequestFormat(reqCtx.SourceFormat)
		result.PolicySignalHeaders = signalHeaders
		if result.RequestFormat == policyFormatResponses {
			input, instructions, err := r.raylineARCPolicyResponsesInput(reqCtx)
			if err != nil {
				result.PreparationFailure = "policy_request_body"
				if errors.Is(err, errPolicyResponsesHistoryUnavailable) {
					result.PreparationFailure = "turns_stored_history"
				}
				r.finalizeRaylineARCAbort(reqCtx, result.PreparationFailure)
				return result
			}
			result.PolicyInput, result.PolicyInstructions = input, instructions
		}
	}
	result.ImageBearing = imageBearing
	result.NonVisionArms = r.nonVisionArms(modelRefs)
	result.DisabledArms = r.disabledArms(modelRefs)
	result.RequiredCapabilities = requestRoutingCapabilities(reqCtx)
	result.IncapableArms = r.incapableArms(modelRefs, result.RequiredCapabilities)
	result.ToolLoopForeignArms, result.ToolLoopArm, result.ToolLoopFamily = r.toolLoopForeignArms(
		algorithm.RaylineARC, reqCtx, result.State, modelRefs,
	)
	return result
}

// raylineARCEphemeralEpisodePrefix keeps a minted identity out of the space a
// real caller's session identity could occupy, so the two can never hash to
// the same encoder session.
const raylineARCEphemeralEpisodePrefix = "rayline-arc-ephemeral:"

// requestRoutingCapabilities reads what this turn needs an arm to hold. It
// reads the neutral request rather than the body, so it sees the same tools
// and tool results the dispatch encoder will.
func requestRoutingCapabilities(reqCtx *RequestContext) []string {
	if reqCtx == nil || reqCtx.SemanticRequest == nil {
		return nil
	}
	return llmprotocol.RequiredRoutingCapabilities(*reqCtx.SemanticRequest)
}

// incapableArms reads the capability list off each candidate's model card. It
// returns nil when the turn requires nothing, which is almost every turn and
// leaves selection exactly as it was.
//
// The exclusion is a refusal rather than a degrade, for the same reason the
// vision one is: an arm without the capability does not answer this turn
// worse, it answers a different turn. A tool result that held a screenshot
// reaches it as an empty tool result, and the answer is built on less than the
// caller sent.
func (r *OpenAIRouter) incapableArms(modelRefs []config.ModelRef, required []string) []bool {
	if r == nil || r.Config == nil || len(required) == 0 || len(modelRefs) == 0 {
		return nil
	}
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		for _, capability := range required {
			if !known || !params.SupportsCapability(capability) {
				arms[index] = true
				break
			}
		}
	}
	return arms
}

// nonVisionArms reads the image-input contract off each candidate's model
// card. It returns nil when no candidate is marked, which is the unmarked
// default and leaves selection exactly as it was.
func (r *OpenAIRouter) nonVisionArms(modelRefs []config.ModelRef) []bool {
	if r == nil || r.Config == nil || len(modelRefs) == 0 {
		return nil
	}
	marked := false
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		if known && !params.SupportsVision() {
			arms[index] = true
			marked = true
		}
	}
	if !marked {
		return nil
	}
	return arms
}

// disabledArms reads the in-service verdict off each candidate's model card.
// It returns nil when no candidate is marked, which is the unmarked default
// and leaves selection exactly as it was.
func (r *OpenAIRouter) disabledArms(modelRefs []config.ModelRef) []bool {
	if r == nil || r.Config == nil || len(modelRefs) == 0 {
		return nil
	}
	marked := false
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		if known && params.IsDisabled() {
			arms[index] = true
			marked = true
		}
	}
	if !marked {
		return nil
	}
	return arms
}

// logRaylineARCTurnRejection records why an episode could not be prepared.
// The failure class alone is a bounded metric label and cannot name the
// offending construct, so the discriminator and its request path are emitted
// here instead. Neither carries request content.
func logRaylineARCTurnRejection(err error, code string) {
	fields := map[string]interface{}{
		"outcome":       "turns_rejected",
		"failure_class": code,
	}
	if path := raylinearc.TurnNormalizationErrorPath(err); path != "" {
		fields["request_path"] = path
	}
	if detail := raylinearc.TurnNormalizationErrorDetail(err); detail != "" {
		fields["detail"] = detail
	}
	logging.ComponentErrorEvent(
		"extproc",
		"rayline_arc_turn_normalize",
		fields,
	)
}

func parseRaylineARCCloseRequest(
	closeHeader string,
	reqCtx *RequestContext,
) string {
	if closeHeader == "" {
		return ""
	}
	switch strings.TrimSpace(reqCtx.Headers[closeHeader]) {
	case "", "false":
		reqCtx.RaylineARCCloseRequested = false
		return ""
	case "true":
		reqCtx.RaylineARCCloseRequested = true
		return ""
	default:
		return "invalid_close_signal"
	}
}

// raylineARCEpisodeStoreFor is the episode store of the recipe serving this
// request: a named recipe's own, or the default recipe's.
func (r *OpenAIRouter) raylineARCEpisodeStoreFor(reqCtx *RequestContext) raylinearc.EpisodeStore {
	if r == nil {
		return nil
	}
	if reqCtx != nil {
		if recipe := reqCtx.Routing.RecipeName(); recipe != "" && recipe != config.DefaultRecipeName {
			return r.RaylineARCRecipeEpisodeStores[recipe]
		}
	}
	return r.RaylineARCEpisodeStore
}

// prepareOrJoinRaylineARCTurn prepares this turn's episode transaction, unless
// an identical request on the same episode is already being decided. Then it
// waits for that decision and returns it with a borrowed transaction instead:
// a resend joins the turn in flight rather than contending for its lease.
//
// The wait is bounded by the request's own context, not acquire_timeout: the
// resend is waiting for its own turn, which may legitimately take as long as a
// cold decision does. If the first copy gives up without deciding, the resend
// starts over and may lead the next attempt.
func (r *OpenAIRouter) prepareOrJoinRaylineARCTurn(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	episodeIDHash string,
	workerCount int,
	sideCall bool,
) (*raylinearc.EpisodeState, *selection.SelectionResult, string) {
	store := r.raylineARCEpisodeStoreFor(reqCtx)
	if store == nil || len(reqCtx.RaylineARCRawBody) == 0 {
		state, failure := r.prepareRaylineARCTransaction(arcConfig, reqCtx, episodeIDHash, workerCount, sideCall)
		return state, nil, failure
	}
	waitContext := reqCtx.TraceContext
	if waitContext == nil {
		waitContext = context.Background()
	}
	key := raylineARCInflightKey(store, episodeIDHash, reqCtx.RaylineARCRawBody, raylineARCTurnInputs(arcConfig, reqCtx, r.CredentialResolver.HeadersToStrip()))
	for {
		entry, leader := r.raylineARCInflight.join(key)
		if leader {
			state, failure := r.prepareRaylineARCTransaction(arcConfig, reqCtx, episodeIDHash, workerCount, sideCall)
			if failure != "" {
				r.raylineARCInflight.finish(entry)
				return nil, nil, failure
			}
			reqCtx.RaylineARCInflight = entry
			reqCtx.RaylineARCTransaction.inflight = entry
			if !reqCtx.RaylineARCTransaction.takesHandovers() {
				entry.refuseHandovers()
			}
			reqCtx.RaylineARCTransaction.onFinalize = chainFinalize(
				reqCtx.RaylineARCTransaction.onFinalize,
				func() { r.raylineARCInflight.finish(entry) },
			)
			return state, nil, ""
		}
		decided, state, ok, err := entry.wait(waitContext)
		if err != nil {
			return nil, nil, boundedARCPrepareFailure(err)
		}
		if !ok {
			// The leader gave up without deciding: this request decides for
			// itself. Wait for the entry to leave the registry first, or the
			// next join would find it again.
			if err := entry.awaitFinished(waitContext); err != nil {
				return nil, nil, boundedARCPrepareFailure(err)
			}
			continue
		}
		reqCtx.RaylineARCTransaction = newBorrowedRaylineARCEpisodeTransaction(store, state, episodeIDHash, entry)
		bindRaylineARCSelectionTransaction(reqCtx)
		logging.ComponentEvent("extproc", "rayline_arc_selection_coalesced", map[string]interface{}{})
		return state, decided, ""
	}
}

// chainFinalize runs first, then next, on the transaction's terminal path.
func chainFinalize(first func(), next func()) func() {
	if first == nil {
		return next
	}
	return func() {
		first()
		next()
	}
}

func (r *OpenAIRouter) prepareRaylineARCTransaction(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	episodeIDHash string,
	workerCount int,
	sideCall bool,
) (*raylinearc.EpisodeState, string) {
	store := r.raylineARCEpisodeStoreFor(reqCtx)
	if store == nil {
		return nil, "episode_store"
	}
	if sideCall && !arcConfig.Episode.RelaxedConsistency() {
		return r.prepareSideCallRaylineARCTransaction(arcConfig, reqCtx, store, episodeIDHash, workerCount)
	}
	if arcConfig.Episode.RelaxedConsistency() {
		return r.prepareRelaxedRaylineARCTransaction(arcConfig, reqCtx, store, episodeIDHash, workerCount)
	}
	prepareContext := reqCtx.TraceContext
	if prepareContext == nil {
		prepareContext = context.Background()
	}
	prepareContext, cancel := context.WithTimeout(
		prepareContext,
		time.Duration(
			arcConfig.Episode.AcquireTimeoutSeconds,
		)*time.Second,
	)
	defer cancel()
	// The owning stream already holds this router open (processWithContext),
	// so the episode store cannot be closed underneath this lease.
	lease, state, err := store.Prepare(
		prepareContext,
		episodeIDHash,
		workerCount,
	)
	if err != nil {
		return nil, boundedARCPrepareFailure(err)
	}
	reqCtx.RaylineARCTransaction = newRaylineARCEpisodeTransaction(
		store,
		lease,
		state,
		episodeIDHash,
		time.Duration(
			arcConfig.Episode.LeaseTTLSeconds,
		)*time.Second,
		nil,
	)
	reqCtx.RaylineARCTransaction.closeRequested = reqCtx.RaylineARCCloseRequested
	reqCtx.RaylineARCTransaction.sessionCloser = r.raylineARCSessionClose
	reqCtx.RaylineARCTransaction.sessionCloseWait = time.Duration(
		arcConfig.Encoder.TotalTimeoutSeconds,
	) * time.Second
	bindRaylineARCSelectionTransaction(reqCtx)
	return state, ""
}

// prepareSideCallRaylineARCTransaction reads a strict episode for a side call
// without its lease. A side call commits nothing, so it has no write to
// serialize: it must neither wait behind a main turn's lease (a long stream
// holds it for minutes) nor block the next main turn. A failed read fails the
// request as a strict episode's failed prepare would.
func (r *OpenAIRouter) prepareSideCallRaylineARCTransaction(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	store raylinearc.EpisodeStore,
	episodeIDHash string,
	workerCount int,
) (*raylinearc.EpisodeState, string) {
	snapshots, ok := store.(raylinearc.EpisodeSnapshotStore)
	if !ok {
		return nil, "episode_store"
	}
	readContext := reqCtx.TraceContext
	if readContext == nil {
		readContext = context.Background()
	}
	readContext, cancel := context.WithTimeout(
		readContext,
		time.Duration(arcConfig.Episode.AcquireTimeoutSeconds)*time.Second,
	)
	defer cancel()
	state, _, err := snapshots.Snapshot(readContext, episodeIDHash, workerCount)
	if err != nil {
		return nil, boundedARCPrepareFailure(err)
	}
	reqCtx.RaylineARCTransaction = newSideCallRaylineARCEpisodeTransaction(state, episodeIDHash)
	bindRaylineARCSelectionTransaction(reqCtx)
	return state, ""
}

// projectRaylineARCTurns renders the decoded request into ARC turns.
//
// The router decodes every public wire format exactly once, so the selector
// reads the neutral messages rather than the wire body. ARC deliberately reads
// neither of the two flattened text fields the selection context also offers,
// the single query string and the prior-turn string list: the encoder was
// trained on role-tagged turns with tool calls flattened, and both of those
// fields carry a different shape.
func (r *OpenAIRouter) projectRaylineARCTurns(
	reqCtx *RequestContext,
	options raylinearc.TurnOptions,
) (turns []raylinearc.Turn, imageBearing bool, err error) {
	request, err := r.raylineARCConversation(reqCtx)
	if err != nil {
		return nil, false, err
	}
	turns, err = raylinearc.ProjectTurns(request, options)
	if err != nil {
		return nil, false, err
	}
	// The projection drops image blocks, so the image fact is read off the
	// same conversation before it is lost. The capability walker already
	// covers instruction blocks, message blocks and the blocks nested inside
	// a tool result, which is the whole of what the provider will receive.
	return turns, requestCarriesImageInput(request), nil
}

func requestCarriesImageInput(request *llmprotocol.Request) bool {
	if request == nil {
		return false
	}
	return llmprotocol.RequiredCapabilities(*request).
		Supports(llmprotocol.CapabilityImageInput)
}

// raylineARCConversation returns the whole conversation the selector must
// read, which is not always the request body.
//
// A Responses request that carries previous_response_id names its earlier
// turns instead of repeating them. The router retains those turns and replays
// them into the provider request, but only after selection has run. The
// selector would otherwise see a long session as a single question and route
// it as one, so the retained turns are prepended here.
//
// The retained turns are decoded through the same public codec as the body, so
// the projection reads one shape and not two.
func (r *OpenAIRouter) raylineARCConversation(
	reqCtx *RequestContext,
) (*llmprotocol.Request, error) {
	request := reqCtx.SemanticRequest
	state := reqCtx.ResponseObjectState
	if request == nil || state == nil ||
		len(state.ConversationHistory) == 0 ||
		// Materialization is idempotent by this flag. It is set on the
		// dispatch path, which runs after selection, so this guard only
		// matters if that order ever changes.
		state.ProviderContextApplied {
		return request, nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, storedHistoryFailure(err)
	}
	history, err := materializeStoredResponseHistory(
		engine,
		state.ConversationHistory,
	)
	if err != nil {
		return nil, storedHistoryFailure(err)
	}
	if len(history) == 0 {
		return request, nil
	}
	// A copy: the request the router dispatches is owned by the dispatch path,
	// and the selector must not widen it.
	merged := *request
	merged.Messages = make(
		[]llmprotocol.Message,
		0,
		len(history)+len(request.Messages),
	)
	merged.Messages = append(merged.Messages, history...)
	merged.Messages = append(merged.Messages, request.Messages...)
	return &merged, nil
}

// storedHistoryFailure fails the episode closed with a bounded class. Routing
// on the body alone would show the selector a truncated conversation, which is
// the failure this projection exists to prevent.
func storedHistoryFailure(err error) error {
	return &raylinearc.TurnNormalizationError{
		Code: "stored_history",
		Err:  err,
	}
}

func boundedARCPrepareFailure(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "episode_canceled"
	case errors.Is(err, raylinearc.ErrEpisodeLeaseHeld) &&
		errors.Is(err, context.DeadlineExceeded):
		return "episode_timeout"
	case errors.Is(err, context.DeadlineExceeded):
		// Out of time without ever seeing another owner: the store stalled.
		// Not contention, so it must not answer session_busy or 429.
		return "episode_store_timeout"
	case errors.Is(err, raylinearc.ErrEpisodeCapacity):
		return "episode_capacity"
	default:
		return "episode_store"
	}
}

// prepareRelaxedRaylineARCTransaction reads a relaxed episode without a lease.
// Nothing here waits on another turn, and nothing here fails the request over
// episode state: if the read fails, the turn decides from a fresh state, as a
// first turn would, and commits nothing.
func (r *OpenAIRouter) prepareRelaxedRaylineARCTransaction(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	store raylinearc.EpisodeStore,
	episodeIDHash string,
	workerCount int,
) (*raylinearc.EpisodeState, string) {
	snapshots, ok := store.(raylinearc.EpisodeSnapshotStore)
	if !ok {
		return nil, "episode_store"
	}
	readContext := reqCtx.TraceContext
	if readContext == nil {
		readContext = context.Background()
	}
	readContext, cancel := context.WithTimeout(
		readContext,
		time.Duration(arcConfig.Episode.AcquireTimeoutSeconds)*time.Second,
	)
	defer cancel()
	state, read, err := snapshots.Snapshot(readContext, episodeIDHash, workerCount)
	// A relaxed cell keeps serving while its store is down, so each read is
	// also the store's readiness probe.
	metrics.SetRaylineARCNamedComponentReady("episode_store", err == nil)
	stateless := false
	if err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_relaxed_read_failed", map[string]interface{}{
			"failure_class": boundedARCPrepareFailure(err),
		})
		state, err = raylinearc.NewEpisodeState(workerCount)
		if err != nil {
			return nil, "episode_store"
		}
		read, stateless = raylinearc.EpisodeReadToken{}, true
	}
	reqCtx.RaylineARCTransaction = newRelaxedRaylineARCEpisodeTransaction(
		snapshots,
		state,
		read,
		episodeIDHash,
		stateless,
	)
	bindRaylineARCSelectionTransaction(reqCtx)
	return state, ""
}
