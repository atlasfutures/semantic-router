package extproc

import (
	"bytes"
	"encoding/json"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// In-place re-serve of a refused turn (pathfinder#6066 P1c, "S2").
//
// A streamed ARC turn on a held family (claude by default) keeps its frames
// from the client -- the response start, pings, thinking -- until the first
// text or tool call, sending SSE comments in their place, as a host retry
// holds its call's frames (stream_host_retry.go). Every prod refusal to date
// came after thinking and before any text, so all of them were refused while
// the client had seen nothing it would have to unsee.
//
// When the model refuses while its frames are still held, the cell excludes
// the refusing model in the turn's context and its vendor family for the
// session, asks the policy again with the turn's own selection inputs (the
// same privacy mask, the same offer, the refused model now excluded), builds
// the dispatch for the arm it chose, and streams that call into the response
// the client already has open, through the host retry's splice. The turn
// then commits once, on the arm that served it. At most max_reserves
// re-serves run, and none starts after budget_seconds from the first
// refusal. A refusal after visible content, a turn past hold_max_seconds, or
// a refusal with no re-serve left is passed through as before (ADR 0120
// P1a: nothing commits and the model is excluded).
//
// The response headers went out before the refusal arrived, so they name the
// refusing arm. The served model and every attempt are stated in-band, on
// re-served turns only: one SSE comment, which every streamed client
// ignores, in the response's last body chunk ahead of its frames
//
//	: vsr-attempts {"v":1,"served_model":...,"attempts":[...]}
//
// The gateway reads it and strips it. The usage the protocol's own fields
// carry is the served attempt's alone.

const (
	refusalReserveRefused = "refused_attempt"
	refusalReserveServed  = "refusal_reserve"

	// refusalReserveKeepAlive is the longest the client goes without a
	// byte while frames are held.
	refusalReserveKeepAlive = 5 * time.Second

	refusalAttemptsCommentPrefix = ": vsr-attempts "
)

// raylineARCReselect is what a turn's ARC selection was made from, kept so a
// refused turn can be decided again in place.
type raylineARCReselect struct {
	selCtx       *selection.SelectionContext
	algorithm    *config.AlgorithmConfig
	decisionName string
	confidence   float64
}

// refusalReservePlan is a held turn's re-serve state.
type refusalReservePlan struct {
	maxReserves int
	budget      time.Duration
	hold        time.Duration
	families    []string
	// family is the vendor family of the arm in flight.
	family string
	// attempts are the refused calls so far, in order.
	attempts      []refusalAttempt
	firstRefusal  time.Time
	lastKeepAlive time.Time
	// released says the arm in flight's frames went to the client: its
	// first text or tool call arrived, the hold ran out, or the hold bound
	// was passed.
	released bool
	// attemptStarted is when the arm in flight was dispatched.
	attemptStarted time.Time
	// commented says the attempt record was sent.
	commented bool
	// redecide decides the turn again with the refused model excluded; nil
	// asks the turn's own selector (raylineARCReselect). Tests replace it.
	redecide func(*RequestContext) (*config.ModelRef, string, error)
}

// refusalAttempt is one call of a re-served turn, as the gateway reads it.
type refusalAttempt struct {
	Model         string               `json:"model"`
	SelectedModel string               `json:"selected_model,omitempty"`
	Outcome       string               `json:"outcome"`
	FailureClass  *string              `json:"failure_class"`
	Signal        *string              `json:"signal"`
	StopCategory  *string              `json:"stop_category"`
	SourceStop    *string              `json:"source_stop"`
	Usage         *refusalAttemptUsage `json:"usage"`
	CostUSD       *float64             `json:"cost_usd"`
	GenerationID  *string              `json:"generation_id"`
	family        string
}

type refusalAttemptUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

// keepRaylineARCReselect keeps an ARC selection's inputs for a re-serve.
func keepRaylineARCReselect(
	ctx *RequestContext, selCtx *selection.SelectionContext, algorithm *config.AlgorithmConfig,
	decisionName string, confidence float64,
) {
	if ctx == nil || selCtx == nil || selCtx.RaylineARC == nil || !raylineARCSelection(algorithm) ||
		algorithm.RaylineARC == nil || !algorithm.RaylineARC.PolicyService.RefusalReserveEnabled() {
		return
	}
	ctx.raylineARCReselect = &raylineARCReselect{
		selCtx: selCtx, algorithm: algorithm, decisionName: decisionName, confidence: confidence,
	}
}

// stageRefusalReserve holds a streamed ARC turn whose arm is of a held
// family, when the cell re-serves refused turns and the caller did not opt
// out. A re-serve's own dispatch keeps the plan it runs under.
func (r *OpenAIRouter) stageRefusalReserve(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil || ctx.RefusalReserve != nil {
		return
	}
	reselect := ctx.raylineARCReselect
	if reselect == nil || dispatch == nil || !ctx.ExpectStreamingResponse || ctx.LooperRequest ||
		ctx.RefusalFallbackOptOut || ctx.VSRRaylineARC == nil {
		return
	}
	// A Responses turn that carries retained history (previous_response_id
	// or a stored conversation) is not held: the rebuild from the client's
	// body would not materialize that history again. Such turns pass a
	// refusal through as before.
	if ctx.ResponseObjectState != nil {
		return
	}
	transaction := ctx.RaylineARCTransaction
	if transaction == nil || transaction.sideCall || transaction.borrowed || transaction.relaxed || transaction.stateless {
		return
	}
	policy := reselect.algorithm.RaylineARC.PolicyService
	maxReserves, budget, hold, families := policy.RefusalReserveLimits()
	family := raylinearc.ModelFamily(armModelOf(ctx))
	if !familyHeld(families, family) {
		return
	}
	ctx.RefusalReserve = &refusalReservePlan{
		maxReserves: maxReserves, budget: budget, hold: hold, families: families,
		family: family, attemptStarted: time.Now(),
	}
	// The hold keeps its frames in the host retry's plan, which also
	// carries the re-serve's splice: an arm with one host gets a plan that
	// never re-sends on a host of its own.
	if ctx.HostRetry == nil {
		ctx.HostRetry = &hostRetryPlan{
			dispatch: dispatch, dispatched: body, attempts: []hostAttempt{{started: ctx.StartTime}},
		}
	}
}

func familyHeld(families []string, family string) bool {
	if family == "" {
		return false
	}
	for _, held := range families {
		if held == family {
			return true
		}
	}
	return false
}

// holdingForRefusalReserve reports that the arm in flight's frames are held
// so a refusal before its first text or tool call can be re-served.
func (r *OpenAIRouter) holdingForRefusalReserve(ctx *RequestContext) bool {
	plan := ctx.RefusalReserve
	if plan == nil || plan.released || ctx.HostRetry == nil || ctx.HostRetry.exposed ||
		!ctx.FullDuplexResponseBody || !ctx.IsStreamingResponse || ctx.StreamingComplete {
		return false
	}
	if !ctx.StartTime.IsZero() && time.Since(ctx.StartTime) >= plan.hold {
		return false
	}
	return !streamShowsVisibleContent(ctx.SemanticStreamState)
}

// holdingResponse reports that the frames of the call in flight are held
// from the client, for a host retry or a refusal re-serve.
func (r *OpenAIRouter) holdingResponse(ctx *RequestContext) bool {
	return r.holdingForHostRetry(ctx) || r.holdingForRefusalReserve(ctx)
}

// noteRefusalReserveRelease records that a held arm's frames went out.
func (r *OpenAIRouter) noteRefusalReserveRelease(ctx *RequestContext) {
	plan := ctx.RefusalReserve
	if plan == nil || plan.released {
		return
	}
	plan.released = true
	metrics.RecordRaylineARCPreludeHold(plan.family, time.Since(plan.attemptStarted).Seconds())
	if len(plan.attempts) > 0 && !plan.firstRefusal.IsZero() {
		metrics.RecordRaylineARCRefusalReserveAdded(time.Since(plan.firstRefusal).Seconds())
	}
}

// streamShowsVisibleContent reports that the stream delivered text, a tool
// call or a carried item: content a re-serve would make the client unsee.
// Reasoning is not: a re-served turn's client sees the served model's.
func streamShowsVisibleContent(state *semanticResponseStreamState) bool {
	if state == nil {
		return false
	}
	for _, item := range state.items {
		if item.text != "" || item.toolCall != nil || item.carried != nil {
			return true
		}
	}
	return false
}

// streamRefusalSignal names how the stream refused, or "" when it has not.
func streamRefusalSignal(state *semanticResponseStreamState) string {
	if state == nil {
		return ""
	}
	for _, item := range state.items {
		if item.refused || item.refusal != "" {
			return "refusal_part"
		}
	}
	if state.stop == llmprotocol.StopContentFilter {
		// Anthropic's stop_reason refusal (with stop_details) and a Chat or
		// Responses content filter alike; source_stop tells them apart.
		return "content_filter"
	}
	failed := state.upstreamFailed
	if failed == nil {
		failed = state.failed
	}
	if failed != nil && streamFailureClass(failed) == turnFailureRefusal {
		return "provider_error_refusal"
	}
	return ""
}

// reserveRefusedTurn re-serves the turn in flight when its model refused
// while its frames were held, and reports whether a re-serve took over.
func (r *OpenAIRouter) reserveRefusedTurn(ctx *RequestContext) bool {
	plan := ctx.RefusalReserve
	if plan == nil || plan.released || ctx.HostRetry == nil || ctx.HostRetry.exposed || ctx.StreamingComplete ||
		!ctx.FullDuplexResponseBody {
		return false
	}
	if plan.holdExpired(ctx) {
		// Past the hold, the frames are the client's: nothing is re-served.
		plan.released = true
		return false
	}
	state := ctx.SemanticStreamState
	signal := ""
	visible := streamShowsVisibleContent(state)
	switch {
	case ctx.InjectedFault == headers.FaultRefusalPrelude && len(plan.attempts) == 0 && visible:
		// The dev fault: the first attempt refuses at its first visible
		// content, which the client has not been sent.
		signal = "fault_injection"
	case visible:
		return false
	default:
		signal = streamRefusalSignal(state)
	}
	if signal == "" {
		return false
	}
	refusing := armModelOf(ctx)
	family := raylinearc.ModelFamily(refusing)
	attempt := r.refusedAttemptOf(ctx, signal)
	if plan.firstRefusal.IsZero() {
		plan.firstRefusal = time.Now()
	}
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "refused_model": refusing, "family": family,
		"signal": signal, "attempt": len(plan.attempts) + 1,
		"hold_ms": time.Since(plan.attemptStarted).Milliseconds(),
	}
	if attempt.StopCategory != nil {
		fields["stop_category"] = *attempt.StopCategory
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	passThrough := func(outcome string) bool {
		fields["outcome"] = outcome
		logging.ComponentWarnEvent("extproc", "rayline_arc_refusal_reserve", fields)
		metrics.RecordRaylineARCRefusalReserve(outcome, family)
		// Nothing more is held: the refusal reaches the client as before,
		// and is settled once, on the turn's own line.
		plan.released = true
		return false
	}
	if len(plan.attempts) >= plan.maxReserves {
		return passThrough("reserves_spent")
	}
	if time.Since(plan.firstRefusal) >= plan.budget || r.responseStreamOverran(ctx) {
		return passThrough("budget_spent")
	}
	reselect := ctx.raylineARCReselect
	redecide := plan.redecide
	if redecide == nil && reselect != nil {
		redecide = func(ctx *RequestContext) (*config.ModelRef, string, error) {
			return r.selectModelFromCandidates(reselect.selCtx, reselect.algorithm, ctx)
		}
	}
	if redecide == nil {
		return passThrough("not_reselectable")
	}
	// A lease lost while the frames were held: another request may own the
	// episode, and a re-serve could never commit.
	if err := selectionDispatchAllowed(ctx); err != nil {
		return passThrough("lease_lost")
	}
	decisionName, confidence := ctx.VSRSelectedDecisionName, 0.0
	if reselect != nil {
		decisionName, confidence = reselect.decisionName, reselect.confidence
	}
	// The refused call's usage line is written only once a re-serve takes
	// over; a refusal passed through is settled by the turn's own line.
	refusedRecord := r.refusedReserveAttemptRecord(ctx, len(plan.attempts)+1)
	// Everything the re-decision and the rebuild change, so a re-serve that
	// does not start leaves the turn as the refused arm's.
	snapshot := snapshotReserveState(ctx, reselect)
	restore := func(outcome string) bool {
		snapshot.restore(ctx, reselect)
		return passThrough(outcome)
	}
	excludeRefusedModelInPlace(ctx, reselect)
	if reselect != nil && reselect.selCtx.RaylineARC != nil {
		if excluded := raylinearc.ModelFamily(refusing); excluded != "" && familyExclusionOn(ctx) {
			arcContext := reselect.selCtx.RaylineARC
			if !containsFold(arcContext.FamilyExclusions, excluded) {
				arcContext.FamilyExclusions = append(append([]string(nil), arcContext.FamilyExclusions...), excluded)
			}
		}
	}

	decideStarted := time.Now()
	selected, method, err := redecide(ctx)
	fields["redecide_ms"] = time.Since(decideStarted).Milliseconds()
	if err != nil || selected == nil {
		if err != nil {
			fields["redecide_error"] = err.Error()
		}
		return restore("no_eligible_arm")
	}
	model := selected.Model
	if selected.LoRAName != "" {
		model = selected.LoRAName
	}
	ctx.VSRSelectedModel = model
	ctx.VSRSelectionMethod = method
	if armModelOf(ctx) == refusing {
		// The policy chose the refusing model again (an exclusion that could
		// not be recorded): serving it again would only be refused again.
		return restore("no_eligible_arm")
	}
	reasoning := applyReasoningModeFromSelectedModel(selected, decisionName, confidence, ctx)

	dispatch, body, staged, unavailable := r.rebuildDispatchForReserve(ctx, model, decisionName, reasoning.UseReasoning)
	if dispatch == nil {
		fields["reason"] = unavailable
		return restore("dispatch_unavailable")
	}
	call, unavailable := r.buildResendCall(ctx, dispatch.logicalModel, dispatch, body)
	if call == nil {
		fields["reason"] = unavailable
		return restore("dispatch_unavailable")
	}
	if err := selectionDispatchAllowed(ctx); err != nil {
		return restore("lease_lost")
	}

	// The re-serve takes over: the refused call is settled on its own line,
	// its family excluded for the session, and the new call spliced in.
	emitLLMUsageRecord(refusedRecord)
	if excluded := noteRaylineARCFamilyExclusion(ctx, refusing); excluded != "" {
		fields["family_excluded"] = excluded
	}
	plan.attempts = append(plan.attempts, attempt)
	fields["outcome"] = "reserved"
	fields["served_model"] = model
	fields["served_family"] = raylinearc.ModelFamily(armModelOf(ctx))
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["served_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "rayline_arc_refusal_reserve", fields)
	metrics.RecordRaylineARCRefusalReserve("reserved", family)

	r.spliceReserve(ctx, dispatch, body, staged, call)
	plan.family = raylinearc.ModelFamily(armModelOf(ctx))
	plan.attemptStarted = time.Now()
	return true
}

// holdExpired reports that the hold ran past hold_max_seconds.
func (plan *refusalReservePlan) holdExpired(ctx *RequestContext) bool {
	return !ctx.StartTime.IsZero() && time.Since(ctx.StartTime) >= plan.hold
}

// familyExclusionOn reports that the turn's cell excludes refusing families.
func familyExclusionOn(ctx *RequestContext) bool {
	decision := ctx.VSRSelectedDecision
	return decision != nil && decision.Algorithm != nil && decision.Algorithm.RaylineARC != nil &&
		decision.Algorithm.RaylineARC.PolicyService.FamilyExclusionEnabled()
}

// reserveSnapshot is the turn as the refused arm left it.
type reserveSnapshot struct {
	trace                                       *selection.RaylineARCTrace
	selectedModel, method, reasoningMode, model string
	request                                     *llmprotocol.Request
	envelope                                    llmprotocol.Envelope
	targetFormat                                llmprotocol.WireFormat
	thinking                                    *plannedThinkingControl
	outbound                                    *dispatchedOutboundParams
	autoCache                                   *dispatchAutoCache
	serviceTier                                 string
	hostedTools                                 []string
	namesToolResults, toOpenRouter              bool
	transactionState, selectionState            *raylinearc.EpisodeState
	families                                    []string
}

func snapshotReserveState(ctx *RequestContext, reselect *raylineARCReselect) reserveSnapshot {
	snapshot := reserveSnapshot{
		trace: ctx.VSRRaylineARC, selectedModel: ctx.VSRSelectedModel, method: ctx.VSRSelectionMethod,
		reasoningMode: ctx.VSRReasoningMode, model: ctx.RequestModel,
		request: ctx.SemanticRequest, envelope: ctx.ProtocolEnvelope, targetFormat: ctx.TargetFormat,
		thinking: ctx.RaylineARCThinkingControl, outbound: ctx.DispatchedOutbound, autoCache: ctx.DispatchAutoCache,
		serviceTier: ctx.DispatchedServiceTier, hostedTools: ctx.DispatchHostedTools,
		namesToolResults: ctx.DispatchNamesToolResults, toOpenRouter: ctx.DispatchedToOpenRouter,
	}
	if transaction := ctx.RaylineARCTransaction; transaction != nil {
		snapshot.transactionState = transaction.state
	}
	if reselect != nil && reselect.selCtx != nil && reselect.selCtx.RaylineARC != nil {
		snapshot.selectionState = reselect.selCtx.RaylineARC.State
		snapshot.families = reselect.selCtx.RaylineARC.FamilyExclusions
	}
	return snapshot
}

// restore puts the refused arm's turn back: its trace, its selection in the
// transaction, its dispatch identity and the episode state as read, so the
// refusal passed through is excluded, staged and settled as it would have
// been without the attempt.
func (snapshot reserveSnapshot) restore(ctx *RequestContext, reselect *raylineARCReselect) {
	ctx.VSRRaylineARC = snapshot.trace
	ctx.VSRSelectedModel, ctx.VSRSelectionMethod = snapshot.selectedModel, snapshot.method
	ctx.VSRReasoningMode, ctx.RequestModel = snapshot.reasoningMode, snapshot.model
	ctx.SemanticRequest, ctx.ProtocolEnvelope, ctx.TargetFormat = snapshot.request, snapshot.envelope, snapshot.targetFormat
	ctx.RaylineARCThinkingControl, ctx.DispatchedOutbound = snapshot.thinking, snapshot.outbound
	ctx.DispatchAutoCache, ctx.DispatchedServiceTier = snapshot.autoCache, snapshot.serviceTier
	ctx.DispatchHostedTools, ctx.DispatchNamesToolResults = snapshot.hostedTools, snapshot.namesToolResults
	ctx.DispatchedToOpenRouter = snapshot.toOpenRouter
	ctx.FlexResend, ctx.RaylineARCEndpointResend, ctx.ThinkingSignatureRetry = nil, nil, nil
	if transaction := ctx.RaylineARCTransaction; transaction != nil {
		transaction.state = snapshot.transactionState
		if trace := snapshot.trace; trace != nil {
			transaction.markSelectionWithAffinity(trace.SelectedArm, trace.SerializedTokens,
				trace.EncoderReplicaID, trace.EncoderVisitedReplicaIDs)
			transaction.markPolicySession(trace.PolicySession, trace.PolicySessionCleared, trace.PolicySessionReplaced)
			stageRaylineARCPolicySelection(ctx, trace, snapshot.selectedModel)
		}
	}
	if reselect != nil && reselect.selCtx != nil && reselect.selCtx.RaylineARC != nil {
		reselect.selCtx.RaylineARC.State = snapshot.selectionState
		reselect.selCtx.RaylineARC.FamilyExclusions = snapshot.families
	}
}

// refusedAttemptOf is the refused call as the attempt record states it.
func (r *OpenAIRouter) refusedAttemptOf(ctx *RequestContext, signal string) refusalAttempt {
	class := turnFailureRefusal
	attempt := refusalAttempt{
		Model: dispatchedModel(ctx), SelectedModel: ctx.VSRSelectedModel, Outcome: "refused",
		FailureClass: &class, Signal: nonEmpty(signal), family: raylinearc.ModelFamily(armModelOf(ctx)),
	}
	attempt.fillFromStream(ctx.SemanticStreamState)
	return attempt
}

// fillFromStream reads what the call's stream stated about itself.
func (attempt *refusalAttempt) fillFromStream(state *semanticResponseStreamState) {
	if state == nil {
		return
	}
	if state.stopDetails != nil {
		attempt.StopCategory = nonEmpty(state.stopDetails.Category)
	}
	attempt.SourceStop = nonEmpty(state.sourceStop)
	attempt.GenerationID = nonEmpty(state.upstreamResponseID)
	usage := partialStreamUsage(state.usage)
	if usage.promptTokensReported || usage.completionTokensReported {
		// The record's fields are named as Anthropic's are, and mean what
		// Anthropic's mean: input_tokens is the UNCACHED input, the cache
		// reads and writes are counted apart from it. The stream's input
		// total includes both, so they come off it here; the gateway prices
		// the three as separate legs.
		uncached := usage.promptTokens - usage.cachedPromptTokens - usage.cacheWriteTokens
		if uncached < 0 {
			uncached = 0
		}
		attempt.Usage = &refusalAttemptUsage{
			InputTokens: int64(uncached), OutputTokens: int64(usage.completionTokens),
			CacheReadInputTokens: int64(usage.cachedPromptTokens), CacheCreationInputTokens: int64(usage.cacheWriteTokens),
		}
	}
	if charged := usage.providerCost.Charged; charged != nil {
		cost := *charged
		attempt.CostUSD = &cost
	}
}

// dispatchedModel is the provider model the call in flight was sent, the
// served-model identity the gateway prices (x-vsr-served-model).
func dispatchedModel(ctx *RequestContext) string {
	if identity, ok := raylineARCServedIdentityOf(ctx); ok {
		return identity.model
	}
	return ctx.VSRSelectedModel
}

// refusedReserveAttemptRecord is the refused call's usage line: what it
// stated it used, classed as a refusal, marked as a re-served turn's call.
// It is emitted only when a re-serve takes over; the turn's own failure
// state is left for the call that does.
func (r *OpenAIRouter) refusedReserveAttemptRecord(ctx *RequestContext, index int) llmUsageRecord {
	latency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		latency = time.Since(ctx.StartTime)
	}
	usage := responseUsageMetrics{}
	if ctx.SemanticStreamState != nil {
		usage = partialStreamUsage(ctx.SemanticStreamState.usage)
	}
	record := r.newLLMUsageRecord(ctx, usage)
	latencyMillis := latency.Milliseconds()
	record.CompletionLatencyMS = &latencyMillis
	record.FailureClass = nonEmpty(turnFailureRefusal)
	contentSent := false
	record.ContentSentBeforeFailure = &contentSent
	record.RefusalReserve = nonEmpty(refusalReserveRefused)
	record.RefusalReserveAttempt = &index
	r.priceUsageRecord(&record, ctx, ctx.RequestModel, usage)
	return record
}

// addRefusalReserveFields marks the served call of a re-served turn.
func addRefusalReserveFields(record *llmUsageRecord, ctx *RequestContext) {
	plan := ctx.RefusalReserve
	if plan == nil || len(plan.attempts) == 0 || record.RefusalReserve != nil {
		return
	}
	index := len(plan.attempts) + 1
	record.RefusalReserve = nonEmpty(refusalReserveServed)
	record.RefusalReserveAttempt = &index
}

// excludeRefusedModelInPlace adds the refusing model to the turn's context
// exclusions in the state the re-decision reads, and clears a retained
// boundary decision on the refused arm, as a refusal's abort would stage
// them; the turn that serves commits them.
func excludeRefusedModelInPlace(ctx *RequestContext, reselect *raylineARCReselect) {
	transaction := ctx.RaylineARCTransaction
	trace := ctx.VSRRaylineARC
	if transaction == nil || transaction.state == nil || trace == nil {
		return
	}
	refusal := refusedTurn{
		arm: trace.SelectedArm, exclude: armModelOf(ctx),
		context: trace.PolicyTurnState, decidedFrom: transaction.storedPolicy(), committed: trace.PolicyNextState,
	}
	next, changed := refusal.apply(transaction.state)
	if !changed {
		return
	}
	transaction.state = next
	if reselect != nil && reselect.selCtx != nil && reselect.selCtx.RaylineARC != nil {
		reselect.selCtx.RaylineARC.State = next
	}
}

// armModelOf is the model of the arm in flight: its action's trained model,
// or its worker's.
func armModelOf(ctx *RequestContext) string {
	if ctx.VSRRaylineARC != nil {
		return refusedModel(ctx)
	}
	return ctx.VSRSelectedModel
}

// rebuildDispatchForReserve builds the dispatch of the re-decided arm from
// the client's own body, as the request path built the first: decoded
// again, the pre-dispatch tools policy, the arm's model, reasoning, pins and
// output bound. It returns the dispatch, its encoded body and the host retry
// plan the new arm's pin stages (nil when none), or why there is none. The
// turn's own hold and splice plan stays installed until the splice starts.
func (r *OpenAIRouter) rebuildDispatchForReserve(
	ctx *RequestContext, model, decisionName string, useReasoning bool,
) (*providerDispatch, []byte, *hostRetryPlan, string) {
	engine, err := r.protocolEngine()
	if err != nil || len(ctx.RaylineARCRawBody) == 0 {
		return nil, nil, nil, "protocol_unavailable"
	}
	request, envelope, _, err := engine.DecodeRequestForMutation(ctx.SourceFormat, ctx.RaylineARCRawBody)
	if err != nil {
		return nil, nil, nil, "decode_failed"
	}
	request.Trusted.SourceFormat = ctx.SourceFormat
	request.Trusted.CorrelationID = ctx.RequestID
	ctx.SemanticRequest = &request
	ctx.ProtocolEnvelope = envelope
	if _, err := r.applyPreDispatchToolsPolicy(ctx); err != nil {
		return nil, nil, nil, "tools_policy_failed"
	}
	// The plans the first dispatch staged answer at the response headers,
	// which are gone; the re-serve's own staging must not replace the hold
	// and splice the turn runs under.
	held := ctx.HostRetry
	defer func() { ctx.HostRetry = held }()
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, model, decisionName, useReasoning, ctx)
	if err != nil {
		return nil, nil, nil, "prepare_failed"
	}
	response := r.buildProviderDispatchResponse(dispatch, ctx)
	if response.GetRequestBody() == nil {
		return nil, nil, nil, "credential_unavailable"
	}
	response, err = r.completeProviderDispatch(ctx.SemanticRequest, dispatch, response, ctx)
	staged := ctx.HostRetry
	ctx.FlexResend, ctx.RaylineARCEndpointResend, ctx.ThinkingSignatureRetry = nil, nil, nil
	if err != nil || response == nil {
		return nil, nil, nil, "complete_failed"
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	if len(body) == 0 {
		return nil, nil, nil, "empty_body"
	}
	ctx.RequestModel = model
	if staged == held {
		staged = nil
	}
	return dispatch, body, staged, ""
}

// spliceReserve streams the re-served call into the open response. The new
// arm gets a plan of its own -- the host retry its pin staged, or a fresh
// one that never re-sends on a host -- so nothing of the refused arm's
// hosts carries over; only the original call's pending trailers do.
func (r *OpenAIRouter) spliceReserve(
	ctx *RequestContext, dispatch *providerDispatch, body []byte, staged *hostRetryPlan, call *flexTwinCall,
) {
	old := ctx.HostRetry
	plan := staged
	if plan == nil {
		plan = &hostRetryPlan{}
	}
	if old != nil {
		old.stop()
		plan.trailersPending = old.trailersPending
	}
	ctx.HostRetry = plan
	plan.dispatch = dispatch
	plan.dispatched = body
	plan.attempts = []hostAttempt{{started: time.Now()}}
	plan.held, plan.exposed, plan.exhausted = nil, false, false
	ctx.StreamContentSeen, ctx.StreamFirstContentAt = false, time.Time{}
	ctx.DeliveredStreamItems = 0
	resetStreamForHostRetry(ctx, body, dispatch)
	r.initializeSemanticResponseStream(ctx)
	plan.splice = startHostRetrySplice(ctx, call)
}

// withRefusalAttemptsComment puts the attempt record ahead of the last body
// chunk of a re-served turn.
func withRefusalAttemptsComment(ctx *RequestContext, response *ext_proc.ProcessingResponse) *ext_proc.ProcessingResponse {
	plan := ctx.RefusalReserve
	if plan == nil || len(plan.attempts) == 0 || plan.commented || !ctx.StreamingComplete {
		return response
	}
	streamed := response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	if streamed == nil {
		return response
	}
	comment := refusalAttemptsComment(ctx)
	if comment == nil {
		return response
	}
	plan.commented = true
	streamed.Body = append(comment, streamed.Body...)
	return response
}

// refusalAttemptsComment is the record of a re-served turn's attempts.
func refusalAttemptsComment(ctx *RequestContext) []byte {
	plan := ctx.RefusalReserve
	served := refusalAttempt{
		Model: dispatchedModel(ctx), SelectedModel: ctx.VSRSelectedModel, Outcome: "served",
	}
	if signal := streamRefusalSignal(ctx.SemanticStreamState); signal != "" {
		class := turnFailureRefusal
		served.Outcome, served.FailureClass, served.Signal = "refused", &class, nonEmpty(signal)
	} else if ctx.ResponseFailureClass != "" {
		served.Outcome, served.FailureClass = "failed", nonEmpty(ctx.ResponseFailureClass)
	}
	served.fillFromStream(ctx.SemanticStreamState)
	record := struct {
		V           int              `json:"v"`
		ServedModel string           `json:"served_model"`
		Attempts    []refusalAttempt `json:"attempts"`
	}{V: 1, ServedModel: served.Model, Attempts: append(append([]refusalAttempt(nil), plan.attempts...), served)}
	encoded, err := json.Marshal(record)
	if err != nil {
		return nil
	}
	// One SSE comment line: the record has no newline, and it ends the
	// comment with the blank line that ends an SSE block.
	encoded = bytes.ReplaceAll(encoded, []byte("\n"), nil)
	comment := append([]byte(refusalAttemptsCommentPrefix), encoded...)
	return append(comment, '\n', '\n')
}

// refusalReserveKeepAliveDue reports that a held turn's client is owed a
// keepalive: nothing was sent for refusalReserveKeepAlive.
func (r *OpenAIRouter) refusalReserveKeepAliveDue(ctx *RequestContext) bool {
	plan := ctx.RefusalReserve
	if plan == nil || !r.holdingForRefusalReserve(ctx) && !ctx.HostRetry.started() || plan.released {
		return false
	}
	return time.Since(plan.lastKeepAliveOr(ctx.StartTime)) >= refusalReserveKeepAlive
}

func (plan *refusalReservePlan) lastKeepAliveOr(start time.Time) time.Time {
	if plan.lastKeepAlive.IsZero() {
		return start
	}
	return plan.lastKeepAlive
}

// refusalReserveKeepAliveWait is how long the loop may wait before a held
// turn's next keepalive, or 0 when none is owed.
func (r *OpenAIRouter) refusalReserveKeepAliveWait(ctx *RequestContext) time.Duration {
	plan := ctx.RefusalReserve
	// Only once the body is under way: the first body reply follows the
	// upstream's first chunk, as every other one does.
	if plan == nil || plan.released || !streamSilenceTimerArmed(ctx) || !ctx.IsStreamingResponse || ctx.StreamingComplete {
		return 0
	}
	// Past the hold nothing is held, and no keepalive is owed unless a
	// re-serve is waiting on its first bytes.
	if plan.holdExpired(ctx) && !ctx.HostRetry.started() {
		return 0
	}
	wait := refusalReserveKeepAlive - time.Since(plan.lastKeepAliveOr(ctx.StartTime))
	if wait < 10*time.Millisecond {
		wait = 10 * time.Millisecond
	}
	return wait
}

// sendRefusalReserveKeepAlive sends a held turn's client an SSE comment.
func sendRefusalReserveKeepAlive(stream ext_proc.ExternalProcessor_ProcessServer, ctx *RequestContext) error {
	ctx.RefusalReserve.lastKeepAlive = time.Now()
	response := buildResponseBodyContinueResponse(responseStreamBodyMutation(ctx, bytes.Clone(hostRetryHoldFrame), false), nil)
	return sendResponse(stream, response, "response body")
}
