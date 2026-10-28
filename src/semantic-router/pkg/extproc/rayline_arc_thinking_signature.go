package extproc

import (
	"encoding/json"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The thinking-signature guard (config router.thinking_signatures; on unless
// keep_unserved is set).
//
// Anthropic verifies the signature of every thinking block a request's
// history carries, and a signature verifies only on the Claude model, and the
// provider family, that wrote it. A conversation whose history holds thinking
// another model signed, or thinking signed before it reached the Router (a
// client's own Anthropic key), is refused with a 400 ("messages.N.content.M:
// Invalid `signature` in `thinking` block") on that turn and on every retry
// of it, because the client resends the same history.
//
// Two layers answer it.
//
// Before the send. A Claude worker reached over Messages gets, of the
// earlier assistant turns' thinking, only the thinking of the episode's
// thinking run on its own model: the stretch of the history the Router served
// on that model, turn after turn (reasoning.WithThinkingRun). The thinking of
// every other earlier assistant turn is stripped, whole per message. The
// Router's own minted markers never get this far: the request decoders strip
// a vsr.thinking.v1 signature and DropReasoningNotFromAnthropic drops the
// thinking left unsigned. A turn with no episode is left as it was.
//
// The latest assistant message of an active tool loop (the last message
// answers its tool_use with a tool_result) is kept even when the run does not
// cover it: with thinking on, Anthropic refuses a tool loop whose last
// assistant message does not start with a thinking block, so stripping it is
// a 400 of its own, and the block may still verify (the same model on the
// client's own key). The re-send below is what answers it if it does not.
//
// After a refusal. A call refused with exactly the invalid-signature 400 (or
// OpenRouter's "encrypted reasoning ... can only be processed by the provider
// that created it") is re-sent once, at once, on the same arm, with the
// thinking of every earlier assistant message stripped. The active tool
// loop's latest message keeps its thinking only when the run covers it; when
// its thinking is stripped, thinking is turned off for that one call, as
// Anthropic's own error advises, since the call would otherwise be refused
// for the missing block. Nothing has reached the client then: the refusal is
// decided before its headers are answered (thinkingSignatureRetryOnResponseHeaders),
// which needs the response body under FULL_DUPLEX_STREAMED, where the Router
// may read the body before it answers the headers. Under BUFFERED the body is
// sent only after the headers are answered, so the refusal takes the ordinary
// path there.

const (
	// thinkingRetryAttempt and thinkingRetryResend mark the two calls of a
	// re-sent turn on llm_usage (thinking_signature_retry).
	thinkingRetryAttempt = "signature_attempt"
	thinkingRetryResend  = "stripped_resend"
	// thinkingRetryBodyWait bounds the wait for a refusal's body before its
	// headers are answered, and thinkingRetryBodyLimit its size.
	thinkingRetryBodyWait  = 5 * time.Second
	thinkingRetryBodyLimit = 256 << 10
)

var (
	// invalidThinkingSignaturePattern is Anthropic's refusal of a thinking
	// block's signature, quoted with backticks or without.
	invalidThinkingSignaturePattern = regexp.MustCompile("(?i)invalid\\s+`?signature`?\\s+in\\s+`?thinking`?\\s+block")
	// foreignEncryptedReasoningPattern is OpenRouter's refusal of reasoning
	// another provider signed.
	foreignEncryptedReasoningPattern = regexp.MustCompile(`(?i)encrypted reasoning.*can only be processed by the provider that created it`)
)

// thinkingSignatureGuardOn reports whether the guard is on for a dispatch:
// the worker is Claude, and the operator did not turn the guard off.
func (r *OpenAIRouter) thinkingSignatureGuardOn(dispatch *providerDispatch) bool {
	return r != nil && r.Config != nil && !r.Config.KeepUnservedClaudeThinking && dispatch != nil &&
		r.armFamily(dispatch.logicalModel) == anthropicFamily
}

// thinkingRunWorker names the thinking run of the Claude model a backend
// serves: the provider endpoint the call goes to (its type and base URL) and
// the model id it is sent with. Arms that differ only in their thinking
// settings, or in the name of their backend entry, are one run. The provider
// OpenRouter picks does not enter: a signature another provider family cannot
// verify is the re-send's to answer.
func (r *OpenAIRouter) thinkingRunWorker(backendName, upstreamModel string) string {
	endpoint := backendName
	if profile, err := r.Config.GetProviderProfileForEndpoint(backendName); err == nil && profile != nil {
		endpoint = profile.Type + "\x00" + strings.TrimRight(profile.BaseURL, "/")
	}
	return reasoning.ThinkingRunWorker(endpoint + "\x00" + upstreamModel)
}

// otherThinkingRunWorkers are the run workers of every other Claude model the
// cell serves. A turn on one ends the runs of the others.
func (r *OpenAIRouter) otherThinkingRunWorkers(own string) []string {
	if r == nil || r.Config == nil {
		return nil
	}
	var others []string
	for name := range r.Config.ModelConfig {
		if r.armFamily(name) != anthropicFamily {
			continue
		}
		_, backend, found, err := r.Config.ResolvePrimaryBackendForModel(name)
		if err != nil || !found {
			continue
		}
		worker := r.thinkingRunWorker(backend, r.Config.ResolveExternalModelID(name, backend))
		if worker != own && !slices.Contains(others, worker) {
			others = append(others, worker)
		}
	}
	return others
}

// guardClaudeThinking applies the guard's first layer to a Claude dispatch
// and returns how many blocks it stripped. It runs after the reasoning carry
// rules, on every Claude dispatch with an episode, so the turn records its
// run whatever the wire; it strips only for a Messages target.
func (r *OpenAIRouter) guardClaudeThinking(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) int {
	if ctx == nil {
		return 0
	}
	ctx.thinkingRetryKeep = nil
	if request == nil || !r.thinkingSignatureGuardOn(dispatch) || ctx.RaylineARCTransaction == nil {
		return 0
	}
	transaction := ctx.RaylineARCTransaction
	worker := r.thinkingRunWorker(dispatch.backendName, dispatch.upstreamModel)
	transaction.thinkingWorker = worker
	transaction.thinkingOthers = r.otherThinkingRunWorkers(worker)
	var provenance []reasoning.ReasoningProvenance
	if transaction.state != nil {
		provenance = transaction.state.ReasoningProvenance
	}
	runStart := thinkingRunStart(request.Messages, reasoning.ThinkingRunAnchors(provenance, worker))
	transaction.thinkingAnchored = runStart >= 0
	active := activeToolLoopAssistant(request.Messages)
	transaction.thinkingAdopted = ""
	if active >= 0 && runStart >= 0 && active >= runStart {
		// The re-send keeps the loop's latest thinking only when the run
		// covers it.
		ctx.thinkingRetryKeep = protocolcodec.MessageThinkingKeys(request.Messages[active])
	} else if active >= 0 && dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
		// Sent though the run does not cover it: if Anthropic takes it, a new
		// run starts at it, so later turns keep it and the prefix this turn's
		// thinking is signed over does not change.
		if keys := protocolcodec.MessageThinkingKeys(request.Messages[active]); len(keys) > 0 {
			transaction.thinkingAdopted = keys[0]
		}
	}
	if dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 {
		return 0
	}
	stripped := protocolcodec.DropMessageThinking(request, func(index int, message llmprotocol.Message) bool {
		return message.Role == llmprotocol.RoleAssistant && index != active && (runStart < 0 || index < runStart)
	})
	if stripped > 0 {
		logging.ComponentEvent("extproc", "claude_thinking_unserved_stripped", map[string]interface{}{
			"request_id":   ctx.RequestID,
			"model":        dispatch.logicalModel,
			"stripped":     stripped,
			"run_anchored": runStart >= 0,
			"tool_loop":    active >= 0,
		})
	}
	return stripped
}

// thinkingRunStart is the index of the first assistant message holding one of
// the run's anchors, or -1 when the history holds none.
func thinkingRunStart(messages []llmprotocol.Message, anchors []string) int {
	if len(anchors) == 0 {
		return -1
	}
	anchored := make(map[string]bool, len(anchors))
	for _, anchor := range anchors {
		anchored[anchor] = true
	}
	for index, message := range messages {
		if message.Role != llmprotocol.RoleAssistant {
			continue
		}
		// A signed block's signature or a redacted block's data: a run may
		// start at either (an adopted redacted-only loop message).
		for _, key := range protocolcodec.MessageThinkingKeys(message) {
			if anchored[reasoning.ReasoningBlockDigest(key)] {
				return index
			}
		}
	}
	return -1
}

// activeToolLoopAssistant is the index of the assistant message whose tool
// call the last message answers with a tool_result, or -1 when the request
// does not end in an answered tool call.
func activeToolLoopAssistant(messages []llmprotocol.Message) int {
	if len(messages) < 2 {
		return -1
	}
	// A Messages client's tool_result blocks may decode to a user or a tool
	// message.
	results, _ := toolResultCallIDs(messages[len(messages)-1])
	if len(results) == 0 {
		return -1
	}
	for index := len(messages) - 2; index >= 0; index-- {
		if messages[index].Role != llmprotocol.RoleAssistant {
			continue
		}
		for _, content := range messages[index].Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				if _, answered := results[content.ToolCall.ID]; answered {
					return index
				}
			}
		}
		return -1
	}
	return -1
}

// thinkingSignatureRetryPlan is what the guard's re-send needs: the dispatch,
// the body it was sent, and the thinking of the active tool loop's latest
// message the run covers (a signed block's signature, a redacted block's
// data), the only thinking the re-send keeps.
type thinkingSignatureRetryPlan struct {
	dispatch   *providerDispatch
	dispatched []byte
	keep       map[string]bool
}

// stageThinkingSignatureRetry keeps what the re-send needs for a Claude
// dispatch over Messages. A turn a flex or endpoint re-send already answers
// keeps those: neither re-sends a 400.
func (r *OpenAIRouter) stageThinkingSignatureRetry(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.ThinkingSignatureRetry = nil
	if !r.thinkingSignatureGuardOn(dispatch) || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		ctx.LooperRequest || len(body) == 0 {
		return
	}
	keep := make(map[string]bool, len(ctx.thinkingRetryKeep))
	for _, key := range ctx.thinkingRetryKeep {
		keep[key] = true
	}
	ctx.ThinkingSignatureRetry = &thinkingSignatureRetryPlan{dispatch: dispatch, dispatched: body, keep: keep}
}

// thinkingSignatureRejected reports whether a provider's error body refuses a
// thinking block's signature: its message, or the provider body OpenRouter
// carries in metadata.raw, says so.
func thinkingSignatureRejected(body []byte) bool {
	var wire struct {
		Error struct {
			Message  string `json:"message"`
			Metadata struct {
				Raw json.RawMessage `json:"raw"`
			} `json:"metadata"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return false
	}
	texts := []string{wire.Error.Message}
	if raw := wire.Error.Metadata.Raw; len(raw) > 0 {
		var text string
		if json.Unmarshal(raw, &text) == nil {
			texts = append(texts, text)
		} else {
			texts = append(texts, string(raw))
		}
	}
	for _, text := range texts {
		if invalidThinkingSignaturePattern.MatchString(text) || foreignEncryptedReasoningPattern.MatchString(text) {
			return true
		}
	}
	return false
}

// refusedMessagePattern reads the message index a refusal names
// ("messages.19.content.12: ...").
var refusedMessagePattern = regexp.MustCompile(`messages\.(\d+)\.content\.\d+`)

// refusedActiveThinking reports whether a refusal may be of a block of the
// active tool loop's latest message in the dispatched body: it names that
// message, or names no message at all.
func refusedActiveThinking(dispatched, refusal []byte) bool {
	var members map[string]json.RawMessage
	var messages []map[string]json.RawMessage
	if json.Unmarshal(dispatched, &members) != nil || json.Unmarshal(members["messages"], &messages) != nil {
		return false
	}
	active := activeWireToolLoopAssistant(messages)
	if active < 0 {
		return false
	}
	matches := refusedMessagePattern.FindAllSubmatch(refusal, -1)
	if len(matches) == 0 {
		// A refusal that names no block (OpenRouter's encrypted-reasoning
		// one) may be of the loop's own: nothing proves it is not.
		return true
	}
	for _, match := range matches {
		if index, err := strconv.Atoi(string(match[1])); err == nil && index == active {
			return true
		}
	}
	return false
}

// thinkingRetryBody is the dispatched Messages body with the thinking of
// every earlier assistant message stripped. The active tool loop's latest
// message keeps the blocks keep names; when it loses any, it loses all, and
// thinking is turned off for the call (with any clear_thinking context edit,
// which requires it). It reports what it changed, and false when nothing.
type thinkingRetryChange struct {
	stripped         int
	thinkingDisabled bool
	// keptSignature is the first key (signature, or redacted data) of the
	// active loop's thinking when the re-send keeps it.
	keptSignature string
}

func thinkingRetryBody(body []byte, keep map[string]bool) ([]byte, thinkingRetryChange, bool) {
	var change thinkingRetryChange
	var members map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil {
		return nil, change, false
	}
	var messages []map[string]json.RawMessage
	if json.Unmarshal(members["messages"], &messages) != nil {
		return nil, change, false
	}
	active := activeWireToolLoopAssistant(messages)
	activeStripped := false
	kept := make([]map[string]json.RawMessage, 0, len(messages))
	for index, message := range messages {
		if wireRole(message) != "assistant" {
			kept = append(kept, message)
			continue
		}
		var blocks []map[string]json.RawMessage
		if json.Unmarshal(message["content"], &blocks) != nil {
			kept = append(kept, message)
			continue
		}
		holds, whole := 0, true
		for _, block := range blocks {
			if key, thinking := wireThinkingKey(block); thinking {
				holds++
				whole = whole && keep[key]
			}
		}
		if holds == 0 || index == active && whole {
			if index == active && holds > 0 {
				for _, block := range blocks {
					if key, thinking := wireThinkingKey(block); thinking && key != "" {
						change.keptSignature = key
						break
					}
				}
			}
			kept = append(kept, message)
			continue
		}
		if index == active {
			activeStripped = true
		}
		remaining := make([]map[string]json.RawMessage, 0, len(blocks)-holds)
		for _, block := range blocks {
			if _, thinking := wireThinkingKey(block); !thinking {
				remaining = append(remaining, block)
			}
		}
		change.stripped += holds
		if len(remaining) == 0 {
			continue
		}
		content, err := json.Marshal(remaining)
		if err != nil {
			return nil, change, false
		}
		message["content"] = content
		kept = append(kept, message)
	}
	if change.stripped == 0 {
		return nil, change, false
	}
	encoded, err := json.Marshal(kept)
	if err != nil {
		return nil, change, false
	}
	members["messages"] = encoded
	if activeStripped && thinkingEnabledOnWire(members["thinking"]) {
		members["thinking"] = json.RawMessage(`{"type":"disabled"}`)
		withoutClearThinkingEdits(members)
		withoutThinkingOnlyEffort(members)
		change.thinkingDisabled = true
	}
	out, err := json.Marshal(members)
	if err != nil {
		return nil, change, false
	}
	return out, change, true
}

func wireRole(message map[string]json.RawMessage) string {
	var role string
	_ = json.Unmarshal(message["role"], &role)
	return role
}

func wireBlockType(block map[string]json.RawMessage) string {
	var kind string
	_ = json.Unmarshal(block["type"], &kind)
	return kind
}

// wireThinkingKey identifies a thinking block on the wire: a thinking
// block's signature, a redacted_thinking block's data.
func wireThinkingKey(block map[string]json.RawMessage) (string, bool) {
	var key string
	switch wireBlockType(block) {
	case "thinking":
		_ = json.Unmarshal(block["signature"], &key)
	case "redacted_thinking":
		_ = json.Unmarshal(block["data"], &key)
	default:
		return "", false
	}
	return key, true
}

// activeWireToolLoopAssistant is activeToolLoopAssistant on a Messages body.
func activeWireToolLoopAssistant(messages []map[string]json.RawMessage) int {
	if len(messages) < 2 || wireRole(messages[len(messages)-1]) != "user" {
		return -1
	}
	answered := map[string]bool{}
	var last []map[string]json.RawMessage
	_ = json.Unmarshal(messages[len(messages)-1]["content"], &last)
	for _, block := range last {
		if wireBlockType(block) != "tool_result" {
			continue
		}
		var id string
		if json.Unmarshal(block["tool_use_id"], &id) == nil && id != "" {
			answered[id] = true
		}
	}
	if len(answered) == 0 {
		return -1
	}
	for index := len(messages) - 2; index >= 0; index-- {
		if wireRole(messages[index]) != "assistant" {
			continue
		}
		var blocks []map[string]json.RawMessage
		_ = json.Unmarshal(messages[index]["content"], &blocks)
		for _, block := range blocks {
			var id string
			if wireBlockType(block) == "tool_use" && json.Unmarshal(block["id"], &id) == nil && answered[id] {
				return index
			}
		}
		return -1
	}
	return -1
}

func thinkingEnabledOnWire(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var thinking struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &thinking) == nil && thinking.Type != "" && thinking.Type != "disabled"
}

// thinkingOnlyEfforts are the output_config.effort levels Anthropic accepts
// only with thinking on.
var thinkingOnlyEfforts = map[string]bool{"xhigh": true, "max": true}

// withoutThinkingOnlyEffort drops an effort level that requires thinking,
// and output_config when nothing is left in it.
func withoutThinkingOnlyEffort(members map[string]json.RawMessage) {
	var output map[string]json.RawMessage
	if json.Unmarshal(members["output_config"], &output) != nil {
		return
	}
	var effort string
	if json.Unmarshal(output["effort"], &effort) != nil || !thinkingOnlyEfforts[effort] {
		return
	}
	delete(output, "effort")
	if len(output) == 0 {
		delete(members, "output_config")
		return
	}
	if encoded, err := json.Marshal(output); err == nil {
		members["output_config"] = encoded
	}
}

// withoutClearThinkingEdits drops the clear_thinking context edits, which
// Anthropic accepts only with thinking on, and the context_management member
// when no edit is left.
func withoutClearThinkingEdits(members map[string]json.RawMessage) {
	var management map[string]json.RawMessage
	if json.Unmarshal(members["context_management"], &management) != nil {
		return
	}
	var edits []map[string]json.RawMessage
	if json.Unmarshal(management["edits"], &edits) != nil {
		return
	}
	kept := edits[:0]
	for _, edit := range edits {
		if !strings.HasPrefix(wireBlockType(edit), "clear_thinking") {
			kept = append(kept, edit)
		}
	}
	if len(kept) == len(edits) {
		return
	}
	if len(kept) == 0 {
		delete(management, "edits")
	} else if encoded, err := json.Marshal(kept); err == nil {
		management["edits"] = encoded
	}
	if len(management) == 0 {
		delete(members, "context_management")
		return
	}
	if encoded, err := json.Marshal(management); err == nil {
		members["context_management"] = encoded
	}
}

// thinkingSignatureRetryOnResponseHeaders answers a staged Claude call's 400
// before its headers are answered: it reads the refusal's body off the
// stream, and when the refusal is the invalid-signature class it re-sends the
// call once with the earlier turns' thinking stripped and returns the
// re-send's reply. Otherwise it puts every message it read back, in order,
// and returns nil, so the refusal takes the ordinary path untouched.
func (r *OpenAIRouter) thinkingSignatureRetryOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if v == nil || v.ResponseHeaders == nil || ctx == nil || ctx.ThinkingSignatureRetry == nil {
		return nil
	}
	plan := ctx.ThinkingSignatureRetry
	status := getStatusFromHeaders(v.ResponseHeaders.GetHeaders())
	if status != http.StatusBadRequest || !ctx.FullDuplexResponseBody || ctx.streamReceiver == nil ||
		v.ResponseHeaders.GetEndOfStream() {
		return nil
	}
	// One turn re-sends once.
	ctx.ThinkingSignatureRetry = nil
	received := time.Now()
	held, body, complete := ctx.streamReceiver.holdBody(thinkingRetryBodyWait, thinkingRetryBodyLimit)
	giveBack := func(reason string) *ext_proc.ProcessingResponse {
		ctx.streamReceiver.pushBack(held)
		if reason != "" {
			logging.ComponentWarnEvent("extproc", "thinking_signature_retry_skipped", map[string]interface{}{
				"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel, "reason": reason,
			})
		}
		return nil
	}
	if !complete || !thinkingSignatureRejected(body) {
		return giveBack("")
	}
	keep := plan.keep
	if refusedActiveThinking(plan.dispatched, body) {
		// The refusal names the loop's own block: the run's record of it
		// does not make it verifiable.
		keep = nil
	}
	retryBody, change, changed := thinkingRetryBody(plan.dispatched, keep)
	if !changed && len(keep) > 0 {
		retryBody, change, changed = thinkingRetryBody(plan.dispatched, nil)
	}
	if !changed {
		return giveBack("nothing_to_strip")
	}
	budget, skipped := r.flexResendBudget()
	if skipped != "" {
		return giveBack(skipped)
	}
	call, unavailable := r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, retryBody)
	if call == nil {
		return giveBack(unavailable)
	}
	// The refused call: its headers and its error are all it said.
	outcome := evaluateResponseHeaderOutcome(v, ctx)
	ctx.UpstreamStatusCode = status
	captureRaylineARCProviderAttempts(v.ResponseHeaders.GetHeaders(), ctx, status)
	finishUpstreamResponseSpan(ctx, outcome)
	ctx.UpstreamErrorText = readUpstreamErrorText(status, body, nil)
	r.settleRefusedAttempt(ctx, upstreamFailureClass(status, nil), status, func(record *llmUsageRecord) {
		record.ThinkingSignatureRetry = nonEmpty(thinkingRetryAttempt)
	})
	refusedText := ctx.UpstreamErrorText
	ctx.UpstreamErrorText = upstreamErrorText{}

	ctx.ThinkingSignatureRetried = true
	ctx.DispatchedOutbound = readDispatchedOutboundParams(retryBody, plan.dispatch)
	// The re-send is the body the episode records: its prefix, and a run
	// that starts again from this turn.
	r.auditRaylineARCUpstream(retryBody, ctx)
	if transaction := ctx.RaylineARCTransaction; transaction != nil && transaction.thinkingWorker != "" {
		// A kept loop message starts the new run, as an adopted one does.
		transaction.thinkingRestart, transaction.thinkingAdopted = true, change.keptSignature
	}
	result := callFlexTwin(ctx, call, r.flexResendBodyLimit(), received.Add(budget))
	fields := map[string]interface{}{
		"request_id":        ctx.RequestID,
		"model":             plan.dispatch.logicalModel,
		"refused_status":    status,
		"resend_status":     result.status,
		"stripped":          change.stripped,
		"thinking_disabled": change.thinkingDisabled,
		"resend_latency_ms": time.Since(received).Milliseconds(),
	}
	addUpstreamErrorTextFields(fields, refusedText)
	if result.err != nil {
		fields["resend_error"] = result.err.Error()
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "thinking_signature_retry", fields)
	return r.serveResent(ctx, plan.dispatch.logicalModel, result)
}
