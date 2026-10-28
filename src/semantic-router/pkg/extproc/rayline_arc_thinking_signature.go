package extproc

import (
	"encoding/json"
	"net/http"
	"regexp"
	"slices"
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
// The same holds for every provider's encrypted reasoning: OpenRouter
// refuses a request carrying Anthropic's to OpenAI ("encrypted reasoning or
// compaction content that can only be processed by the provider that created
// it"), after a policy switch from Claude to gpt, and the reverse.
//
// Two layers answer it.
//
// Before the send. Every worker gets, of the earlier assistant turns'
// reasoning, only what the episode's run on its own model covers: the
// stretch of the history the Router served on that model, turn after turn
// (reasoning.WithThinkingRun), found by an anchor key
// (protocolcodec.MessageAnchorKeys). Outside it, a Claude worker loses the
// reasoning whole and any other worker loses what only its issuer can read
// (guardHistoryReasoning). The Router's own minted markers never get this
// far: the request decoders strip a vsr.thinking.v1 signature and
// DropReasoningNotFromAnthropic drops the thinking left unsigned. A turn with
// no episode is left as it was.
//
// The latest assistant message of an active tool loop (the last message
// answers its tool call) follows the same rule, but a Claude worker is sent
// it even when the run does not cover it: with thinking on, Anthropic
// refuses a tool loop whose last assistant message does not start with a
// thinking block, so stripping it is a 400 of its own, and the block may
// still verify (the same model on the client's own key). The re-send below
// is what answers it if it does not.
//
// After a refusal. A call on any worker refused with exactly the
// invalid-signature 400 or OpenRouter's encrypted-reasoning one is re-sent
// once, at once, on the same arm, with the signed and encrypted reasoning of
// every assistant message stripped, the active tool loop's latest reply's
// included (Messages: thinking blocks; Chat: reasoning_details; Responses:
// encrypted reasoning items). Anthropic binds a signature to the history
// before it, so no block survives the removal of any before it; the run
// starts again at the re-send's reply. Where the loop's reply lost its
// thinking and the model can turn thinking off, the re-send turns it off, as
// Anthropic's own error advises, since the call would otherwise be refused
// for the missing block; an always-on model keeps its settings. Nothing has
// reached the client then: the refusal is
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
// the operator did not turn it off.
func (r *OpenAIRouter) thinkingSignatureGuardOn(dispatch *providerDispatch) bool {
	return r != nil && r.Config != nil && !r.Config.KeepUnservedClaudeThinking && dispatch != nil
}

// thinkingRunWorker names the thinking run of the model a backend serves: the provider endpoint the call goes to (its type and base URL) and
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

// otherThinkingRunWorkers are the run workers of every other model the cell
// serves. A turn on one ends the runs of the others.
func (r *OpenAIRouter) otherThinkingRunWorkers(own string) []string {
	if r == nil || r.Config == nil {
		return nil
	}
	var others []string
	for name := range r.Config.ModelConfig {
		_, backend, found, err := r.Config.ResolvePrimaryBackendForModel(name)
		if err != nil || !found {
			continue
		}
		worker := r.thinkingRunWorker(backend, r.Config.ResolveExternalModelID(name, backend))
		if worker != own && !slices.Contains(others, worker) {
			others = append(others, worker)
		}
	}
	slices.Sort(others)
	return others
}

// guardHistoryReasoning applies the guard's first layer to a dispatch and
// returns how many reasoning contents it stripped or changed. It runs after
// the reasoning carry rules, on every dispatch with an episode, whatever the
// worker and the wire.
//
// Earlier assistant messages the episode's run on this model does not cover
// lose what only another issuer can read. A Claude worker loses their
// reasoning whole: Claude verifies every thinking signature, and its
// reasoning-details replay is its own run's. Any other worker loses their
// redacted_thinking, thinking signatures and reasoning_details, and keeps
// their reasoning text, which the carry rules hand on as reasoning_content
// (protocolcodec.DropForeignOpaqueReasoning); over Messages, which takes
// thinking only signed, it loses their reasoning whole.
//
// The active tool loop's latest message follows the same rule, except on a
// Claude worker: with thinking on, Anthropic refuses a loop whose last
// assistant message does not start with a thinking block, so Claude is sent
// it as the client sent it, and a run starts at it when Claude takes it.
func (r *OpenAIRouter) guardHistoryReasoning(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) int {
	if ctx == nil {
		return 0
	}
	if request == nil || !r.thinkingSignatureGuardOn(dispatch) || ctx.RaylineARCTransaction == nil {
		return 0
	}
	transaction := ctx.RaylineARCTransaction
	claude := r.armFamily(dispatch.logicalModel) == anthropicFamily
	worker := r.thinkingRunWorker(dispatch.backendName, dispatch.upstreamModel)
	transaction.thinkingWorker = worker
	transaction.thinkingOthers = r.otherThinkingRunWorkers(worker)
	var provenance []reasoning.ReasoningProvenance
	if transaction.state != nil {
		provenance = transaction.state.ReasoningProvenance
	}
	runStart := thinkingRunStart(request.Messages, reasoning.ThinkingRunAnchors(provenance, worker))
	transaction.thinkingAnchored = runStart >= 0
	transaction.thinkingReplyIndex = len(request.Messages)
	covered := func(index int) bool { return runStart >= 0 && index >= runStart }
	// The active loop's latest reply: the assistant message holding the
	// answered call and the assistant messages right before it, which a
	// Responses client sends as items of the same reply (its reasoning, then
	// its calls).
	activeStart, active := activeToolLoopReply(request.Messages)
	inActive := func(index int) bool { return active >= 0 && index >= activeStart && index <= active }
	transaction.thinkingAdopted = ""
	keepActive := false
	switch {
	case active >= 0 && covered(activeStart):
		keepActive = true
	case active >= 0 && claude:
		// Sent though the run does not cover it: if Claude takes it, a new
		// run starts at it, so later turns keep it and the prefix this turn's
		// thinking is signed over does not change.
		keepActive = true
		reply := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
		for index := activeStart; index <= active; index++ {
			reply.Content = append(reply.Content, request.Messages[index].Content...)
		}
		if keys := protocolcodec.MessageAnchorKeys(reply, activeStart); len(keys) > 0 {
			transaction.thinkingAdopted = keys[0]
		}
	}
	drop := func(index int, message llmprotocol.Message) bool {
		return message.Role == llmprotocol.RoleAssistant && !covered(index) && (!inActive(index) || !keepActive)
	}
	var stripped int
	if claude {
		stripped = protocolcodec.DropMessageThinking(request, drop)
	} else {
		stripped = protocolcodec.DropForeignOpaqueReasoning(request,
			dispatch.targetFormat == llmprotocol.AnthropicMessagesV1, drop)
	}
	if stripped > 0 {
		logging.ComponentEvent("extproc", "unserved_reasoning_stripped", map[string]interface{}{
			"request_id":   ctx.RequestID,
			"model":        dispatch.logicalModel,
			"wire_format":  dispatch.targetFormat,
			"stripped":     stripped,
			"run_anchored": runStart >= 0,
			"tool_loop":    active >= 0,
		})
	}
	return stripped
}

// thinkingRunStart is the index of the first message of the reply holding
// one of the run's anchors, or -1 when the history holds none. A reply is a
// run of consecutive assistant messages: a Responses client resends one
// reply as several items (its reasoning, then its message and calls), so a
// reply is read, and its text anchored, as one message at its first index,
// as ResponseAnchorKey records it.
func thinkingRunStart(messages []llmprotocol.Message, anchors []string) int {
	if len(anchors) == 0 {
		return -1
	}
	anchored := make(map[string]bool, len(anchors))
	for _, anchor := range anchors {
		anchored[anchor] = true
	}
	for start := 0; start < len(messages); start++ {
		if messages[start].Role != llmprotocol.RoleAssistant {
			continue
		}
		reply := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
		end := start
		for ; end < len(messages) && messages[end].Role == llmprotocol.RoleAssistant; end++ {
			reply.Content = append(reply.Content, messages[end].Content...)
		}
		for _, key := range protocolcodec.MessageAnchorKeys(reply, start) {
			if anchored[reasoning.ThinkingRunAnchorDigest(key)] {
				return start
			}
		}
		start = end - 1
	}
	return -1
}

// activeToolLoopReply is the active tool loop's latest reply as indexes of
// its first and last message, or -1, -1. The reply is the run of consecutive
// assistant messages before the history's trailing results: a Responses
// client resends one reply as separate items (its reasoning, then each call),
// and a Messages client's results may decode, with any text beside them, to
// several tool and user messages. The loop is active when those results
// answer any of the reply's calls, in whatever order.
func activeToolLoopReply(messages []llmprotocol.Message) (int, int) {
	last := len(messages) - 1
	for last >= 0 && messages[last].Role != llmprotocol.RoleAssistant {
		last--
	}
	if last < 0 || last == len(messages)-1 {
		return -1, -1
	}
	answered := map[string]struct{}{}
	for _, message := range messages[last+1:] {
		results, _ := toolResultCallIDs(message)
		for id := range results {
			answered[id] = struct{}{}
		}
	}
	if len(answered) == 0 {
		return -1, -1
	}
	start := last
	for start > 0 && messages[start-1].Role == llmprotocol.RoleAssistant {
		start--
	}
	for _, message := range messages[start : last+1] {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				if _, ok := answered[content.ToolCall.ID]; ok {
					return start, last
				}
			}
		}
	}
	return -1, -1
}

// thinkingSignatureRetryPlan is what the guard's re-send needs: the dispatch,
// the body it was sent, whether the worker is Claude, and whether its model
// can turn reasoning off (config.RouterConfig.ModelCanDisableReasoning).
type thinkingSignatureRetryPlan struct {
	dispatch   *providerDispatch
	dispatched []byte
	claude     bool
	canDisable bool
}

// stageThinkingSignatureRetry keeps what the re-send needs for a dispatch. A
// turn a flex or endpoint re-send already answers keeps those: neither
// re-sends a 400.
func (r *OpenAIRouter) stageThinkingSignatureRetry(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.ThinkingSignatureRetry = nil
	if !r.thinkingSignatureGuardOn(dispatch) || ctx.LooperRequest || len(body) == 0 {
		return
	}
	ctx.ThinkingSignatureRetry = &thinkingSignatureRetryPlan{
		dispatch: dispatch, dispatched: body,
		claude:     r.armFamily(dispatch.logicalModel) == anthropicFamily,
		canDisable: r.Config.ModelCanDisableReasoning(dispatch.logicalModel),
	}
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

// retryBody is the plan's dispatched body with the reasoning a refusal may
// be of stripped, in the dispatch's wire format.
func (plan *thinkingSignatureRetryPlan) retryBody(refusal []byte) ([]byte, thinkingRetryChange, bool) {
	switch plan.dispatch.targetFormat {
	case llmprotocol.AnthropicMessagesV1:
		return thinkingRetryBody(plan.dispatched, plan.canDisable)
	case llmprotocol.OpenAIChatV1:
		// A Chat refusal's message index (Anthropic's, behind OpenRouter)
		// does not name a Chat message, so the loop's own reasoning_details
		// cannot be shown not to be the refused ones: they go too.
		return chatRetryBody(plan.dispatched, plan.claude && plan.canDisable)
	case llmprotocol.OpenAIResponsesV1:
		return responsesRetryBody(plan.dispatched)
	}
	return nil, thinkingRetryChange{}, false
}

// chatRetryBody is a dispatched Chat body with every assistant message's
// reasoning_details removed (the encrypted and signed reasoning OpenRouter
// carries). The reasoning text is kept. For a Claude worker whose model can
// turn reasoning off (disableReasoning), a re-send whose active loop reply
// lost its signed details asks for no reasoning, as the Messages re-send turns
// thinking off: Claude refuses a loop whose last reply lacks its thinking.
func chatRetryBody(body []byte, disableReasoning bool) ([]byte, thinkingRetryChange, bool) {
	var change thinkingRetryChange
	var members map[string]json.RawMessage
	var messages []map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil || json.Unmarshal(members["messages"], &messages) != nil {
		return nil, change, false
	}
	active := activeChatToolLoopReplyEnd(messages)
	activeStripped := false
	for index, message := range messages {
		if wireRole(message) != "assistant" {
			continue
		}
		details, held := message["reasoning_details"]
		if !held {
			continue
		}
		if !hasJSONMember(message, "reasoning_content") && !hasJSONMember(message, "reasoning") {
			// The details were the reasoning's only text: its plaintext stays.
			if text := protocolcodec.ReasoningDetailsPlaintext(details); text != "" {
				encoded, err := json.Marshal(text)
				if err != nil {
					return nil, change, false
				}
				message["reasoning_content"] = encoded
			}
		}
		delete(message, "reasoning_details")
		change.stripped++
		activeStripped = activeStripped || index == active
	}
	if change.stripped == 0 {
		return nil, change, false
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return nil, change, false
	}
	members["messages"] = encoded
	if activeStripped && disableReasoning {
		for _, member := range []string{"reasoning", "reasoning_effort"} {
			if _, set := members[member]; set {
				delete(members, member)
				change.thinkingDisabled = true
			}
		}
	}
	out, err := json.Marshal(members)
	return out, change, err == nil
}

// activeChatToolLoopReplyEnd is the index of the assistant message whose
// tool calls a Chat body's trailing tool messages answer, or -1. A Chat
// reply is one message.
func activeChatToolLoopReplyEnd(messages []map[string]json.RawMessage) int {
	last := len(messages) - 1
	answered := map[string]bool{}
	for ; last >= 0 && wireRole(messages[last]) != "assistant"; last-- {
		var id string
		if json.Unmarshal(messages[last]["tool_call_id"], &id) == nil && id != "" {
			answered[id] = true
		}
	}
	if last < 0 || len(answered) == 0 {
		return -1
	}
	var calls []struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(messages[last]["tool_calls"], &calls)
	for _, call := range calls {
		if answered[call.ID] {
			return last
		}
	}
	return -1
}

// responsesRetryBody is a dispatched Responses body without its encrypted
// reasoning items.
func responsesRetryBody(body []byte) ([]byte, thinkingRetryChange, bool) {
	var change thinkingRetryChange
	var members map[string]json.RawMessage
	var input []map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil || json.Unmarshal(members["input"], &input) != nil {
		return nil, change, false
	}
	kept := make([]map[string]json.RawMessage, 0, len(input))
	for _, item := range input {
		if wireBlockType(item) == "reasoning" && hasJSONMember(item, "encrypted_content") {
			change.stripped++
			continue
		}
		kept = append(kept, item)
	}
	if change.stripped == 0 {
		return nil, change, false
	}
	encoded, err := json.Marshal(kept)
	if err != nil {
		return nil, change, false
	}
	members["input"] = encoded
	out, err := json.Marshal(members)
	return out, change, err == nil
}

func hasJSONMember(members map[string]json.RawMessage, name string) bool {
	value, present := members[name]
	return present && len(value) > 0 && string(value) != "null" && string(value) != `""`
}

// thinkingRetryBody is the dispatched Messages body with the thinking of
// every assistant message stripped, the active tool loop's latest reply's
// included. A kept block's signature covers the history before it (Anthropic
// binds it to its prefix), so no block survives the removal of any before
// it, and the run starts again at the re-send's reply. When the loop's reply
// loses its thinking, thinking is turned off for the call where the model
// can turn it off (with the clear_thinking edit and an effort that require
// it): otherwise Anthropic refuses the loop for its missing thinking block.
// An always-on model keeps its thinking settings, and its re-send may be
// refused for that. It reports what it changed, and false when nothing.
type thinkingRetryChange struct {
	stripped         int
	thinkingDisabled bool
}

func thinkingRetryBody(body []byte, canDisable bool) ([]byte, thinkingRetryChange, bool) {
	var change thinkingRetryChange
	var members map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil {
		return nil, change, false
	}
	var messages []map[string]json.RawMessage
	if json.Unmarshal(members["messages"], &messages) != nil {
		return nil, change, false
	}
	start, end := activeWireToolLoopReply(messages)
	activeStripped := false
	kept := make([]map[string]json.RawMessage, 0, len(messages))
	for index, message := range messages {
		var blocks []map[string]json.RawMessage
		if wireRole(message) != "assistant" || json.Unmarshal(message["content"], &blocks) != nil {
			kept = append(kept, message)
			continue
		}
		remaining := make([]map[string]json.RawMessage, 0, len(blocks))
		for _, block := range blocks {
			if _, thinking := wireThinkingKey(block); !thinking {
				remaining = append(remaining, block)
			}
		}
		holds := len(blocks) - len(remaining)
		if holds == 0 {
			kept = append(kept, message)
			continue
		}
		change.stripped += holds
		if end >= 0 && index >= start && index <= end {
			activeStripped = true
		}
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
	if activeStripped && canDisable && thinkingEnabledOnWire(members["thinking"]) {
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

// activeWireToolLoopReply is activeToolLoopReply on a Messages body.
func activeWireToolLoopReply(messages []map[string]json.RawMessage) (int, int) {
	last := len(messages) - 1
	for last >= 0 && wireRole(messages[last]) != "assistant" {
		last--
	}
	if last < 0 || last == len(messages)-1 {
		return -1, -1
	}
	answered := map[string]bool{}
	for _, message := range messages[last+1:] {
		var blocks []map[string]json.RawMessage
		_ = json.Unmarshal(message["content"], &blocks)
		for _, block := range blocks {
			var id string
			if wireBlockType(block) == "tool_result" && json.Unmarshal(block["tool_use_id"], &id) == nil && id != "" {
				answered[id] = true
			}
		}
	}
	if len(answered) == 0 {
		return -1, -1
	}
	start := last
	for start > 0 && wireRole(messages[start-1]) == "assistant" {
		start--
	}
	for _, message := range messages[start : last+1] {
		var blocks []map[string]json.RawMessage
		_ = json.Unmarshal(message["content"], &blocks)
		for _, block := range blocks {
			var id string
			if wireBlockType(block) == "tool_use" && json.Unmarshal(block["id"], &id) == nil && answered[id] {
				return start, last
			}
		}
	}
	return -1, -1
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
	retryBody, change, changed := plan.retryBody(body)
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
	// The re-send is served whole, outside full duplex: no host retry can
	// hold its frames or take it over, so the staged plan is dropped.
	ctx.HostRetry = nil

	ctx.ThinkingSignatureRetried = true
	ctx.DispatchedOutbound = readDispatchedOutboundParams(retryBody, plan.dispatch)
	// The re-send is the body the episode records: its prefix, and a run
	// that starts again from this turn.
	r.auditRaylineARCUpstream(retryBody, ctx)
	if transaction := ctx.RaylineARCTransaction; transaction != nil && transaction.thinkingWorker != "" {
		// The re-send kept no reasoning: the run starts again at its reply.
		transaction.thinkingRestart, transaction.thinkingAdopted = true, ""
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
