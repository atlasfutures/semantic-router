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

// Held turns send only their new messages (pathfinder ADR 0155, proposed;
// arc_serving_contract.md, "Staged turns"). On a delta cell
// (policy_service.decide_mode: delta):
//
//  1. Every strict main turn's decide asks the service to stage the turn.
//  2. When the turn commits -- the client has the whole 2xx reply -- and
//     before its episode lease is released, VSR appends the reply's
//     assistant output to the staged turn. The reply carries the staged
//     prefix digest and count; any hold (fallback lease) it carries is not
//     read.
//  3. The next turn, if it extends that staged prefix, sends only its new
//     messages (turn_delta) under its own shell, and is decided
//     synchronously like any other turn: the service decides it on exactly
//     the conversation a full request would carry, so no decision changes.
//
// Nothing here fails a turn. A delta the service cannot extend is sent in
// full; a failed append leaves the next turn to be sent in full.
//
// The fence is the service's, as for every decide: its per-episode busy
// lock refuses a call that overlaps one VSR stopped waiting for. The append
// is bounded inside the lease (policyAppendTimeout) for the next turn's
// sake.
//
// What a turn carries to the next is an allow-list: the staged entry holds
// only the service's digest and count, VSR's own digest of the same
// messages and the committed turn it belongs to. Every commit replaces it or
// drops it, so nothing outlives the turn that wrote it.

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
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Decision modes a delta cell records (selection.RaylineARCTrace.PolicyDecisionMode).
const (
	policyDecisionModeSync  = "sync"
	policyDecisionModeDelta = "delta"
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
	// the same messages (policyDeltaDigest), which the next request must
	// reproduce over its first count messages.
	count        int
	prefixDigest string
	digest       string
	// appendedFrom is where the appended reply starts, and historyDigest
	// VSR's digest of the messages before it: a mismatch is then told apart
	// as the history's or the appended reply's.
	appendedFrom  int
	historyDigest string
	// unsettled is an append VSR stopped waiting for (a timeout or a
	// transport failure): the service may still hold the episode busy, so the
	// next turn sends in full and retries session_busy one acquire timeout
	// longer (policyFirstBusyWait).
	unsettled bool
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

// policyStaging is what a delta cell's strict main turn hands its episode
// transaction: what the append posted at its commit needs.
type policyStaging struct {
	turns         *policyStagedTurns
	appender      policyAppender
	admit         func() (func(), error)
	pkg           raylinearc.PolicyPackageRef
	episodeIDHash string
	requestID     string
	requestFormat string
	contextEpoch  string
	// messages and roles are the conversation the turn was decided on.
	messages []json.RawMessage
	roles    []string
	// clientBody is the reply the client received, once it is complete.
	clientBody []byte
	// appendTimeout bounds the append.
	appendTimeout time.Duration
}

// policyDeltaEligible reports whether this cell's turn may stage at all: a
// delta cell's strict main turn, without exploration, in a format the
// append can carry, on a service that advertises staged turns.
func (scorer *policyServiceScorer) policyDeltaEligible(request *raylinearc.PolicyDecisionRequest, sideCall bool) bool {
	return scorer.delta && scorer.stagedTurnsServed.Load() && !sideCall && request.EpisodeMode == "" &&
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
	staged, ok := scorer.staged.get(episodeIDHash)
	switch {
	case ok && staged.turnIndex == state.TurnIndex && staged.unsettled:
		// Ahead of every other reason: it widens the turn's busy wait.
		return policyStagedTurn{}, "append_unsettled"
	case scorer.deltaUnsupported.Load():
		return policyStagedTurn{}, "delta_unsupported"
	case !ok:
		return policyStagedTurn{}, "not_staged"
	case staged.turnIndex != state.TurnIndex:
		return policyStagedTurn{}, "stale_turn"
	case transition != raylinearc.PolicyTransitionExtends || turn == nil || staged.contextEpoch != turn.ContextEpoch():
		return policyStagedTurn{}, "new_context"
	case staged.requestFormat != format:
		return policyStagedTurn{}, "format"
	case len(messages) <= staged.count:
		return policyStagedTurn{}, "no_new_messages"
	case policyDeltaDigest(messages, staged.count) != staged.digest:
		if policyDeltaDigest(messages, staged.appendedFrom) != staged.historyDigest {
			return policyStagedTurn{}, "history_mismatch"
		}
		return policyStagedTurn{}, "appended_mismatch"
	}
	return staged, "delta"
}

// policyDeltaDigest digests the first n messages as the delta check
// compares them: what the service's canonical conversation projection
// (canonical_v1 and canonical_v2 packages, the only ones it serves a delta
// for) reads the same. Insignificant are:
//   - a cache_control marker on a message or on one of its top-level content
//     blocks (never one nested deeper, such as a tool's input);
//   - the members the projection removes from a top-level content block
//     whatever their value (pathfinder canonical_conversation.py _clean): a
//     tool_use or server_tool_use block's caller and a thinking block's
//     signature; and a text block's citations when null or [] (non-empty
//     citations are encoded, so they still count);
//   - a system or user message whose content is one text block, which is its
//     text (Claude Code demotes its trailing system message so, Claude
//     Desktop its newest user prompt; the projection reads a string as that
//     one block).
//
// Object key order and whitespace are insignificant. Anything else that
// differs, thinking text and redacted_thinking data included, makes the turn
// go in full.
func policyDeltaDigest(messages []json.RawMessage, n int) string {
	hash := sha256.New()
	for _, message := range messages[:n] {
		canonical := policyDeltaCanonical(message)
		hash.Write([]byte(strconv.Itoa(len(canonical))))
		hash.Write([]byte{':'})
		hash.Write(canonical)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func policyDeltaCanonical(message json.RawMessage) []byte {
	var object map[string]any
	decoder := json.NewDecoder(bytes.NewReader(message))
	decoder.UseNumber()
	if decoder.Decode(&object) != nil || object == nil || decoder.More() {
		return message
	}
	delete(object, "cache_control")
	if blocks, ok := object["content"].([]any); ok {
		for _, block := range blocks {
			if fields, ok := block.(map[string]any); ok {
				delete(fields, "cache_control")
				policyDeltaCleanBlock(fields)
			}
		}
		if role := object["role"]; (role == "system" || role == "user") && len(blocks) == 1 {
			if fields, ok := blocks[0].(map[string]any); ok && len(fields) == 2 && fields["type"] == "text" {
				// canonical-conversation.v1 reads "" as no block and an empty
				// text block as one: those are not demoted, in either role.
				if text, ok := fields["text"].(string); ok && text != "" {
					object["content"] = text
				}
			}
		}
	}
	canonical, err := raylinearc.MarshalUnescaped(object)
	if err != nil {
		return message
	}
	return canonical
}

// policyDeltaCleanBlock removes from one top-level content block exactly
// the members the service's canonical projection removes from it (beside
// cache_control), so a client that echoes them otherwise -- Workshop's agent
// sends no citations or caller and an unsigned thinking block's signature as
// "" -- extends the staged turn as the full request encodes it.
func policyDeltaCleanBlock(fields map[string]any) {
	switch fields["type"] {
	case "text":
		if citations, ok := fields["citations"]; ok {
			if list, isList := citations.([]any); citations == nil || (isList && len(list) == 0) {
				delete(fields, "citations")
			}
		}
	case "tool_use", "server_tool_use":
		delete(fields, "caller")
	case "thinking":
		delete(fields, "signature")
	}
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

// policyAppendTimeout bounds the append a delta turn posts inside its lease
// after the client may already have its reply. A next turn can arrive then
// and waits for the lease up to the episode's acquire timeout, so the append
// takes at most half of it. It is also no longer than a decide may take.
//
// The bound is VSR's wait, not the service's work: an append the service
// still runs past it holds the service's episode lock, so the next turn
// retries session_busy one acquire timeout longer (append_unsettled). An
// append that outlives even that leaves the next turn session_busy (429),
// as a sync decide that timed out does.
func policyAppendTimeout(total, acquire time.Duration) time.Duration {
	if acquire <= 0 {
		acquire = episodeFinalizeTimeout
	}
	if total <= 0 {
		total = episodeFinalizeTimeout
	}
	return raylineARCDurationMin(total, acquire/2)
}

// armDelta reads, at readiness, whether the service serves staged turns. A
// delta cell whose service does not advertise them sends every turn in
// full, and says so once per readiness check.
func (scorer *policyServiceScorer) armDelta(capabilities []string) {
	if !scorer.delta {
		return
	}
	served := slices.Contains(capabilities, raylinearc.PolicyCapabilityStagedTurns)
	scorer.stagedTurnsServed.Store(served)
	if !served {
		logging.ComponentWarnEvent("extproc", "rayline_arc_policy_delta_unsupported", map[string]interface{}{
			"package_alias": scorer.alias,
			"capability":    raylinearc.PolicyCapabilityStagedTurns,
			"decide_mode":   config.RaylineARCDecideModeSync,
		})
	}
}

// newPolicyStaging is what the turn's transaction needs to append its
// reply: the append goes where the decide went, the one client or the
// episode's session.
func (scorer *policyServiceScorer) newPolicyStaging(
	arcContext *selection.RaylineARCSelectionContext,
	format string,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	roles []string,
	decider policyDecider,
	session *raylinearc.PolicySessionBinding,
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
		admit:         scorer.appendAdmit,
		pkg:           raylinearc.PolicyPackageRef{Alias: scorer.alias, PackageSHA256: scorer.sha256},
		episodeIDHash: arcContext.EpisodeIDHash,
		requestID:     arcContext.RequestID,
		requestFormat: format,
		contextEpoch:  turn.ContextEpoch(),
		messages:      messages,
		roles:         roles,
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
		// Reasoning no captured client shows the echo of is not appended:
		// the next turn decides in full rather than staging a guess.
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
		fail(class)
		if !policyServiceAnswered(err) && !raylinearc.PolicySessionUnanswered(err) {
			// The service may still be appending under its episode lock.
			return &policyStagedTurn{unsettled: true}
		}
		return nil
	}
	if response.Package != staging.pkg || response.Messages != len(messages) {
		return fail("append_mismatch")
	}
	staged := &policyStagedTurn{
		contextEpoch:  request.ContextEpoch,
		requestFormat: staging.requestFormat,
		count:         response.Messages,
		prefixDigest:  response.PrefixDigest,
		digest:        policyDeltaDigest(messages, len(messages)),
		appendedFrom:  len(staging.messages),
		historyDigest: policyDeltaDigest(messages, len(staging.messages)),
	}
	fields := map[string]interface{}{
		"request_id": staging.requestID, "episode_id_hash": staging.episodeIDHash,
		"messages": response.Messages, "latency_ms": time.Since(started).Milliseconds(),
		"session_action": boundedPolicySessionAction(response.Encoding.SessionAction),
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

// anthropicEchoContent is a Messages reply's content as the service stages
// it: a text block carries "citations": [] where the reply had none, and a
// tool_use block "caller": {"type":"direct"} where it named none, as Claude
// Code 2.1.309 echoes it (testdata/thread_continuation). Every other block
// and member is kept as the reply had it. Both members are ones the delta
// digest ignores at those values (policyDeltaCleanBlock), so a client that
// echoes them otherwise matches too; the next turn's digest check is what
// proves the echo matched, and a mismatch costs only a full resend
// (delta_mismatch_reason appended_mismatch).
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
// reasoning whose echo the delta digest cannot be shown to read as the
// service does, read before any normalization: a Messages redacted_thinking
// block or a thinking block that is anything but {type, thinking} with an
// optional string signature, a Chat message's reasoning members, a Responses
// reasoning item. An unreadable reply counts as thinking.
//
// A Messages thinking block is appended, signed or not: the service encodes
// its text and drops its signature whatever its value, as the digest does,
// so Claude Code 2.1.309's unchanged echo (testdata/continuity/cc-2.1.309-
// thread-create-thinking-tool-turn, message 11) and Workshop's, which sends
// an unsigned block's signature as "", both extend it. A client that echoes
// the text otherwise costs only a full resend (appended_mismatch).
func policyReplyThinks(format string, body []byte) bool {
	switch format {
	case policyFormatAnthropic:
		var reply struct {
			Content []map[string]json.RawMessage `json:"content"`
		}
		if json.Unmarshal(body, &reply) != nil {
			return true
		}
		for _, block := range reply.Content {
			var kind string
			_ = json.Unmarshal(block["type"], &kind)
			switch kind {
			case "redacted_thinking":
				return true
			case "thinking":
				var text, signature string
				if json.Unmarshal(block["thinking"], &text) != nil || string(block["thinking"]) == "null" {
					return true
				}
				members := 2
				if raw, ok := block["signature"]; ok {
					if json.Unmarshal(raw, &signature) != nil || string(raw) == "null" {
						return true
					}
					members++
				}
				if len(block) != members {
					return true
				}
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

// stageRaylineARCAppendSource keeps the reply the client receives for the
// turn's append; a turn that stages nothing keeps nothing.
func stageRaylineARCAppendSource(ctx *RequestContext, clientBody []byte) {
	if ctx == nil || ctx.RaylineARCTransaction == nil || ctx.RaylineARCTransaction.policyStaging == nil {
		return
	}
	ctx.RaylineARCTransaction.policyStaging.clientBody = clientBody
}

// policyFirstBusyWait is how long a turn's first decide retries
// session_busy: one acquire timeout, two after an append VSR stopped waiting
// for (unsettled), whose work may still hold the service's episode lock.
func policyFirstBusyWait(busyWait time.Duration, unsettled bool) time.Duration {
	if unsettled {
		return 2 * busyWait
	}
	return busyWait
}

// policyAppendUnsettled reports whether this episode's last committed turn,
// in this process, left an append VSR stopped waiting for. It is read on
// every turn of a delta cell, whether or not this one stages.
func (scorer *policyServiceScorer) policyAppendUnsettled(episodeIDHash string, turnIndex uint64) bool {
	if !scorer.delta {
		return false
	}
	staged, ok := scorer.staged.get(episodeIDHash)
	return ok && staged.turnIndex == turnIndex && staged.unsettled
}

// policyStagedFullMode is the decision mode of a turn sent in full: sync on
// a staged turn, none on a turn that stages nothing. A re-ask after a delta
// decide is sent in full, so it is recorded so.
func policyStagedFullMode(staging bool) string {
	if staging {
		return policyDecisionModeSync
	}
	return ""
}

// addPolicyDecisionModeFields records a delta cell's decision mode on a
// dispatch-time record: delta when the turn was decided on a turn_delta,
// sync when it was sent in full. A sync cell's records are unchanged.
func addPolicyDecisionModeFields(fields map[string]interface{}, trace *selection.RaylineARCTrace) {
	if trace == nil || trace.PolicyDecisionMode == "" {
		return
	}
	fields["decision_mode"] = trace.PolicyDecisionMode
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
