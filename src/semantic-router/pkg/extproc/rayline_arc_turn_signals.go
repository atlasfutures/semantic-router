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
	"encoding/json"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Explicit turn-signal headers, for a gateway or harness that knows what a
// request is. They take precedence over anything read from the body.
const (
	// raylineARCCallKindHeader is "main" or "side". A side call (compaction
	// helper, title generation, any call outside the main conversation)
	// keeps the held arm and leaves the episode unchanged.
	raylineARCCallKindHeader = "x-rayline-call-kind"
	// raylineARCCompactionHeader is the positive ordinal of the compaction
	// this request is the first request after. A repeated ordinal is a
	// repeat, not a new compaction.
	raylineARCCompactionHeader = "x-rayline-compaction"
	// raylineARCAgentKeySourceHeader says how the gateway keyed a harness
	// subagent's episode (x-rayline-session): "agent" when on the harness's
	// agent id, otherwise "role" or "task". Only "agent" is an identity, so
	// only it makes the subagent its own episode.
	raylineARCAgentKeySourceHeader = "x-rayline-agent-key-source"
	// raylineARCParentSessionHeader and raylineARCParentAgentHeader name the
	// conversation a subagent was spawned from: its episode id and the
	// parent's agent id. Recorded as metadata only.
	raylineARCParentSessionHeader = "x-rayline-parent-session"
	raylineARCParentAgentHeader   = "x-rayline-parent-agent"
)

// raylineARCTrustedSignalHeaders returns the turn-signal headers this cell
// reads: the request's, when a gateway is configured to set them, and none
// otherwise, so a client cannot mark its own turns as side calls.
func raylineARCTrustedSignalHeaders(
	policy *config.RaylineARCPolicyServiceConfig,
	requestHeaders map[string]string,
) map[string]string {
	if policy == nil || !policy.TrustTurnSignalHeaders {
		return map[string]string{}
	}
	trusted := map[string]string{}
	for _, header := range raylineARCTurnSignalHeadersForRemoval() {
		trusted[header] = requestHeaders[header]
	}
	return trusted
}

// raylineARCTurnSignalHeadersForRemoval are never forwarded upstream: they
// speak to this router, not to a provider.
func raylineARCTurnSignalHeadersForRemoval() []string {
	return []string{
		raylineARCCallKindHeader, raylineARCCompactionHeader,
		raylineARCAgentKeySourceHeader, raylineARCParentSessionHeader, raylineARCParentAgentHeader,
	}
}

// Sources a call kind can come from, as logged.
const (
	callKindSourceHeader             = "header"
	callKindSourceClaudeSubagent     = "claude_code_subagent"
	callKindSourceClaudeSubagentOwn  = "claude_code_subagent_own_episode"
	callKindSourceClaudeCompactionRq = "claude_code_compaction_request"
	callKindSourceClaudeTitle        = "claude_code_title"
	callKindSourceCodexCompactionRq  = "codex_compaction_request"
)

// raylineARCPolicyTurnSignals reads what this request says about itself:
// whether it is a main turn or a side call, and whether it opens a compacted
// context. Only explicit headers and literals a harness writes count; a
// request with neither is unknown and is served as a main turn, as before.
func raylineARCPolicyTurnSignals(
	headers map[string]string,
	format string,
	request raylinearc.PolicyClientRequest,
	messages []json.RawMessage,
	episodeIDHash string,
) raylinearc.PolicyTurnSignals {
	signals := raylinearc.PolicyTurnSignals{}
	signals.CallKind, signals.CallKindSource = raylineARCPolicyCallKind(
		headers, format, request.System, messages, request.Input, episodeIDHash,
	)
	compaction := raylinearc.PolicyCompactionSignal{}
	if raw := strings.TrimSpace(headers[raylineARCCompactionHeader]); raw != "" {
		if ordinal, err := strconv.Atoi(raw); err == nil && ordinal > 0 {
			compaction.Ordinal = ordinal
		} else {
			logInvalidTurnSignal(raylineARCCompactionHeader, episodeIDHash)
		}
	}
	switch format {
	case policyFormatAnthropic:
		compaction.SummaryDigest = raylinearc.ClaudeCodeCompactionSummaryDigest(messages)
	case policyFormatResponses:
		compaction.SummaryDigest = raylinearc.CodexCompactionSummaryDigest(request.Input)
	}
	if compaction.Ordinal > 0 || compaction.SummaryDigest != "" {
		signals.Compaction = &compaction
	}
	// A subagent's link is recorded whenever the gateway keyed one (any
	// harness: codex subagents are keyed on their own thread id) or Claude
	// Code declared one in its billing block.
	if strings.TrimSpace(headers[raylineARCAgentKeySourceHeader]) != "" ||
		(format == policyFormatAnthropic && raylinearc.ClaudeCodeSubagentClaim(request.System)) {
		signals.Subagent = raylineARCSubagentSignal(headers)
	}
	return signals
}

// raylineARCSubagentSignal reads how a subagent was keyed and what spawned it.
func raylineARCSubagentSignal(headers map[string]string) *raylinearc.PolicySubagentSignal {
	signal := &raylinearc.PolicySubagentSignal{KeySource: "unknown"}
	switch source := strings.TrimSpace(headers[raylineARCAgentKeySourceHeader]); source {
	case raylinearc.SubagentKeySourceAgent, "role", "task":
		signal.KeySource = source
	}
	if parent := strings.TrimSpace(headers[raylineARCParentSessionHeader]); parent != "" {
		signal.ParentEpisodeIDHash = raylinearc.HashEpisodeID(parent)
	}
	if agent := strings.TrimSpace(headers[raylineARCParentAgentHeader]); agent != "" {
		signal.ParentAgentIDHash = raylinearc.HashEpisodeID(agent)
	}
	return signal
}

// raylineARCPolicyCallKind classifies a request as a main turn or a side
// call. The header wins; otherwise a harness literal decides; otherwise the
// call is unknown.
func raylineARCPolicyCallKind(
	headers map[string]string,
	format string,
	system json.RawMessage,
	messages []json.RawMessage,
	input []json.RawMessage,
	episodeIDHash string,
) (raylinearc.PolicyCallKind, string) {
	switch value := strings.ToLower(strings.TrimSpace(headers[raylineARCCallKindHeader])); value {
	case "":
	case string(raylinearc.PolicyCallMain), string(raylinearc.PolicyCallSide):
		return raylinearc.PolicyCallKind(value), callKindSourceHeader
	default:
		logInvalidTurnSignal(raylineARCCallKindHeader, episodeIDHash)
	}
	switch format {
	case policyFormatAnthropic:
		switch {
		case raylinearc.ClaudeCodeSubagentClaim(system):
			// A subagent keyed on its own agent id is its own conversation:
			// a main turn of its own episode, deciding its own model and
			// level. Keyed any other way (a role, a task digest, nothing)
			// it is not an identity, so it stays a side call as before.
			if strings.TrimSpace(headers[raylineARCAgentKeySourceHeader]) == raylinearc.SubagentKeySourceAgent {
				return raylinearc.PolicyCallMain, callKindSourceClaudeSubagentOwn
			}
			return raylinearc.PolicyCallSide, callKindSourceClaudeSubagent
		case raylinearc.IsClaudeCodeTitleRequest(system):
			return raylinearc.PolicyCallSide, callKindSourceClaudeTitle
		case raylinearc.IsClaudeCodeCompactionRequest(messages):
			return raylinearc.PolicyCallSide, callKindSourceClaudeCompactionRq
		}
	case policyFormatResponses:
		if raylinearc.IsCodexCompactionRequest(input) {
			return raylinearc.PolicyCallSide, callKindSourceCodexCompactionRq
		}
	}
	return raylinearc.PolicyCallUnknown, ""
}

// raylineARCPolicyCallKindOfBody classifies the request from its body as
// received, before the episode is read, so a side call never takes the
// episode lease. It reads the same fields selection does: a Responses
// request's retained history is prepended, so its last input item is the
// body's.
func raylineARCPolicyCallKindOfBody(
	headers map[string]string,
	format string,
	body []byte,
	episodeIDHash string,
) (raylinearc.PolicyCallKind, string) {
	var request struct {
		System   json.RawMessage `json:"system"`
		Messages json.RawMessage `json:"messages"`
		Input    json.RawMessage `json:"input"`
	}
	_ = json.Unmarshal(body, &request)
	var messages, input []json.RawMessage
	if format == policyFormatAnthropic {
		_ = json.Unmarshal(request.Messages, &messages)
	}
	if format == policyFormatResponses {
		// The Responses shorthand: a string input is one user message.
		var text string
		if json.Unmarshal(request.Input, &text) == nil {
			item, _ := json.Marshal(map[string]string{"type": "message", "role": "user", "content": text})
			input = []json.RawMessage{item}
		} else {
			_ = json.Unmarshal(request.Input, &input)
		}
	}
	return raylineARCPolicyCallKind(headers, format, request.System, messages, input, episodeIDHash)
}

func logInvalidTurnSignal(header string, episodeIDHash string) {
	logging.ComponentWarnEvent("extproc", "rayline_arc_turn_signal_invalid", map[string]interface{}{
		"header": header, "episode_id_hash": episodeIDHash,
	})
}

// logRaylineARCPolicyTurn records how a policy-service request was read
// against its episode. An unknown call kind is reported as unknown, not as
// the main turn it is served as.
func logRaylineARCPolicyTurn(
	episodeIDHash string,
	signals raylinearc.PolicyTurnSignals,
	transition string,
	completedTurns uint64,
	turn *raylinearc.PolicyEpisodeState,
	held bool,
	boundaryRetained bool,
) {
	fields := map[string]interface{}{
		"episode_id_hash":   episodeIDHash,
		"call_kind":         string(signals.CallKind),
		"call_kind_source":  signals.CallKindSource,
		"transition":        transition,
		"completed_turns":   completedTurns,
		"epoch_start_turn":  turn.EpochStartTurn,
		"compaction_count":  turn.CompactionCount,
		"context_epoch":     turn.Epoch,
		"model_held":        held,
		"boundary_retained": boundaryRetained,
	}
	if subagent := signals.Subagent; subagent != nil {
		fields["subagent_key_source"] = subagent.KeySource
		fields["parent_episode_id_hash"] = subagent.ParentEpisodeIDHash
		fields["parent_agent_id_hash"] = subagent.ParentAgentIDHash
	}
	logging.ComponentEvent("extproc", "rayline_arc_policy_turn", fields)
}
