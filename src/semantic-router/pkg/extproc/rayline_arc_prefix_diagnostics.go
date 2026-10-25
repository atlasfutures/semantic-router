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
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyPrefixShapes remembers, per episode, the prefix shapes of the last
// main (or unknown) requests this process decided, so a prefix_break can say
// where the transcript diverged and whether the system prompt or the tools
// changed. It is diagnostic only: in memory, bounded, never persisted, and a
// turn whose previous request another replica decided logs the divergence
// unknown. The episode's own state records only a digest of the whole prefix.
type policyPrefixShapeCache struct {
	mu       sync.Mutex
	episodes map[string][]*raylinearc.PolicyPrefixShape
	order    []string
}

const (
	// policyPrefixShapeEpisodes bounds the episodes remembered; the oldest
	// inserted is forgotten first.
	policyPrefixShapeEpisodes = 512
	// policyPrefixShapesPerEpisode keeps the last two requests: the one the
	// episode committed and a later one that failed before committing.
	policyPrefixShapesPerEpisode = 2
)

var policyPrefixShapes = &policyPrefixShapeCache{episodes: map[string][]*raylinearc.PolicyPrefixShape{}}

// recorded is the newest remembered shape of the prefix state records, or
// nil. Two requests can share a transcript and differ in system prompt or
// tools (a retry); the newer one is what the service last encoded.
func (cache *policyPrefixShapeCache) recorded(
	episode string,
	state *raylinearc.PolicyEpisodeState,
) *raylinearc.PolicyPrefixShape {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	shapes := cache.episodes[episode]
	for index := len(shapes) - 1; index >= 0; index-- {
		if shapes[index].Records(state) {
			return shapes[index]
		}
	}
	return nil
}

func (cache *policyPrefixShapeCache) remember(episode string, shape *raylinearc.PolicyPrefixShape) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	shapes, known := cache.episodes[episode]
	if !known {
		if len(cache.order) >= policyPrefixShapeEpisodes {
			delete(cache.episodes, cache.order[0])
			cache.order = cache.order[1:]
		}
		cache.order = append(cache.order, episode)
	}
	shapes = append(shapes, shape)
	if len(shapes) > policyPrefixShapesPerEpisode {
		shapes = shapes[len(shapes)-policyPrefixShapesPerEpisode:]
	}
	cache.episodes[episode] = shapes
}

// observePolicyPrefix logs why a main or unknown turn did not extend the
// recorded prefix (transition prefix_break), and an extending turn whose
// system prompt or tools changed, which the message digest does not cover
// but the policy service's encoder session does. It then remembers this
// request's shape for the caller to remember once the service has decided it
// (rememberPolicyPrefix): a request refused before the decide was never
// encoded. A side call is neither logged nor remembered: it never records a
// prefix, and its shape is nil.
func observePolicyPrefix(
	arcContext *selection.RaylineARCSelectionContext,
	clientRequest raylinearc.PolicyClientRequest,
	digests *raylinearc.PrefixDigests,
	previous *raylinearc.PolicyEpisodeState,
	signals raylinearc.PolicyTurnSignals,
	transition string,
) *raylinearc.PolicyPrefixShape {
	if signals.CallKind == raylinearc.PolicyCallSide {
		return nil
	}
	system := clientRequest.System
	if clientRequest.Instructions != nil {
		system, _ = json.Marshal(*clientRequest.Instructions)
	}
	current := raylinearc.NewPolicyPrefixShapeFrom(digests, system, clientRequest.Tools)
	if transition != raylinearc.PolicyTransitionPrefixBreak && transition != raylinearc.PolicyTransitionExtends {
		return current
	}
	recorded := policyPrefixShapes.recorded(arcContext.EpisodeIDHash, previous)
	shellChanged := recorded != nil && (recorded.System != current.System || recorded.Tools != current.Tools)
	if transition == raylinearc.PolicyTransitionExtends && !shellChanged {
		return current
	}
	fields := map[string]interface{}{
		"request_id":             arcContext.RequestID,
		"episode_id_hash":        arcContext.EpisodeIDHash,
		"transition":             transition,
		"call_kind":              string(signals.CallKind),
		"previous_message_count": previous.PrefixLen,
		"message_count":          len(digests.Messages()),
		"previous_shape_known":   recorded != nil,
		"system_changed":         nil,
		"tools_changed":          nil,
		"first_divergent_index":  nil,
		"divergent_role":         nil,
		"divergent_block_type":   nil,
		// The divergent message's block layout in this request and in the
		// recorded one: names, lengths and short digests, never content.
		"divergent_layout":          nil,
		"previous_divergent_layout": nil,
	}
	if recorded != nil {
		fields["system_changed"] = recorded.System != current.System
		fields["tools_changed"] = recorded.Tools != current.Tools
		if index, ok := recorded.FirstDivergence(current); ok {
			fields["first_divergent_index"] = index
			// The index is past the current transcript's end when the
			// request is a strict prefix of the recorded one.
			if messages := digests.Messages(); index < len(messages) {
				role, block := raylinearc.PolicyMessageKind(messages[index])
				fields["divergent_role"], fields["divergent_block_type"] = role, block
				fields["divergent_layout"] = raylinearc.NewPolicyMessageLayout(messages[index]).Fields()
			}
			// The same message as the recorded request sent it, when it was
			// among the last ones that request kept the layout of.
			if layout, kept := recorded.Layout(index); kept {
				fields["previous_divergent_layout"] = layout.Fields()
			}
		}
	}
	addPolicyRequestShapeFields(fields, arcContext, clientRequest)
	logging.ComponentEvent("extproc", "rayline_arc_policy_prefix_diagnostic", fields)
	return current
}

// rememberPolicyPrefix keeps a decided request's shape; a nil shape (a side
// call) is not kept.
func rememberPolicyPrefix(episode string, shape *raylinearc.PolicyPrefixShape) {
	if shape != nil {
		policyPrefixShapes.remember(episode, shape)
	}
}

// addPolicyRequestShapeFields adds the request's stream flag, output budget
// and tool count, and whether it carries Claude Code's billing header: the
// fields that tell a harness's helper call from its main turn.
func addPolicyRequestShapeFields(
	fields map[string]interface{},
	arcContext *selection.RaylineARCSelectionContext,
	clientRequest raylinearc.PolicyClientRequest,
) {
	var body struct {
		Stream          *bool `json:"stream"`
		MaxTokens       *int  `json:"max_tokens"`
		MaxOutputTokens *int  `json:"max_output_tokens"`
	}
	_ = json.Unmarshal(arcContext.RawRequest, &body)
	maxTokens := body.MaxTokens
	if maxTokens == nil {
		maxTokens = body.MaxOutputTokens
	}
	var tools []json.RawMessage
	_ = json.Unmarshal(clientRequest.Tools, &tools)
	fields["stream"] = body.Stream != nil && *body.Stream
	fields["max_tokens"] = nil
	if maxTokens != nil {
		fields["max_tokens"] = *maxTokens
	}
	fields["has_tools"] = len(tools) > 0
	fields["tool_count"] = len(tools)
	fields["claude_code_billing_header"] = arcContext.RequestFormat == policyFormatAnthropic &&
		raylinearc.HasClaudeCodeBillingHeader(clientRequest.System)
}

// policyTokenSplit is a policy-service decision's encoded tokens split into
// those its encoder session already held and those it encoded, as far as the
// decide response says: it reports only the token count and the session
// action. A created or rebuilt session encoded every token and a reused one
// none; an appended session's split is not reported, so both are nil. The
// trace's own counters stay as they were, because the decision service
// meters cache reads from them.
func policyTokenSplit(trace *selection.RaylineARCTrace) (interface{}, interface{}) {
	if trace.TokenCountsUnknown {
		return nil, nil
	}
	switch trace.SessionAction {
	case "created", "rebuilt":
		return 0, trace.SerializedTokens
	case "reused":
		return trace.SerializedTokens, 0
	}
	return nil, nil
}
