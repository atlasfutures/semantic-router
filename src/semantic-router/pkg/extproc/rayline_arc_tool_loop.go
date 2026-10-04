package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// This file holds a tool loop on the model family that opened it
// (hold_family_in_tool_loop).
//
// A turn is mid-loop when its last message returns the result of a tool call
// the assistant made, and the arm that made the call is the episode's
// previous arm: the last turn the episode committed. A switch there changes
// the reasoning the loop was built on (see protocolcodec's reasoning rows):
// a Messages target drops unsigned reasoning, and a Chat or Responses target
// strips Claude's signature and drops its redacted_thinking. On DeepSeek and
// MiMo a tool history without their own reasoning costs the turn itself.

// toolLoopForeignArms marks the arms outside the previous arm's family when
// this turn answers that arm's tool call. It returns nil, and selection is
// unchanged, when the hold is off, the episode has no previous arm, the turn
// is not mid-loop, or every arm is of the one family.
func (r *OpenAIRouter) toolLoopForeignArms(
	arc *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	state *raylinearc.EpisodeState,
	modelRefs []config.ModelRef,
) ([]bool, int, string) {
	if arc == nil || !arc.HoldFamilyInToolLoop || state == nil || state.PreviousArm == nil ||
		reqCtx == nil || reqCtx.SemanticRequest == nil {
		return nil, -1, ""
	}
	previous := *state.PreviousArm
	if previous < 0 || previous >= len(modelRefs) || !answersToolCall(reqCtx.SemanticRequest.Messages) {
		return nil, -1, ""
	}
	family := r.armFamily(modelRefs[previous].Model)
	foreign := make([]bool, len(modelRefs))
	marked := false
	for index, ref := range modelRefs {
		if r.armFamily(ref.Model) != family {
			foreign[index] = true
			marked = true
		}
	}
	if !marked {
		return nil, -1, ""
	}
	return foreign, previous, family
}

// answersToolCall reports whether the last message returns the result of a
// tool call made by the assistant since the last user message: a Chat tool
// message, an Anthropic user message holding tool_result blocks (with or
// without text beside them), or a Responses function_call_output. A result
// whose call lives in stored response state (DeferredLink) answers a call by
// definition.
func answersToolCall(messages []llmprotocol.Message) bool {
	if len(messages) == 0 {
		return false
	}
	results, deferred := toolResultCallIDs(messages[len(messages)-1])
	if deferred {
		return true
	}
	if len(results) == 0 {
		return false
	}
	for index := len(messages) - 2; index >= 0; index-- {
		message := messages[index]
		switch message.Role {
		case llmprotocol.RoleAssistant:
			for _, content := range message.Content {
				if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil {
					continue
				}
				if _, answered := results[content.ToolCall.ID]; answered {
					return true
				}
			}
		case llmprotocol.RoleUser:
			// Another tool result in the same loop; a user message with
			// none opens the turn, and the search ends there.
			if earlier, deferredEarlier := toolResultCallIDs(message); len(earlier) == 0 && !deferredEarlier {
				return false
			}
		}
		// Tool and system messages sit inside the loop; the search goes on.
	}
	return false
}

// toolResultCallIDs names the calls a message returns results for, and
// reports whether any result's call lives in stored response state.
func toolResultCallIDs(message llmprotocol.Message) (map[string]struct{}, bool) {
	ids := map[string]struct{}{}
	deferred := false
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
			continue
		}
		if content.ToolResult.DeferredLink {
			deferred = true
			continue
		}
		if content.ToolResult.CallID != "" {
			ids[content.ToolResult.CallID] = struct{}{}
		}
	}
	return ids, deferred
}

// armFamily is the model family an arm serves: who made the model, so the
// thinking-on and thinking-off arms of one model, and two sizes of one
// vendor's model, are one family. The model card's publisher says so when
// the card has one. Otherwise it is the vendor segment of the provider model
// the arm dispatches to (deepseek/deepseek-v4-flash is deepseek), which holds
// only where the provider namespaces models by vendor: a provider that serves
// them under its own namespace (local/..., accounts/...) needs the publisher
// on the card. A name with no vendor segment is its own family, without any
// @variant suffix.
//
// A metadata-only model, dispatched by an external gateway, has no backend
// to resolve the provider model through. Its provider_model_id is still its
// own default model id, so that is the name the vendor is read from: an
// alias claude-sonnet bound to anthropic/claude-sonnet-4.5 is anthropic.
func (r *OpenAIRouter) armFamily(model string) string {
	name := strings.TrimSpace(model)
	if r != nil && r.Config != nil {
		params, known := r.Config.ModelConfig[name]
		if known && strings.TrimSpace(params.Publisher) != "" {
			return strings.ToLower(strings.TrimSpace(params.Publisher))
		}
		if _, backend, found, err := r.Config.ResolvePrimaryBackendForModel(name); err == nil && found {
			name = r.Config.ResolveExternalModelID(name, backend)
		} else if providerModel := strings.TrimSpace(params.ExternalModelIDs["default"]); known && providerModel != "" {
			name = providerModel
		}
	}
	if vendor, _, ok := strings.Cut(name, "/"); ok && vendor != "" {
		return strings.ToLower(vendor)
	}
	if base, _, ok := strings.Cut(name, "@"); ok && base != "" {
		name = base
	}
	return strings.ToLower(name)
}

// withToolLoopFamily folds the tool-loop hold into the mask. It runs last, so
// every hard constraint has already said what it excludes, and it yields to
// them: when the family has no eligible arm left, the hold is lifted rather
// than refusing the turn. Both outcomes are logged.
func withToolLoopFamily(
	arcContext *selection.RaylineARCSelectionContext,
	armCount int,
	excluded []bool,
) []bool {
	if len(arcContext.ToolLoopForeignArms) != armCount {
		return excluded
	}
	combined := make([]bool, armCount)
	eligible, foreign := 0, 0
	for index := range combined {
		hard := len(excluded) == armCount && excluded[index]
		combined[index] = hard || arcContext.ToolLoopForeignArms[index]
		if !combined[index] {
			eligible++
		}
		if arcContext.ToolLoopForeignArms[index] && !hard {
			foreign++
		}
	}
	outcome := "held"
	if eligible == 0 {
		outcome = "lifted_no_eligible_arm"
	}
	logToolLoopHold(arcContext, outcome, foreign)
	if eligible == 0 {
		return excluded
	}
	return combined
}

func logToolLoopHold(arcContext *selection.RaylineARCSelectionContext, outcome string, foreign int) {
	logging.ComponentEvent("extproc", "rayline_arc_tool_loop_family_hold", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"previous_arm":    arcContext.ToolLoopArm,
		"family":          arcContext.ToolLoopFamily,
		"outcome":         outcome,
		"excluded_arms":   foreign,
	})
}
