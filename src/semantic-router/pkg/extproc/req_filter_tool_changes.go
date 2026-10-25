package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Claude Code's mid-conversation tool changes (beta
// mid-conversation-tool-changes-2026-07-01) announce tools that load after a
// session starts, such as claude.ai connector tools, as content blocks inside
// a role:"system" message:
//
//	{"type":"tool_addition","tool":{"type":"tool_definition","definition":{...}}}
//	{"type":"tool_addition","tool":{"type":"tool_reference","name":"..."}}
//	{"type":"tool_removal","tool":{"type":"tool_reference","name":"..."}}
//
// OpenRouter's Messages API does not accept these blocks: it refuses the whole
// request at validation with 400 "Invalid Anthropic Messages API request",
// before any generation, so every turn after a tool loads fails.
const (
	toolAdditionBlockType = "tool_addition"
	toolRemovalBlockType  = "tool_removal"
)

// foldMidConversationToolChanges rewrites a Messages body bound for
// OpenRouter so the tool set the blocks describe is stated the way OpenRouter
// reads it: definitions are added to top-level `tools`, removals drop the
// named tool, and the blocks themselves leave the transcript. The changes are
// applied in transcript order, so the tools sent are the ones in force at the
// final turn. A message left with no content is dropped; a cache breakpoint a
// dropped block carried moves to the last block its message keeps.
//
// A decision that narrows tools (decisionLeavesToolsToClient is false) has
// already settled which tools travel, so its additions are dropped rather
// than added: an addition must not bring back a tool it removed.
func foldMidConversationToolChanges(body []byte, dispatch *providerDispatch, ctx *RequestContext) ([]byte, error) {
	if dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		!providerIsOpenRouter(dispatch.profile) {
		return body, nil
	}
	if !bytes.Contains(body, []byte(toolAdditionBlockType)) && !bytes.Contains(body, []byte(toolRemovalBlockType)) {
		return body, nil
	}
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return body, nil
	}

	tools := []json.RawMessage{}
	toolIndex := map[string]int{}
	for _, tool := range gjson.GetBytes(body, "tools").Array() {
		if name := tool.Get("name").String(); name != "" {
			toolIndex[name] = len(tools)
		}
		tools = append(tools, json.RawMessage(tool.Raw))
	}
	removed := map[int]bool{}
	admitAdditions := decisionLeavesToolsToClient(ctx, body)

	changed := false
	keptMessages := []json.RawMessage{}
	for _, message := range messages.Array() {
		content := message.Get("content")
		if !content.IsArray() {
			keptMessages = append(keptMessages, json.RawMessage(message.Raw))
			continue
		}
		keptBlocks := []json.RawMessage{}
		var orphanCacheControl string
		folded := false
		for _, block := range content.Array() {
			switch block.Get("type").String() {
			case toolAdditionBlockType:
				if !admitAdditions {
					break
				}
				if definition := block.Get("tool.definition"); block.Get("tool.type").String() == "tool_definition" &&
					definition.IsObject() {
					name := definition.Get("name").String()
					if at, ok := toolIndex[name]; ok && name != "" {
						tools[at] = json.RawMessage(definition.Raw)
						delete(removed, at)
					} else {
						if name != "" {
							toolIndex[name] = len(tools)
						}
						tools = append(tools, json.RawMessage(definition.Raw))
					}
				} else if at, ok := toolIndex[block.Get("tool.name").String()]; ok {
					// A reference loads a tool already declared, usually
					// deferred: it is in force, and no longer deferred.
					delete(removed, at)
					if gjson.GetBytes(tools[at], "defer_loading").Exists() {
						loaded, err := sjson.DeleteBytes(append([]byte(nil), tools[at]...), "defer_loading")
						if err != nil {
							return nil, fmt.Errorf("failed to load a referenced tool: %w", err)
						}
						tools[at] = loaded
					}
				}
			case toolRemovalBlockType:
				if at, ok := toolIndex[block.Get("tool.name").String()]; ok {
					removed[at] = true
				}
			default:
				keptBlocks = append(keptBlocks, json.RawMessage(block.Raw))
				continue
			}
			folded = true
			if cc := block.Get("cache_control"); cc.Exists() {
				orphanCacheControl = cc.Raw
			}
		}
		if !folded {
			keptMessages = append(keptMessages, json.RawMessage(message.Raw))
			continue
		}
		changed = true
		if len(keptBlocks) == 0 {
			continue
		}
		if orphanCacheControl != "" {
			last := keptBlocks[len(keptBlocks)-1]
			if !gjson.GetBytes(last, "cache_control").Exists() {
				moved, err := sjson.SetRawBytes(append([]byte(nil), last...), "cache_control", []byte(orphanCacheControl))
				if err != nil {
					return nil, fmt.Errorf("failed to move a tool change's cache breakpoint: %w", err)
				}
				keptBlocks[len(keptBlocks)-1] = moved
			}
		}
		encodedBlocks, err := json.Marshal(keptBlocks)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize message content: %w", err)
		}
		rewritten, err := sjson.SetRawBytes([]byte(message.Raw), "content", encodedBlocks)
		if err != nil {
			return nil, fmt.Errorf("failed to rewrite message content: %w", err)
		}
		keptMessages = append(keptMessages, rewritten)
	}
	if !changed {
		return body, nil
	}

	encodedMessages, err := json.Marshal(keptMessages)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize messages: %w", err)
	}
	out, err := sjson.SetRawBytes(body, "messages", encodedMessages)
	if err != nil {
		return nil, fmt.Errorf("failed to rewrite messages: %w", err)
	}
	inForce := []json.RawMessage{}
	for at, tool := range tools {
		if !removed[at] {
			inForce = append(inForce, tool)
		}
	}
	if len(inForce) == 0 {
		return sjson.DeleteBytes(out, "tools")
	}
	encodedTools, err := json.Marshal(inForce)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize tools: %w", err)
	}
	return sjson.SetRawBytes(out, "tools", encodedTools)
}

// decisionLeavesToolsToClient reports that the router sends the client's
// tools as declared, so a mid-conversation addition may join them. Two
// checks, because either alone misses a case:
//   - outcome: every tool the client declared is still in the dispatched
//     body. Any narrowing path (a tools plugin, selection, blocked params)
//     shows up here without being named.
//   - policy: no decision setting that narrows tools is on. This covers a
//     client that declared no tools, where the outcome check has nothing to
//     compare.
func decisionLeavesToolsToClient(ctx *RequestContext, body []byte) bool {
	if ctx == nil {
		return true
	}
	dispatched := map[string]bool{}
	for _, name := range toolNames(body) {
		dispatched[name] = true
	}
	for _, name := range toolNames(ctx.ProtocolEnvelope.Request) {
		if !dispatched[name] {
			return false
		}
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil {
		return true
	}
	if toolsCfg := resolveDecisionToolsConfig(ctx); toolsCfg != nil && toolsCfg.Enabled &&
		(toolsCfg.EffectiveMode() != config.ToolsPluginModePassthrough || toolsCfg.SelectionEnabled()) {
		return false
	}
	if selection := decision.GetToolSelectionConfig(); selection != nil && selection.Enabled {
		return false
	}
	if params := decision.GetRequestParamsConfig(); params != nil {
		for _, blocked := range params.BlockedParams {
			if blocked == "tools" {
				return false
			}
		}
	}
	return true
}

// toolNames are the names a body declares in `tools`, in the Messages and
// Responses shape (name) or the Chat shape (function.name).
func toolNames(body []byte) []string {
	var names []string
	for _, tool := range gjson.GetBytes(body, "tools").Array() {
		if name := tool.Get("name").String(); name != "" {
			names = append(names, name)
		} else if name := tool.Get("function.name").String(); name != "" {
			names = append(names, name)
		}
	}
	return names
}

// Claude Code's per-turn control (beta per-turn-control-2026-07-01) sets
// generation control on an in-band message, {"role":"system",
// "output_config":{"effort":"high"},"content":[...]}, usually the message that
// also carries its tool changes. The Messages codec carries an unnamed
// message member through to a same-format target, so OpenRouter would receive
// it; its Messages API validates the transcript against the public schema,
// which has no message-level output_config. The request's own top-level
// output_config still states the turn's effort.

// stripMessageOutputConfig removes output_config from every message of a
// Messages body bound for OpenRouter. Any other body is returned unchanged.
func stripMessageOutputConfig(body []byte, dispatch *providerDispatch) ([]byte, error) {
	if dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		!providerIsOpenRouter(dispatch.profile) || !bytes.Contains(body, []byte(`"output_config"`)) {
		return body, nil
	}
	out := body
	for index, message := range gjson.GetBytes(body, "messages").Array() {
		if !message.Get("output_config").Exists() {
			continue
		}
		stripped, err := sjson.DeleteBytes(out, fmt.Sprintf("messages.%d.output_config", index))
		if err != nil {
			return nil, fmt.Errorf("failed to strip a message's output_config: %w", err)
		}
		out = stripped
	}
	return out, nil
}
