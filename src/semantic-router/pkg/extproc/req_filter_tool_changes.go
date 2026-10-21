package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
// A decision whose tools plugin filters tools (mode none or filtered) has
// already settled which tools travel, so its additions are dropped rather
// than added: an addition must not bring back a tool the policy removed.
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
	toolsCfg := resolveDecisionToolsConfig(ctx)
	admitAdditions := toolsCfg == nil || !toolsCfg.Enabled ||
		toolsCfg.EffectiveMode() == config.ToolsPluginModePassthrough

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
