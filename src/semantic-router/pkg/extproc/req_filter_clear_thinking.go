package extproc

import (
	"encoding/json"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// dropClearThinkingWithoutThinking removes the client's clear_thinking_*
// context-management edits from a Messages body that travels without thinking
// on. Claude Code sends clear_thinking_20251015 beside adaptive thinking; a
// thinking-off arm sends thinking disabled (or none), and Anthropic refuses the
// edit then, which OpenRouter relays as 400 "Invalid Anthropic Messages API
// request" (prod 2026-10-21, claude-sonnet-5.5 on the thinking-off action).
//
// It reads the final bytes, after the codec, its source replay and the planned
// v5 control render, so it judges the thinking that actually travels. The edit
// only clears thinking blocks, so a turn without thinking loses nothing. Other
// edits stay; context_management goes when it is left with none.
//
// Only Claude models are touched. OpenRouter accepts the edit beside no
// thinking for the other Messages arms (GLM, Mistral), and their bytes are
// pinned to pathfinder's renderer by the v5 golden corpus.
func dropClearThinkingWithoutThinking(body []byte) ([]byte, error) {
	model := gjson.GetBytes(body, "model").String()
	if !strings.HasPrefix(model, "anthropic/") && !strings.HasPrefix(model, "claude-") {
		return body, nil
	}
	edits := gjson.GetBytes(body, "context_management.edits")
	if !edits.IsArray() {
		return body, nil
	}
	switch gjson.GetBytes(body, "thinking.type").String() {
	case "adaptive", "enabled":
		return body, nil
	case "":
		// An effort with no thinking member is adaptive thinking on a Claude
		// 5 model: dev 2026-10-21, opus-5.5 took the edit beside effort high
		// and no thinking, 7 of 7 turns 200. Those bytes stay as they are.
		if gjson.GetBytes(body, "output_config.effort").Exists() {
			return body, nil
		}
	}
	kept := make([]json.RawMessage, 0, len(edits.Array()))
	for _, edit := range edits.Array() {
		if strings.HasPrefix(edit.Get("type").String(), "clear_thinking_") {
			continue
		}
		kept = append(kept, json.RawMessage(edit.Raw))
	}
	if len(kept) == len(edits.Array()) {
		return body, nil
	}
	management := gjson.GetBytes(body, "context_management").Map()
	if len(kept) == 0 && len(management) == 1 {
		return sjson.DeleteBytes(body, "context_management")
	}
	encoded, err := json.Marshal(kept)
	if err != nil {
		return nil, err
	}
	return sjson.SetRawBytes(body, "context_management.edits", encoded)
}
