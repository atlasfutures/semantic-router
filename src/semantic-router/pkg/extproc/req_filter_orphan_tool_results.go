package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Claude Code's side calls (the conversation summary, for one) send a cut-down
// history: the messages before a cut are gone, so the first user message can
// open with tool_result blocks whose tool_use is not in the request. The
// ingress codec, the turn projection and the policy service all refuse such a
// result (orphan_tool_result, unresolved_tool_id, "tool result has no pending
// call"), so the turn failed with a 400 the client did nothing to cause.
//
// The result's content is still conversation the model should read; only its
// link is gone. foldOrphanToolResults therefore rewrites each orphan result as
// the ordinary user content it carries, before anything decodes the body, so
// the codec, the projection, the policy service and the provider all see the
// same served shape.
const (
	toolResultBlockType = "tool_result"
	toolUseBlockType    = "tool_use"
	// toolReferenceBlockType names a tool ToolSearch loaded; Messages accepts
	// it only inside a tool_result.
	toolReferenceBlockType = "tool_reference"
	// emptyOrphanToolResultText stands in for an orphan result with no content:
	// Messages refuses an empty text block, and dropping the block could leave
	// its message with nothing in it.
	emptyOrphanToolResultText = "(empty tool result)"
)

// foldOrphanToolResults rewrites, in a Messages body, every tool_result block
// whose tool_use_id names no tool_use in an earlier message as the content it
// carries: a string becomes one text block, a block list is kept as it is, and
// a cache breakpoint the result carried moves to the last block it becomes.
// is_error has no user-content form and is not carried. It returns the body
// and the number of results folded; any other body is returned unchanged.
func foldOrphanToolResults(body []byte, format llmprotocol.WireFormat) ([]byte, int, error) {
	// No byte-level shortcut: a block type may arrive JSON-escaped.
	if format != llmprotocol.AnthropicMessagesV1 {
		return body, 0, nil
	}
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return body, 0, nil
	}
	out := body
	folded := 0
	calls := map[string]struct{}{}
	for index, message := range messages.Array() {
		content := message.Get("content")
		if !content.IsArray() {
			continue
		}
		blocks := content.Array()
		// Messages wants a message's tool results before its other content,
		// so the results that stay lead and the folded content joins the rest.
		linkedResults := make([]json.RawMessage, 0, len(blocks))
		rest := make([]json.RawMessage, 0, len(blocks))
		changed := false
		// movedFolds indexes, in rest, folded blocks that move past a linked
		// result. Their cache breakpoints are dropped: Messages wants longer
		// TTLs before shorter ones, and moving a breakpoint can break that.
		var movedFolds, pendingFolds []int
		for _, block := range blocks {
			if block.Get("type").String() != toolResultBlockType {
				rest = append(rest, json.RawMessage(block.Raw))
				continue
			}
			if _, linked := calls[block.Get("tool_use_id").String()]; linked {
				linkedResults = append(linkedResults, json.RawMessage(block.Raw))
				movedFolds = append(movedFolds, pendingFolds...)
				pendingFolds = nil
				continue
			}
			replacement, err := orphanToolResultContent(block)
			if err != nil {
				return nil, 0, err
			}
			for _, replaced := range replacement {
				pendingFolds = append(pendingFolds, len(rest))
				rest = append(rest, replaced)
			}
			changed = true
			folded++
		}
		for _, moved := range movedFolds {
			if !gjson.GetBytes(rest[moved], "cache_control").Exists() {
				continue
			}
			unmarked, err := sjson.DeleteBytes(rest[moved], "cache_control")
			if err != nil {
				return nil, 0, fmt.Errorf("failed to drop a moved cache breakpoint: %w", err)
			}
			rest[moved] = unmarked
		}
		rewritten := append(linkedResults, rest...)
		// Calls are recorded after the message's results are checked: a
		// result answers a call from an earlier message, never its own.
		for _, block := range blocks {
			if block.Get("type").String() == toolUseBlockType {
				calls[block.Get("id").String()] = struct{}{}
			}
		}
		if !changed {
			continue
		}
		raw, err := json.Marshal(rewritten)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to encode a message's folded tool results: %w", err)
		}
		if out, err = sjson.SetRawBytes(out, fmt.Sprintf("messages.%d.content", index), raw); err != nil {
			return nil, 0, fmt.Errorf("failed to fold a message's orphan tool results: %w", err)
		}
	}
	return out, folded, nil
}

// foldIngressOrphanToolResults is foldOrphanToolResults at ingress: it logs a
// fold, and a body it cannot rewrite travels as received (and is refused as
// before) rather than failing the turn here.
func foldIngressOrphanToolResults(body []byte, format llmprotocol.WireFormat, requestID string) []byte {
	out, folded, err := foldOrphanToolResults(body, format)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "orphan_tool_result_fold_failed", map[string]interface{}{
			"request_id": requestID, "error": err.Error(),
		})
		return body
	}
	if folded > 0 {
		logging.ComponentEvent("extproc", "orphan_tool_results_folded", map[string]interface{}{
			"request_id": requestID, "folded": folded,
		})
	}
	return out
}

// textBlock is a Messages text block, its members in wire order.
type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// orphanToolResultContent is the user content one orphan tool_result carries.
// Two kinds of nested block do not survive the move to the top level, so they
// are left out: a tool_reference (ToolSearch's result) is valid only inside a
// tool_result, and Messages refuses a blank top-level text block that it
// accepts inside one.
func orphanToolResultContent(block gjson.Result) ([]json.RawMessage, error) {
	var blocks []json.RawMessage
	switch content := block.Get("content"); {
	case content.IsArray():
		for _, nested := range content.Array() {
			kind := nested.Get("type").String()
			if kind == toolReferenceBlockType || kind == "text" && strings.TrimSpace(nested.Get("text").String()) == "" {
				continue
			}
			blocks = append(blocks, json.RawMessage(nested.Raw))
		}
	case content.Type == gjson.String && strings.TrimSpace(content.String()) != "":
		text, err := json.Marshal(textBlock{Type: "text", Text: content.String()})
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, text)
	}
	if len(blocks) == 0 {
		text, err := json.Marshal(textBlock{Type: "text", Text: emptyOrphanToolResultText})
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, text)
	}
	cache := block.Get("cache_control")
	last := blocks[len(blocks)-1]
	if cache.Exists() && !gjson.GetBytes(last, "cache_control").Exists() {
		marked, err := sjson.SetRawBytes(last, "cache_control", []byte(cache.Raw))
		if err != nil {
			return nil, fmt.Errorf("failed to move an orphan tool result's cache breakpoint: %w", err)
		}
		blocks[len(blocks)-1] = marked
	}
	return blocks, nil
}
