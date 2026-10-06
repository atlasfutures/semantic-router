package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// refusedArgumentText is argument text no log line may carry.
const refusedArgumentText = "SECRET-ARGUMENT-TEXT"

// toolArgumentsStream is an Anthropic stream whose edit tool_use block's
// arguments arrive as chunks, then tail.
func toolArgumentsStream(chunks []string, tail string) []string {
	frames := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"edit\",\"input\":{}}}\n\n",
	}
	for _, chunk := range chunks {
		partial, _ := json.Marshal(chunk)
		frames = append(frames, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":"+string(partial)+"}}\n\n")
	}
	return append(frames, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n", tail)
}

// A provider's tool arguments refused mid-stream leave a
// tool_arguments_refused line that says why, joined to the turn by its
// request_id, and no line of the turn carries the arguments' text.
func TestRefusedToolArgumentsLogTheirFactsWithoutContent(t *testing.T) {
	arguments := "{\"path\":\"" + refusedArgumentText + ".py\",\"edits\":[{\"oldText\":\"a\nb\"}]}"
	stop := "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	chunks := toolArgumentsStream([]string{arguments[:12], arguments[12:]}, stop)
	_, _, logs := driveMaxTokensToolCut(t, llmprotocol.OpenAIResponsesV1, chunks)

	failed := findLogEvent(t, logs, "turn_failed")
	if failed["failure_detail"] != "invalid_stream_tool_arguments" {
		t.Fatalf("turn_failed = %#v, want invalid_stream_tool_arguments", failed)
	}
	refused := findLogEvent(t, logs, "tool_arguments_refused")
	want := map[string]string{
		"request_id": "req-cut", "stage": "item_completed", "tool_name": "edit",
		"argument_bytes": fmt.Sprint(len(arguments)), "first_byte": "{", "valid_utf8": "true",
		"stdlib_object": "false", "duplicate_key": "false", "lone_surrogate": "false",
		"raw_control": "true", "replacement_char": "false", "max_depth": "3",
		"chunks": "2", "chunk_split_rune": "false",
	}
	for field, value := range want {
		if got := fmt.Sprint(refused[field]); got != value {
			t.Errorf("tool_arguments_refused %s = %s, want %s (%#v)", field, got, value, refused)
		}
	}
	for _, entry := range logs.All() {
		line := entry.Message + fmt.Sprint(entry.ContextMap())
		if strings.Contains(line, refusedArgumentText) || strings.Contains(line, "oldText") {
			t.Fatalf("a log line carries argument text: %s", line)
		}
	}
}

// A held cut call failed by a stop that does not cut is logged at the
// terminal, with the facts taken when it was cut.
func TestACutCallFailedAtTheTerminalLogsItsFacts(t *testing.T) {
	chunks := maxTokensToolCutChunks(t, "rejection/240-anthropic-stream-truncated-tool-end-turn-in.json")
	_, _, logs := driveMaxTokensToolCut(t, llmprotocol.OpenAIResponsesV1, chunks)
	refused := findLogEvent(t, logs, "tool_arguments_refused")
	if refused["stage"] != "terminal" || refused["first_byte"] != "{" || fmt.Sprint(refused["stdlib_object"]) != "false" {
		t.Fatalf("tool_arguments_refused = %#v, want a terminal refusal of an object prefix", refused)
	}
}

// A turn that ends any other way logs no tool_arguments_refused.
func TestAWholeToolCallLogsNoArgumentRefusal(t *testing.T) {
	chunks := maxTokensToolCutChunks(t, "stream/040-anthropic-max-tokens-mid-tool-in.json")
	_, _, logs := driveMaxTokensToolCut(t, llmprotocol.OpenAIResponsesV1, chunks)
	for _, entry := range logs.All() {
		if entry.ContextMap()["event"] == "tool_arguments_refused" {
			t.Fatalf("a length stop logged a refusal: %v", entry.ContextMap())
		}
	}
}
