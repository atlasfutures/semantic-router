package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// refusalMarker is argument text no refusal may carry anywhere.
const refusalMarker = "SECRET-ARGUMENT-TEXT"

// anthropicToolStream is an Anthropic stream with one edit tool_use block
// whose arguments arrive as the given input_json_delta chunks, then tail.
func anthropicToolStream(chunks []string, tail string) string {
	var body strings.Builder
	body.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n")
	body.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"edit\",\"input\":{}}}\n\n")
	for _, chunk := range chunks {
		partial, _ := json.Marshal(chunk)
		body.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":" + string(partial) + "}}\n\n")
	}
	body.WriteString("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
	return body.String() + tail
}

// runToolStream pushes body through an Anthropic stream to target and
// returns the client wire and the first error.
func runToolStream(t *testing.T, target llmprotocol.WireFormat, body string) (string, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push([]byte(body))
	final, _, _, finalErr := stream.Finalize(pushErr)
	if pushErr == nil {
		pushErr = finalErr
	}
	return string(bytes.Join(append(frames, final...), nil)), pushErr
}

// requireRefusalFacts requires err to be invalid_stream_tool_arguments with
// facts, and neither the error, its facts, nor the client's error frame to
// carry the arguments' text (the deltas before it carry them, as sent).
func requireRefusalFacts(t *testing.T, wire string, err error) *llmprotocol.ToolArgumentsFacts {
	t.Helper()
	requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
	var facts *llmprotocol.ToolArgumentsFacts
	if !errors.As(err, &facts) || facts == nil {
		t.Fatalf("refusal %v carries no tool-argument facts", err)
	}
	frames := strings.Split(strings.TrimSpace(wire), "\n\n")
	errorFrame := frames[len(frames)-1]
	if !strings.HasPrefix(errorFrame, "event: error") {
		t.Fatalf("the client's last frame is not the refusal: %s", errorFrame)
	}
	for name, text := range map[string]string{
		"error": err.Error(), "facts": fmt.Sprintf("%+v", *facts), "error frame": errorFrame,
	} {
		if strings.Contains(text, refusalMarker) || strings.Contains(text, "oldText") {
			t.Fatalf("the %s carries argument text: %s", name, text)
		}
	}
	return facts
}

// A whole value refused at the block's completion says what it was: here a
// raw newline inside a string, which JSON requires escaped.
func TestARefusedToolCallCarriesItsArgumentFactsAtCompletion(t *testing.T) {
	arguments := "{\"path\":\"" + refusalMarker + ".py\",\"edits\":[{\"oldText\":\"a\nb\",\"newText\":\"c\"}]}"
	chunks := []string{arguments[:5], arguments[5:21], arguments[21:]}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			wire, err := runToolStream(t, target, anthropicToolStream(chunks, anthropicStop("tool_use")))
			facts := requireRefusalFacts(t, wire, err)
			want := llmprotocol.ToolArgumentsFacts{
				Stage: "item_completed", ToolName: "edit", Bytes: len(arguments), FirstByte: "{",
				ValidUTF8: true, StdlibObject: false, JSONErrorOffset: facts.JSONErrorOffset,
				RawControl: true, MaxDepth: 3, Chunks: 3,
			}
			if *facts != want || facts.JSONErrorOffset <= 0 {
				t.Fatalf("facts = %+v, want %+v with a positive error offset", *facts, want)
			}
		})
	}
}

func TestARefusedNonObjectNamesItsFirstByte(t *testing.T) {
	wire, err := runToolStream(t, llmprotocol.OpenAIResponsesV1,
		anthropicToolStream([]string{`["` + refusalMarker + `"]`}, anthropicStop("tool_use")))
	facts := requireRefusalFacts(t, wire, err)
	if facts.FirstByte != "[" || facts.StdlibObject || facts.MaxDepth != 1 || facts.Chunks != 1 {
		t.Fatalf("facts = %+v, want a [ array of depth 1 in one chunk", *facts)
	}
}

// A held cut call that the turn proves malformed keeps the facts taken when
// it was cut: by output after it, or by a stop that does not cut.
func TestAHeldCutCallRefusedLaterCarriesItsArgumentFacts(t *testing.T) {
	prefix := []string{`{"path":"` + refusalMarker, `.py","edits":[{"old`}
	later := "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
	for stage, tail := range map[string]string{"after_cut": later, "terminal": anthropicStop("end_turn")} {
		t.Run(stage, func(t *testing.T) {
			wire, err := runToolStream(t, llmprotocol.OpenAIResponsesV1, anthropicToolStream(prefix, tail))
			facts := requireRefusalFacts(t, wire, err)
			if facts.Stage != stage || facts.Chunks != 2 || facts.FirstByte != "{" || facts.StdlibObject || facts.MaxDepth != 3 {
				t.Fatalf("facts = %+v, want stage %s over a two-chunk object prefix", *facts, stage)
			}
		})
	}
}

func TestEndsInsideRune(t *testing.T) {
	ge := "≥"
	for text, want := range map[string]bool{
		"": false, "abc": false, ge: false, "a" + ge: false,
		ge[:1]: true, ge[:2]: true, "a" + ge[:2]: true, "\xff": false,
	} {
		if got := endsInsideRune(text); got != want {
			t.Errorf("endsInsideRune(%q) = %v, want %v", text, got, want)
		}
	}
}
