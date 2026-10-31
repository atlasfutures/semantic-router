package extproc

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

// A GPT reply with reasoning reaches a Messages client as a marker-signed
// thinking block, a redacted_thinking block holding OpenAI's encrypted
// reasoning, and the tool call (testdata/delta_gpt_echo: a dev capture of a
// test session, 2026-10-31). The staged copy, rebuilt from the neutral stream,
// must digest as a client's verbatim echo of those frames.
func TestDeltaStagesAGPTReasoningReplyAsItsEcho(t *testing.T) {
	raw, err := os.ReadFile("testdata/delta_gpt_echo/luna-reasoning-tool.sse")
	if err != nil {
		t.Fatal(err)
	}
	engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{})
	if err != nil {
		t.Fatal(err)
	}
	state := &semanticResponseStreamState{usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, items: map[int]*semanticStreamItem{}}
	_, events, _, err := stream.Push(raw)
	if err != nil {
		t.Fatal(err)
	}
	state.observe(events)
	_, events, _, _ = stream.Finalize(nil)
	state.observe(events)
	response, err := state.response()
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, *response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	if policyReplyThinks(policyFormatAnthropic, encoded.Body) {
		t.Fatal("a GPT reasoning reply must be appended, not held back as thinking")
	}
	staged, ok := policyAssistantOutput(policyFormatAnthropic, encoded.Body)
	if !ok || len(staged) != 1 {
		t.Fatalf("staged %d messages, ok=%v", len(staged), ok)
	}
	// The client's assembly (Claude Code accumulates blocks from the frames it receives).
	echo := map[string]any{"role": "assistant", "content": assembleSSE(t, raw)}
	echoRaw, _ := json.Marshal(echo)
	if policyDeltaDigest([]json.RawMessage{echoRaw}, 1) != policyDeltaDigest(staged, 1) {
		t.Fatalf("staged %s\necho   %s", policyDeltaCanonical(staged[0]), policyDeltaCanonical(echoRaw))
	}
}

func assembleSSE(t *testing.T, raw []byte) []any {
	type block = map[string]any
	blocks := map[int]block{}
	partial := map[int]string{}
	order := []int{}
	for _, line := range splitLines(raw) {
		if len(line) < 5 || string(line[:5]) != "data:" {
			continue
		}
		var ev map[string]any
		if json.Unmarshal(line[5:], &ev) != nil {
			continue
		}
		switch ev["type"] {
		case "content_block_start":
			i := int(ev["index"].(float64))
			blocks[i] = ev["content_block"].(map[string]any)
			order = append(order, i)
		case "content_block_delta":
			i := int(ev["index"].(float64))
			d := ev["delta"].(map[string]any)
			b := blocks[i]
			switch d["type"] {
			case "text_delta":
				b["text"] = str(b["text"]) + d["text"].(string)
			case "thinking_delta":
				b["thinking"] = str(b["thinking"]) + d["thinking"].(string)
			case "signature_delta":
				b["signature"] = str(b["signature"]) + d["signature"].(string)
			case "input_json_delta":
				partial[i] += d["partial_json"].(string)
			}
		}
	}
	out := []any{}
	for _, i := range order {
		b := blocks[i]
		if b["type"] == "tool_use" && partial[i] != "" {
			var input any
			_ = json.Unmarshal([]byte(partial[i]), &input)
			b["input"] = input
		}
		out = append(out, b)
	}
	return out
}

func str(v any) string { s, _ := v.(string); return s }

func splitLines(raw []byte) [][]byte {
	var out [][]byte
	start := 0
	for i, c := range raw {
		if c == '\n' {
			out = append(out, raw[start:i])
			start = i + 1
		}
	}
	return out
}
