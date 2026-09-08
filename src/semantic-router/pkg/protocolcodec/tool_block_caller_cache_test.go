package protocolcodec

import (
	"bytes"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// toolTurnWithCaller is one turn of an ordinary agentic tool loop: the
// assistant called a tool, the client ran it, and both blocks come back in the
// next request. Anthropic's programmatic tool calling names the issuer of a
// tool_use block in caller, and validateAnthropicContentExtensions lists that
// member in an unconditional reject map
// (codec_anthropic_content.go:214), so the whole request is refused with
// unsupported_content_caller on every target.
const toolTurnWithCaller = `{"model":"router-auto","max_tokens":64,"messages":[` +
	`{"role":"user","content":[{"type":"text","text":"read calc.py"}]},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"read_file",` +
	`"input":{"path":"calc.py"},"caller":{"type":"direct"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"def add"}]}]}`

// toolTurnWithCacheBreakpoint is the same turn with a cache breakpoint instead.
// A client marks the last content block of a message it wants cached, so on a
// tool turn that breakpoint lands on a tool block. appendCachelessToolCall and
// appendCachelessToolResult refuse such a block when encoding to Chat
// (codec_openai_chat_encode.go:198-208), so every turn after the first in a
// tool-using session is refused once the client marks the breakpoint.
const toolTurnWithCacheBreakpoint = `{"model":"router-auto","max_tokens":64,"messages":[` +
	`{"role":"user","content":[{"type":"text","text":"read calc.py"}]},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"read_file",` +
	`"input":{"path":"calc.py"},"cache_control":{"type":"ephemeral"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"def add",` +
	`"cache_control":{"type":"ephemeral"}}]}]}`

// TestToolBlockCallerAndCacheBreakpointTravel is the ask.
//
// caller names who issued a tool call and cache_control says what may be
// reused. Neither changes what the model is asked, so a target that cannot
// express one should drop it and count the drop, and let the turn run.
// appendLossy (codec_openai_chat_encode.go:67) is the seam that already does
// this for a request member Chat Completions cannot carry.
func TestToolBlockCallerAndCacheBreakpointTravel(t *testing.T) {
	engine := NewBuiltinEngine()
	turns := []struct {
		name string
		body string
	}{
		{name: "caller", body: toolTurnWithCaller},
		{name: "cache_breakpoint", body: toolTurnWithCacheBreakpoint},
	}
	for _, turn := range turns {
		t.Run(turn.name, func(t *testing.T) {
			result, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(turn.body), nil,
			)
			if err != nil {
				t.Fatalf("the whole tool turn was refused: %v", err)
			}
			if !bytes.Contains(result.Body, []byte(`"tool_calls"`)) {
				t.Fatalf("the translated turn no longer carries its tool call: %s", result.Body)
			}
		})
	}
}

// TestToolBlockCallerAndCacheBreakpointAreRefusedToday pins the present
// behaviour so a change to it shows up in the diff rather than only in the
// test above. caller is refused on both targets because the refusal is raised
// while decoding the request; the cache breakpoint is refused only where the
// target cannot express it.
func TestToolBlockCallerAndCacheBreakpointAreRefusedToday(t *testing.T) {
	engine := NewBuiltinEngine()
	refusals := []struct {
		name   string
		body   string
		target llmprotocol.WireFormat
		code   string
	}{
		{"caller_to_chat", toolTurnWithCaller, llmprotocol.OpenAIChatV1, "unsupported_content_caller"},
		{"caller_to_messages", toolTurnWithCaller, llmprotocol.AnthropicMessagesV1, "unsupported_content_caller"},
		{"cache_breakpoint_to_chat", toolTurnWithCacheBreakpoint, llmprotocol.OpenAIChatV1, "unsupported_cache_directive"},
	}
	for _, refusal := range refusals {
		t.Run(refusal.name, func(t *testing.T) {
			_, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, refusal.target, []byte(refusal.body), nil,
			)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != refusal.code {
				t.Fatalf("TranslateRequest() error = %v, want code %s", err, refusal.code)
			}
		})
	}
}
