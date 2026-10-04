package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Responses client sends OpenRouter's top-level cache_control on every
// turn. Routed to a Claude worker, the request is translated to Messages,
// which caches only at explicit breakpoints, so without a placed breakpoint
// the dispatched turn asks for no caching and a growing conversation is
// billed uncached every time.

const responsesAutoCacheTurn = `{"model":"anthropic/claude-opus","instructions":"You are a coding agent.",` +
	`"cache_control":{"type":"ephemeral","ttl":"1h"},` +
	`"tools":[{"type":"function","name":"shell","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}],` +
	`"input":[` +
	`{"type":"message","role":"user","content":[{"type":"input_text","text":"List the files."}]},` +
	`{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{\"cmd\":\"ls\"}"},` +
	`{"type":"function_call_output","call_id":"call_1","output":"README.md\nmain.go"},` +
	`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"There are two files."}]},` +
	`{"type":"message","role":"user","content":[{"type":"input_text","text":"Open main.go."}]}%s]}`

// The tail a Codex tool turn ends with: the call and its output.
const responsesAutoCacheToolTail = `,{"type":"function_call","call_id":"call_2","name":"shell","arguments":"{\"cmd\":\"cat main.go\"}"},` +
	`{"type":"function_call_output","call_id":"call_2","output":"package main"}`

type placedBreakpoint struct {
	path      string
	directive anthropicCacheControlWire
}

func TestResponsesAutoCacheBecomesOneMessagesBreakpoint(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantPath string
	}{
		{name: "turn ends with user text", body: fmt.Sprintf(responsesAutoCacheTurn, ""), wantPath: "messages[4].content[0]"},
		{name: "turn ends with a tool result", body: fmt.Sprintf(responsesAutoCacheTurn, responsesAutoCacheToolTail), wantPath: "messages[6].content[0]"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := routeRequest(t, llmprotocol.OpenAIResponsesV1, test.body, llmprotocol.AnthropicMessagesV1)
			breakpoints := anthropicBreakpoints(t, encoded.Body)
			if len(breakpoints) != 1 {
				t.Fatalf("breakpoints = %+v, want exactly one", breakpoints)
			}
			got := breakpoints[0]
			if got.path != test.wantPath {
				t.Fatalf("breakpoint on %s, want the last block of the last message (%s)", got.path, test.wantPath)
			}
			if got.directive.Type != "ephemeral" || got.directive.TTL != "1h" {
				t.Fatalf("placed directive = %+v, want the client's ephemeral 1h", got.directive)
			}
			if bytes.Contains(topLevelMember(t, encoded.Body, "cache_control"), []byte("ephemeral")) {
				t.Fatalf("a translated request sent a top-level cache_control: %s", encoded.Body)
			}
			if !hasDiagnostic(encoded.Diagnostics, "cache_control", llmprotocol.DiagnosticApproximated) {
				t.Fatalf("the placement was not recorded: %+v", encoded.Diagnostics)
			}
		})
	}

	t.Run("control: no directive, no breakpoint", func(t *testing.T) {
		body, err := sjson.Delete(fmt.Sprintf(responsesAutoCacheTurn, ""), "cache_control")
		if err != nil {
			t.Fatal(err)
		}
		encoded := routeRequest(t, llmprotocol.OpenAIResponsesV1, body, llmprotocol.AnthropicMessagesV1)
		if breakpoints := anthropicBreakpoints(t, encoded.Body); len(breakpoints) != 0 {
			t.Fatalf("a request without the directive got breakpoints %+v", breakpoints)
		}
		if hasDiagnostic(encoded.Diagnostics, "cache_control", "") {
			t.Fatalf("a request without the directive recorded %+v", encoded.Diagnostics)
		}
	})
}

func TestResponsesAutoCacheOnOtherTargets(t *testing.T) {
	body := fmt.Sprintf(responsesAutoCacheTurn, "")
	responses := routeRequest(t, llmprotocol.OpenAIResponsesV1, body, llmprotocol.OpenAIResponsesV1)
	if got := string(topLevelMember(t, responses.Body, "cache_control")); got != `{"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("Responses target cache_control = %s", got)
	}
	chat := routeRequest(t, llmprotocol.OpenAIResponsesV1, body, llmprotocol.OpenAIChatV1)
	if bytes.Contains(chat.Body, []byte("cache_control")) {
		t.Fatalf("Chat target carries a cache_control: %s", chat.Body)
	}
	if !hasDiagnostic(chat.Diagnostics, "cache_control", llmprotocol.DiagnosticDropped) {
		t.Fatalf("the Chat drop was not counted: %+v", chat.Diagnostics)
	}
}

// Anthropic's API defines top-level cache_control as automatic caching. The
// ingress used to refuse it.
const anthropicAutoCacheTurn = `{"model":"claude-opus","max_tokens":1024,"cache_control":{"type":"ephemeral"},` +
	`"system":[{"type":"text","text":"You are a coding agent."}],` +
	`"messages":[{"role":"user","content":"List the files."},` +
	`{"role":"assistant","content":[{"type":"text","text":"There are two files."}]},` +
	`{"role":"user","content":"Open main.go."}]}`

func TestAnthropicAutoCacheIsAcceptedAndCarried(t *testing.T) {
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(anthropicAutoCacheTurn))
	if err != nil {
		t.Fatalf("top-level cache_control was refused: %v", err)
	}
	if request.AutoCache == nil || request.AutoCache.Type != "ephemeral" {
		t.Fatalf("decoded automatic-cache directive = %+v", request.AutoCache)
	}

	messages := routeRequest(t, llmprotocol.AnthropicMessagesV1, anthropicAutoCacheTurn, llmprotocol.AnthropicMessagesV1)
	want, err := sjson.SetBytes([]byte(anthropicAutoCacheTurn), "model", "routed-model")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(messages.Body, want) {
		t.Fatalf("a same-format turn did not replay the client's bytes\n got: %s\nwant: %s", messages.Body, want)
	}
	if breakpoints := anthropicBreakpoints(t, messages.Body); len(breakpoints) != 0 {
		t.Fatalf("a Messages client's own member also got a placed breakpoint: %+v", breakpoints)
	}

	responses := routeRequest(t, llmprotocol.AnthropicMessagesV1, anthropicAutoCacheTurn, llmprotocol.OpenAIResponsesV1)
	if got := string(topLevelMember(t, responses.Body, "cache_control")); got != `{"type":"ephemeral"}` {
		t.Fatalf("Responses target cache_control = %s", got)
	}
	chat := routeRequest(t, llmprotocol.AnthropicMessagesV1, anthropicAutoCacheTurn, llmprotocol.OpenAIChatV1)
	if bytes.Contains(chat.Body, []byte("cache_control")) {
		t.Fatalf("Chat target carries a cache_control: %s", chat.Body)
	}
	if !hasDiagnostic(chat.Diagnostics, "cache_control", llmprotocol.DiagnosticDropped) {
		t.Fatalf("the Chat drop was not counted: %+v", chat.Diagnostics)
	}
}

func TestMalformedAutoCacheIsRefused(t *testing.T) {
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		code   string
	}{
		{"Messages type", llmprotocol.AnthropicMessagesV1, mustSet(t, anthropicAutoCacheTurn, "cache_control", map[string]any{"type": "permanent"}), "invalid_cache_directive"},
		{"Messages ttl", llmprotocol.AnthropicMessagesV1, mustSet(t, anthropicAutoCacheTurn, "cache_control", map[string]any{"type": "ephemeral", "ttl": "7d"}), "invalid_cache_directive"},
		{"Messages shape", llmprotocol.AnthropicMessagesV1, mustSet(t, anthropicAutoCacheTurn, "cache_control", true), ""},
		{"Responses type", llmprotocol.OpenAIResponsesV1, mustSet(t, fmt.Sprintf(responsesAutoCacheTurn, ""), "cache_control", map[string]any{"type": "permanent"}), "invalid_cache_directive"},
		{"Responses shape", llmprotocol.OpenAIResponsesV1, mustSet(t, fmt.Sprintf(responsesAutoCacheTurn, ""), "cache_control", "ephemeral"), ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequest(test.format, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest {
				t.Fatalf("err = %v, want a typed invalid request", err)
			}
			if test.code != "" && protocolError.Code != test.code {
				t.Fatalf("code = %q, want %q", protocolError.Code, test.code)
			}
		})
	}
}

// The limit and the block choice are rules of the Messages encoder, reached
// by any non-Messages source that carries the directive; they are pinned on
// the neutral request because no ingress today pairs the directive with
// client breakpoints.
func TestAutoCacheBreakpointRespectsTheMessagesLimit(t *testing.T) {
	cached := &llmprotocol.CacheDirective{Type: "ephemeral"}
	request := autoCacheRequest(
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "Context A", Cache: cached},
			{Kind: llmprotocol.ContentText, Text: "Context B", Cache: cached},
		}},
		llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "Read both."},
		}},
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "Summarize them."},
		}},
	)
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentText, Text: "You are a reviewer.", Cache: cached},
	}}}
	request.Tools = []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Cache: cached}}

	body, diagnostics := encodeAnthropic(t, request)
	if got := anthropicBreakpoints(t, body); len(got) != 4 {
		t.Fatalf("breakpoints = %+v, want the client's four and no more", got)
	}
	if !hasDiagnostic(diagnostics, "cache_control", llmprotocol.DiagnosticDropped) {
		t.Fatalf("the dropped directive was not counted: %+v", diagnostics)
	}

	request.Tools[0].Cache = nil
	body, _ = encodeAnthropic(t, request)
	got := anthropicBreakpoints(t, body)
	if len(got) != 4 || got[len(got)-1].path != "messages[2].content[0]" {
		t.Fatalf("with three client breakpoints, breakpoints = %+v, want one placed on messages[2].content[0]", got)
	}
}

func TestAutoCacheBreakpointSkipsBlocksThatCannotCarryOne(t *testing.T) {
	thinking := llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "The tests passed.", Signature: "c2ln"}
	call := llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "toolu_1", Name: "shell", Arguments: `{"cmd":"go test ./..."}`}}
	result := llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "toolu_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ok"}}}}
	user := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Run the tests."}}}

	tests := []struct {
		name     string
		messages []llmprotocol.Message
		want     string
	}{
		{
			name: "final message ends with thinking",
			messages: []llmprotocol.Message{user,
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{call}},
				{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{result}},
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "All green."}, thinking}},
			},
			want: "messages[3].content[0]",
		},
		{
			name: "final message is only thinking",
			messages: []llmprotocol.Message{user,
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{call}},
				{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{result}},
				{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{thinking}},
			},
			want: "messages[2].content[0]",
		},
		{
			name: "final block is empty text",
			messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "Run the tests."}, {Kind: llmprotocol.ContentText},
			}}},
			want: "messages[0].content[0]",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body, _ := encodeAnthropic(t, autoCacheRequest(test.messages...))
			got := anthropicBreakpoints(t, body)
			if len(got) != 1 || got[0].path != test.want {
				t.Fatalf("breakpoints = %+v, want one on %s", got, test.want)
			}
		})
	}

	t.Run("the chosen block already holds one", func(t *testing.T) {
		held := user
		held.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Run the tests.", Cache: &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"}}}
		body, diagnostics := encodeAnthropic(t, autoCacheRequest(held))
		got := anthropicBreakpoints(t, body)
		if len(got) != 1 || got[0].directive.TTL != "5m" {
			t.Fatalf("breakpoints = %+v, want the client's own one kept", got)
		}
		if hasDiagnostic(diagnostics, "cache_control", "") {
			t.Fatalf("an honoured directive recorded %+v", diagnostics)
		}
	})
}

// Anthropic caches the prefix in the order tools, system, messages. A
// conversation with no message block that can hold a breakpoint -- here an
// assistant turn of signed thinking only -- falls back to the last system
// block, which caches tools and system, and then to the last tool.
func TestAutoCacheBreakpointFallsBackToSystemThenTools(t *testing.T) {
	thinkingOnly := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentReasoning, Text: "Plan the review.", Signature: "c2ln"},
	}}
	system := func(cache *llmprotocol.CacheDirective) []llmprotocol.InstructionBlock {
		return []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "You are a reviewer."},
			{Kind: llmprotocol.ContentText, Text: "Answer in one paragraph.", Cache: cache},
		}}}
	}
	tools := func(cache *llmprotocol.CacheDirective) []llmprotocol.Tool {
		return []llmprotocol.Tool{
			{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "shell", InputSchema: json.RawMessage(`{"type":"object"}`), Cache: cache},
		}
	}
	held := &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"}

	t.Run("system block", func(t *testing.T) {
		request := autoCacheRequest(thinkingOnly)
		request.Instructions, request.Tools = system(nil), tools(nil)
		body, diagnostics := encodeAnthropic(t, request)
		got := anthropicBreakpoints(t, body)
		if len(got) != 1 || got[0].path != "system.content[1]" || got[0].directive.TTL != "1h" {
			t.Fatalf("breakpoints = %+v, want one on the last system block", got)
		}
		if !hasDiagnostic(diagnostics, "cache_control", llmprotocol.DiagnosticApproximated) {
			t.Fatalf("the placement was not recorded: %+v", diagnostics)
		}
	})
	t.Run("tool", func(t *testing.T) {
		request := autoCacheRequest(thinkingOnly)
		request.Tools = tools(nil)
		body, _ := encodeAnthropic(t, request)
		got := anthropicBreakpoints(t, body)
		if len(got) != 1 || got[0].path != "tools[1]" {
			t.Fatalf("breakpoints = %+v, want one on the last tool", got)
		}
	})
	t.Run("nowhere", func(t *testing.T) {
		body, diagnostics := encodeAnthropic(t, autoCacheRequest(thinkingOnly))
		if got := anthropicBreakpoints(t, body); len(got) != 0 {
			t.Fatalf("breakpoints = %+v, want none", got)
		}
		if !hasDiagnostic(diagnostics, "cache_control", llmprotocol.DiagnosticDropped) {
			t.Fatalf("the dropped directive was not counted: %+v", diagnostics)
		}
	})
	t.Run("system block already holds one", func(t *testing.T) {
		request := autoCacheRequest(thinkingOnly)
		request.Instructions = system(held)
		body, diagnostics := encodeAnthropic(t, request)
		got := anthropicBreakpoints(t, body)
		if len(got) != 1 || got[0].directive.TTL != "5m" {
			t.Fatalf("breakpoints = %+v, want the client's own one kept", got)
		}
		if hasDiagnostic(diagnostics, "cache_control", "") {
			t.Fatalf("an honoured directive recorded %+v", diagnostics)
		}
	})
	t.Run("tool already holds one", func(t *testing.T) {
		request := autoCacheRequest(thinkingOnly)
		request.Tools = tools(held)
		body, diagnostics := encodeAnthropic(t, request)
		got := anthropicBreakpoints(t, body)
		if len(got) != 1 || got[0].directive.TTL != "5m" {
			t.Fatalf("breakpoints = %+v, want the client's own one kept", got)
		}
		if hasDiagnostic(diagnostics, "cache_control", "") {
			t.Fatalf("an honoured directive recorded %+v", diagnostics)
		}
	})
	t.Run("limit reached before the system block", func(t *testing.T) {
		request := autoCacheRequest(thinkingOnly)
		request.Instructions = system(nil)
		request.Tools = nil
		for _, name := range []string{"a", "b", "c", "d"} {
			request.Tools = append(request.Tools, llmprotocol.Tool{Name: name, InputSchema: json.RawMessage(`{"type":"object"}`), Cache: held})
		}
		body, diagnostics := encodeAnthropic(t, request)
		if got := anthropicBreakpoints(t, body); len(got) != 4 {
			t.Fatalf("breakpoints = %+v, want the client's four and no more", got)
		}
		if !hasDiagnostic(diagnostics, "cache_control", llmprotocol.DiagnosticDropped) {
			t.Fatalf("the dropped directive was not counted: %+v", diagnostics)
		}
	})
}

func autoCacheRequest(messages ...llmprotocol.Message) llmprotocol.Request {
	return llmprotocol.Request{
		Generation: 1, Model: "claude-opus", Messages: messages,
		Sampling:  llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(1024)},
		AutoCache: &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "1h"},
		Trusted:   llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIResponsesV1},
	}
}

func encodeAnthropic(t *testing.T, request llmprotocol.Request) ([]byte, llmprotocol.Diagnostics) {
	t.Helper()
	encoded, err := NewBuiltinEngine().EncodeRequest(llmprotocol.AnthropicMessagesV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	return encoded.Body, encoded.Diagnostics
}

func routeRequest(t *testing.T, source llmprotocol.WireFormat, body string, target llmprotocol.WireFormat) RequestResult {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(source, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(target, request, envelope)
	if err != nil {
		t.Fatalf("encode to %s: %v", target, err)
	}
	return encoded
}

// anthropicBreakpoints lists every cache_control a Messages body states on a
// tool, a system block or a message block, in body order.
func anthropicBreakpoints(t *testing.T, body []byte) []placedBreakpoint {
	t.Helper()
	var wire struct {
		Tools    []map[string]json.RawMessage `json:"tools"`
		System   json.RawMessage              `json:"system"`
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	var found []placedBreakpoint
	add := func(path string, raw json.RawMessage) {
		if len(raw) == 0 {
			return
		}
		var directive anthropicCacheControlWire
		if err := json.Unmarshal(raw, &directive); err != nil {
			t.Fatal(err)
		}
		found = append(found, placedBreakpoint{path: path, directive: directive})
	}
	for index, tool := range wire.Tools {
		add(fmt.Sprintf("tools[%d]", index), tool["cache_control"])
	}
	blocks := func(prefix string, raw json.RawMessage) {
		var list []map[string]json.RawMessage
		if json.Unmarshal(raw, &list) != nil {
			return // string content holds no breakpoint
		}
		for index, block := range list {
			add(fmt.Sprintf("%s.content[%d]", prefix, index), block["cache_control"])
		}
	}
	blocks("system", wire.System)
	for index, message := range wire.Messages {
		blocks(fmt.Sprintf("messages[%d]", index), message.Content)
	}
	return found
}

func topLevelMember(t *testing.T, body []byte, name string) json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatal(err)
	}
	return object[name]
}

func hasDiagnostic(diagnostics llmprotocol.Diagnostics, field string, action llmprotocol.DiagnosticAction) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && (action == "" || diagnostic.Action == action) {
			return true
		}
	}
	return false
}

func mustSet(t *testing.T, body, path string, value any) string {
	t.Helper()
	updated, err := sjson.Set(body, path, value)
	if err != nil {
		t.Fatal(err)
	}
	return updated
}
