package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Every tool-using turn after the first was answered 500 by the dev cell on
// 2026-09-04, and one shape was answered 400. Both come from the same place:
// the Anthropic tool blocks Claude Code sends carry two members the neutral
// contract refused.
//
//	cache_control on a tool_use / tool_result block
//	    Claude Code marks the last content block of a cached message. When that
//	    message ends in a tool call or its result, the breakpoint lands on the
//	    tool block. The Chat encoder answered
//	    "Chat Completions cannot attach cache_control to tool call content",
//	    which the ext_proc filter turned into a 500 and the proxy hid behind an
//	    OpenRouter fallback, so the cell served only the first turn of any
//	    tool-using session.
//	caller on a tool_use block
//	    Anthropic's programmatic tool calling names who issued the call. The
//	    client emits it as {"type":"direct"}; claude-cli 2.1.260 builds the
//	    literal `caller:{type:"direct"}` for its plugin and batch tool paths and
//	    strips it again when the negotiated API cannot take it. Ingress refused
//	    it outright with unsupported_content_caller, a 400 the proxy has no
//	    fallback for, so it reached the user as an API error.
//
// A cache breakpoint is a hint about reuse and caller names the issuer; neither
// changes what the model is asked. Chat Completions can express neither, so the
// Chat encoder drops them and counts the drop. An Anthropic arm can express
// both, so they travel unchanged.
const (
	claudeCodeToolUseCallerBody = `{` +
		`"model":"rayline-router","max_tokens":64,` +
		`"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"read calc.py"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use",` +
		`"id":"toolu_01a60eedb6394ca68315","name":"Read",` +
		`"input":{"file_path":"/tmp/calc.py"},"caller":{"type":"direct"}}]},` +
		`{"role":"user","content":[{"type":"tool_result",` +
		`"tool_use_id":"toolu_01a60eedb6394ca68315","content":"def add"}]}],` +
		`"stream":true}`

	claudeCodeToolCacheBreakpointBody = `{` +
		`"model":"rayline-router","max_tokens":64,` +
		`"tools":[{"name":"Read","description":"Read a file",` +
		`"input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"read calc.py"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use",` +
		`"id":"toolu_01a60eedb6394ca68315","name":"Read",` +
		`"input":{"file_path":"/tmp/calc.py"},` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"}}]},` +
		`{"role":"user","content":[{"type":"tool_result",` +
		`"tool_use_id":"toolu_01a60eedb6394ca68315","content":"def add",` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"}}]}],` +
		`"stream":true}`
)

// The whole captured turn routes, not just the two members in isolation. The
// fixture is a claude-cli 2.1.260 turn-2 body: the client asked for a file, the
// tool ran, and the tool_use and its tool_result came back in the next request.
// Text and schemas are trimmed; the block structure, ids and beta members are
// the capture's own.
func TestClaudeCodeToolTurnCaptureRoutesToEveryLeg(t *testing.T) {
	engine := NewBuiltinEngine()
	body, err := os.ReadFile(filepath.Join("testdata", "claude-code", "tool-turn-request.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			if _, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, target, body, nil,
			); err != nil {
				t.Fatalf("a captured Claude Code tool turn was refused: %v", err)
			}
		})
	}
}

// Ingress accepts caller, and the field survives a same-format routing hop.
func TestToolUseCallerIsAcceptedAndReachesAnAnthropicArm(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(claudeCodeToolUseCallerBody), func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("a tool_use block carrying caller was refused: %v", err)
	}
	block := anthropicToolUseBlock(t, result.Body)
	if string(bytes.TrimSpace(block.Caller)) != `{"type":"direct"}` {
		t.Fatalf("re-encoded tool_use caller = %s, want {\"type\":\"direct\"}", block.Caller)
	}
}

// Chat Completions has no field for it, so the turn still runs and the drop is
// counted under content.caller.
func TestToolUseCallerIsDroppedAndCountedForChat(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		[]byte(claudeCodeToolUseCallerBody), nil,
	)
	if err != nil {
		t.Fatalf("a tool_use block carrying caller was refused for a Chat arm: %v", err)
	}
	if bytes.Contains(result.Body, []byte(`"caller"`)) {
		t.Fatalf("the Chat body still carries caller: %s", result.Body)
	}
	requireDroppedField(t, result.Diagnostics, "content.caller")
}

// caller belongs to tool_use alone. The canonical-field policy still refuses it
// on a block that has no such member, and still refuses an unknown sibling.
func TestCallerOnAnotherBlockAndUnknownSiblingsStayRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	bodies := map[string]string{
		"caller on a text block": `{"model":"m","max_tokens":16,"messages":[` +
			`{"role":"user","content":[{"type":"text","text":"hi",` +
			`"caller":{"type":"direct"}}]}]}`,
		"unknown sibling on a tool_use block": `{"model":"m","max_tokens":16,"messages":[` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read",` +
			`"input":{},"caller":{"type":"direct"},"invented_field":true}]}]}`,
	}
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			if _, _, _, err := engine.DecodeRequest(
				llmprotocol.AnthropicMessagesV1, []byte(body),
			); err == nil {
				t.Fatal("a non-canonical block was accepted")
			}
		})
	}
}

// The 500 the dev cell answered on every follow-up turn.
func TestToolBlockCacheBreakpointIsDroppedAndCountedForChat(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		[]byte(claudeCodeToolCacheBreakpointBody), nil,
	)
	if err != nil {
		t.Fatalf("a tool turn carrying a cache breakpoint was refused: %v", err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	for _, message := range wire.Messages {
		if bytes.Contains(message.Content, []byte(`"cache_control"`)) {
			t.Fatalf("a Chat message still carries cache_control: %s", message.Content)
		}
	}
	requireDroppedField(t, result.Diagnostics, "content.cache_control")
}

// A tools[] breakpoint is a different member on a different object. It has
// always been carried, and the drop above must not start swallowing it.
func TestToolDeclarationCacheBreakpointStillTravelsToChat(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		[]byte(claudeCodeToolCacheBreakpointBody), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Tools) != 1 || wire.Tools[0].CacheControl == nil {
		t.Fatalf("the tool declaration lost its cache breakpoint: %s", result.Body)
	}
}

// An Anthropic arm can hold the breakpoint, so routing must not erase it.
func TestToolBlockCacheBreakpointReachesAnAnthropicArm(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(claudeCodeToolCacheBreakpointBody), func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("a tool turn carrying a cache breakpoint was refused: %v", err)
	}
	kept := anthropicCacheControlByBlockType(t, result.Body)
	for _, typeName := range []string{"tool_use", "tool_result"} {
		cache, present := kept[typeName]
		if !present || cache == nil {
			t.Fatalf("routing erased the %s cache breakpoint: %s", typeName, result.Body)
		}
		if cache.Type != "ephemeral" || cache.TTL != "1h" {
			t.Fatalf("%s cache breakpoint = %+v", typeName, cache)
		}
	}
}

func requireDroppedField(t *testing.T, diagnostics llmprotocol.Diagnostics, field string) {
	t.Helper()
	names := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		names = append(names, string(diagnostic.Action)+" "+diagnostic.Field)
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped {
			return
		}
	}
	t.Fatalf("no dropped diagnostic for %q; recorded: %s", field, strings.Join(names, ", "))
}

func anthropicToolUseBlock(t *testing.T, body []byte) anthropicContentWire {
	t.Helper()
	for _, block := range anthropicRequestBlocks(t, body) {
		if block.Type == "tool_use" {
			return block
		}
	}
	t.Fatalf("the re-encoded body holds no tool_use block: %s", body)
	return anthropicContentWire{}
}

func anthropicCacheControlByBlockType(t *testing.T, body []byte) map[string]*anthropicCacheControlWire {
	t.Helper()
	found := make(map[string]*anthropicCacheControlWire)
	for _, block := range anthropicRequestBlocks(t, body) {
		found[block.Type] = block.CacheControl
	}
	return found
}

func anthropicRequestBlocks(t *testing.T, body []byte) []anthropicContentWire {
	t.Helper()
	var wire anthropicRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	var blocks []anthropicContentWire
	for _, message := range wire.Messages {
		var decoded []anthropicContentWire
		if err := json.Unmarshal(message.Content, &decoded); err != nil {
			continue
		}
		blocks = append(blocks, decoded...)
	}
	return blocks
}
