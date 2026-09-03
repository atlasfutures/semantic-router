package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code's request shape is not one contract but a set of beta opt-ins,
// and which ones it negotiates depends on how it authenticated. Through the
// rayline proxy it believes it holds an OAuth subscription session and turns
// on the full set, so the body carries members the direct path never sends.
// Each member below is recorded with the beta header that introduces it, taken
// from the anthropic-beta header of the captures made on 2026-09-03 against
// the dev cell with claude-cli 2.1.259:
//
//	cache_control.scope          prompt-caching-scope-2026-01-05    (undocumented)
//	cache_control.ttl            extended-cache-ttl-2025-04-11
//	tools[].eager_input_streaming  advanced-tool-use-2025-11-20 as sent; Anthropic
//	                             documents it as generally available, superseding
//	                             fine-grained-tool-streaming-2025-05-14
//	diagnostics                  cache-diagnosis-2026-04-07
//	a system-role message        mid-conversation-system-2026-04-07
//	output_config.effort         effort-2025-11-24
//	thinking.type adaptive       interleaved-thinking-2025-05-14
//	context_management           context-management-2025-06-27
//
// The bodies below reproduce the field paths of the captured requests. They
// are the regression fixture for every member: the cell answered 400 to the
// proxy-mode body and 400 to the direct-mode body, and both must route.
const (
	claudeCodeProxyModeBody = `{` +
		`"model":"rayline-router",` +
		`"system":[` +
		`{"type":"text","text":"billing header"},` +
		`{"type":"text","text":"You are a Claude agent.",` +
		`"cache_control":{"type":"ephemeral","ttl":"1h","scope":"global"}},` +
		`{"type":"text","text":"More instructions",` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"}}],` +
		`"messages":[` +
		`{"role":"user","content":"hello"},` +
		`{"role":"system","content":[{"type":"text","text":"mid-conversation",` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"}}]}],` +
		`"tools":[{"name":"Agent","description":"Launch an agent",` +
		`"input_schema":{"type":"object"},"eager_input_streaming":true}],` +
		`"metadata":{"user_id":"u"},` +
		`"max_tokens":32000,` +
		`"output_config":{"effort":"high"},` +
		`"diagnostics":{"previous_message_id":null},` +
		`"stream":true}`

	claudeCodeDirectModeBody = `{` +
		`"model":"rayline-router",` +
		`"system":[{"type":"text","text":"You are a Claude agent.",` +
		`"cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hello"}],` +
		`"metadata":{"user_id":"u"},` +
		`"max_tokens":32000,` +
		`"thinking":{"type":"adaptive","display":"omitted"},` +
		`"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},` +
		`"output_config":{"effort":"high"},` +
		`"stream":true}`
)

func TestClaudeCodeBodiesRouteToEveryLeg(t *testing.T) {
	engine := NewBuiltinEngine()
	bodies := map[string]string{
		"proxy mode":  claudeCodeProxyModeBody,
		"direct mode": claudeCodeDirectModeBody,
	}
	targets := []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
	}
	for name, body := range bodies {
		for _, target := range targets {
			t.Run(name+" to "+string(target), func(t *testing.T) {
				if _, err := engine.TranslateRequest(
					llmprotocol.AnthropicMessagesV1, target, []byte(body), nil,
				); err != nil {
					t.Fatalf("a body Claude Code sends by default was refused: %v", err)
				}
			})
		}
	}
}

// The scope reaches an Anthropic arm unchanged. The Router does not know what
// it means, so rewriting or dropping it would move a cache entry on the
// caller's behalf.
func TestProxyModeCacheScopeReachesAnAnthropicArm(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(claudeCodeProxyModeBody), func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if scope := systemCacheScopes(t, result.Body); scope != "global" {
		t.Fatalf("re-encoded cache scopes = %q, want global", scope)
	}
}

// systemCacheScopes joins the scope of every re-encoded system breakpoint, so
// a scope that moved to the wrong block fails as loudly as one that vanished.
func systemCacheScopes(t *testing.T, body []byte) string {
	t.Helper()
	var wire anthropicRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	var blocks []anthropicContentWire
	if err := json.Unmarshal(wire.System, &blocks); err != nil {
		t.Fatal(err)
	}
	scopes := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.CacheControl != nil && block.CacheControl.Scope != "" {
			scopes = append(scopes, block.CacheControl.Scope)
		}
	}
	return strings.Join(scopes, ",")
}
