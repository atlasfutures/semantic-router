package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// claudeAutoCacheCase is one request through the router's ingress, provider
// dispatch and encode, to a single worker of the given wire format.
type claudeAutoCacheCase struct {
	source llmprotocol.WireFormat
	target llmprotocol.WireFormat
	claude bool
	body   string
}

func dispatchClaudeAutoCache(t *testing.T, test claudeAutoCacheCase) ([]byte, *RequestContext) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(test.target)
	if test.claude {
		params := router.Config.ModelConfig[logicalModel]
		params.Publisher = "anthropic"
		router.Config.ModelConfig[logicalModel] = params
	}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: test.source,
		RequestID: "claude-auto-cache", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(test.body), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused %s: %+v", test.body, ctx.ImmediateProtocolError)
	}
	if _, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return body, ctx
}

// messagesBreakpoints lists where a dispatched Messages body holds a
// cache_control: "tools[i]", "system[i]" or "messages[i][j]", with its
// value, and reports a top-level member as "top".
func messagesBreakpoints(t *testing.T, body []byte) map[string]string {
	t.Helper()
	var wire struct {
		CacheControl json.RawMessage `json:"cache_control"`
		System       json.RawMessage `json:"system"`
		Tools        []map[string]json.RawMessage
		Messages     []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("dispatched %s: %v", body, err)
	}
	found := map[string]string{}
	if len(wire.CacheControl) > 0 {
		found["top"] = string(wire.CacheControl)
	}
	blocks := func(prefix string, raw json.RawMessage) {
		var list []map[string]json.RawMessage
		if json.Unmarshal(raw, &list) != nil {
			return
		}
		for index, block := range list {
			if value := block["cache_control"]; len(value) > 0 {
				found[prefix+"["+strconv.Itoa(index)+"]"] = string(value)
			}
		}
	}
	for index, tool := range wire.Tools {
		if value := tool["cache_control"]; len(value) > 0 {
			found["tools["+strconv.Itoa(index)+"]"] = string(value)
		}
	}
	blocks("system", wire.System)
	for index, message := range wire.Messages {
		blocks("messages["+strconv.Itoa(index)+"]", message.Content)
	}
	return found
}

func hasAutoCacheDiagnostic(ctx *RequestContext, reason string) bool {
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == "cache_control" && diagnostic.Action == llmprotocol.DiagnosticGenerated &&
			diagnostic.Reason == reason {
			return true
		}
	}
	return false
}

const (
	autoCacheConversation = `"input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"List the files."}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"There are two files."}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"Open main.go."}]}]`
	responsesNoDirective = `{"model":"m","instructions":"You are a coding agent.",` + autoCacheConversation + `}`
	responsesCacheKey    = `{"model":"m","instructions":"You are a coding agent.",` +
		`"prompt_cache_key":"session-1",` + autoCacheConversation + `}`
	responsesTopLevel1h = `{"model":"m","instructions":"You are a coding agent.",` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"},` + autoCacheConversation + `}`
)

// An OpenAI-shaped client relies on automatic prefix caching and states no
// breakpoint. Routed to a Claude worker over Messages, its turn carries the
// one breakpoint automatic caching would place, on the last block, whether
// or not it named a cache shard; the diagnostic says which.
func TestClaudeAutoCacheForAnOpenAIShapedClient(t *testing.T) {
	for name, test := range map[string]struct {
		body   string
		reason string
	}{
		"no directive":     {responsesNoDirective, "automatic_cache_default"},
		"prompt_cache_key": {responsesCacheKey, "automatic_cache_prompt_cache_key"},
	} {
		t.Run(name, func(t *testing.T) {
			body, ctx := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
				source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.AnthropicMessagesV1,
				claude: true, body: test.body,
			})
			got := messagesBreakpoints(t, body)
			if len(got) != 1 || got["messages[2][0]"] != `{"type":"ephemeral"}` {
				t.Fatalf("breakpoints %v in %s, want one 5m breakpoint on the last block", got, body)
			}
			if !hasAutoCacheDiagnostic(ctx, test.reason) {
				t.Fatalf("no %s diagnostic in %+v", test.reason, ctx.ProtocolDiagnostics)
			}
			if ctx.SemanticRequest.AutoCache != nil {
				t.Fatal("the client's own request was given the directive")
			}
		})
	}
}

// A client's own top-level directive is placed as the codec places it, with
// its ttl; the Router supplies nothing of its own.
func TestClaudeAutoCacheKeepsAnExplicitTopLevelDirective(t *testing.T) {
	body, ctx := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
		source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.AnthropicMessagesV1,
		claude: true, body: responsesTopLevel1h,
	})
	got := messagesBreakpoints(t, body)
	if len(got) != 1 || got["messages[2][0]"] != `{"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("breakpoints %v in %s, want the client's 1h directive on the last block", got, body)
	}
	if ctx.DispatchAutoCache != nil || hasAutoCacheDiagnostic(ctx, "automatic_cache_default") ||
		hasAutoCacheDiagnostic(ctx, "automatic_cache_prompt_cache_key") {
		t.Fatalf("the Router supplied a directive beside the client's: %+v", ctx.ProtocolDiagnostics)
	}
}

// A Chat client's per-block breakpoints are kept. The automatic one is added
// by placeAutoCacheBreakpoint's rules only: on the last block when that block
// holds none, never a second one on a block that already holds the client's.
func TestClaudeAutoCacheKeepsPerBlockBreakpoints(t *testing.T) {
	const systemOnly = `{"model":"m","messages":[` +
		`{"role":"system","content":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral","ttl":"1h"}}]},` +
		`{"role":"user","content":"Open main.go."}]}`
	body, _ := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
		source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1, claude: true, body: systemOnly,
	})
	got := messagesBreakpoints(t, body)
	if len(got) != 2 || got["system[0]"] != `{"type":"ephemeral","ttl":"1h"}` ||
		got["messages[0][0]"] != `{"type":"ephemeral"}` {
		t.Fatalf("breakpoints %v in %s, want the client's on system and one on the last block", got, body)
	}

	const lastHeld = `{"model":"m","messages":[` +
		`{"role":"system","content":"You are a coding agent."},` +
		`{"role":"user","content":[{"type":"text","text":"Open main.go.","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`
	body, _ = dispatchClaudeAutoCache(t, claudeAutoCacheCase{
		source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1, claude: true, body: lastHeld,
	})
	got = messagesBreakpoints(t, body)
	if len(got) != 1 || got["messages[0][0]"] != `{"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("breakpoints %v in %s, want only the client's on the last block", got, body)
	}
}

// A client whose own breakpoints use all four Messages allows gets none
// added, and the drop is counted.
func TestClaudeAutoCacheAddsNoneBeyondTheLimit(t *testing.T) {
	const fourBreakpoints = `{"model":"m","messages":[` +
		`{"role":"system","content":[{"type":"text","text":"One.","cache_control":{"type":"ephemeral"}},` +
		`{"type":"text","text":"Two.","cache_control":{"type":"ephemeral"}}]},` +
		`{"role":"user","content":[{"type":"text","text":"Three.","cache_control":{"type":"ephemeral"}}]},` +
		`{"role":"assistant","content":[{"type":"text","text":"Four.","cache_control":{"type":"ephemeral"}}]},` +
		`{"role":"user","content":"Open main.go."}]}`
	body, ctx := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
		source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1, claude: true, body: fourBreakpoints,
	})
	got := messagesBreakpoints(t, body)
	if len(got) != 4 || got["messages[2][0]"] != "" {
		t.Fatalf("breakpoints %v in %s, want the client's four and none on the last block", got, body)
	}
	dropped := false
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == "cache_control" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped = true
		}
	}
	if !dropped {
		t.Fatalf("the dropped breakpoint was not counted: %+v", ctx.ProtocolDiagnostics)
	}
}

// Only a Claude worker reached over Messages from an OpenAI-shaped client is
// given the directive: a non-Claude Messages worker, a Claude worker over
// Chat, and a Messages client all dispatch no cache_control.
func TestClaudeAutoCacheAppliesOnlyToClaudeOverMessagesFromOpenAIClients(t *testing.T) {
	for name, test := range map[string]claudeAutoCacheCase{
		"non-Claude Messages worker": {
			source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.AnthropicMessagesV1, body: responsesNoDirective,
		},
		"Claude worker over Chat": {
			source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.OpenAIChatV1, claude: true, body: responsesNoDirective,
		},
		"Messages client": {
			source: llmprotocol.AnthropicMessagesV1, target: llmprotocol.AnthropicMessagesV1, claude: true,
			body: `{"model":"m","max_tokens":64,"system":"You are a coding agent.",` +
				`"messages":[{"role":"user","content":"Open main.go."}]}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			body, ctx := dispatchClaudeAutoCache(t, test)
			if containsCacheControl(body) {
				t.Fatalf("dispatched %s", body)
			}
			if ctx.DispatchAutoCache != nil {
				t.Fatalf("the Router planned a directive: %+v", ctx.DispatchAutoCache)
			}
		})
	}
}

// The directive and its placement depend only on the request, so the same
// request dispatches the same bytes, and a conversation's prefix stays cache
// stable from turn to turn.
func TestClaudeAutoCacheIsDeterministic(t *testing.T) {
	test := claudeAutoCacheCase{
		source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.AnthropicMessagesV1, claude: true, body: responsesCacheKey,
	}
	first, _ := dispatchClaudeAutoCache(t, test)
	second, _ := dispatchClaudeAutoCache(t, test)
	if !bytes.Equal(first, second) {
		t.Fatalf("the same request dispatched\n%s\nthen\n%s", first, second)
	}
	if !containsCacheControl(first) {
		t.Fatalf("dispatched %s, want a breakpoint", first)
	}
}
