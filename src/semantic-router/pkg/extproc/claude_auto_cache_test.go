package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
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
	dispatch, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	response, err := router.finalizeProviderDispatchResponse(dispatch, router.buildProviderDispatchResponse(dispatch, ctx), ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), ctx
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
	if hasAutoCacheDiagnostic(ctx, "automatic_cache_default") ||
		hasAutoCacheDiagnostic(ctx, "automatic_cache_prompt_cache_key") {
		t.Fatalf("the Router supplied a directive beside the client's: %+v", ctx.ProtocolDiagnostics)
	}
}

// A client that placed its own breakpoints has stated its cache intent, so
// the Router adds none: not beside a breakpoint on a system block, a message
// block, a tool, or a block inside a tool result. The client's breakpoints
// are dispatched as written.
func TestClaudeAutoCacheAddsNothingBesideClientBreakpoints(t *testing.T) {
	const ephemeral1h = `{"type":"ephemeral","ttl":"1h"}`
	for name, test := range map[string]struct {
		body string
		want map[string]string
	}{
		"system block": {
			body: `{"model":"m","messages":[` +
				`{"role":"system","content":[{"type":"text","text":"You are a coding agent.","cache_control":` + ephemeral1h + `}]},` +
				`{"role":"user","content":"Open main.go."}]}`,
			want: map[string]string{"system[0]": ephemeral1h},
		},
		"earlier message block": {
			body: `{"model":"m","messages":[` +
				`{"role":"user","content":[{"type":"text","text":"List the files.","cache_control":` + ephemeral1h + `}]},` +
				`{"role":"assistant","content":"There are two files."},` +
				`{"role":"user","content":"Open main.go."}]}`,
			want: map[string]string{"messages[0][0]": ephemeral1h},
		},
		"tool": {
			body: `{"model":"m","messages":[{"role":"user","content":"Open main.go."}],` +
				`"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}},"cache_control":` + ephemeral1h + `}]}`,
			want: map[string]string{"tools[0]": ephemeral1h},
		},
		"inside a tool result": {
			body: `{"model":"m","messages":[` +
				`{"role":"user","content":"Open main.go."},` +
				`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},` +
				`{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"package main","cache_control":` + ephemeral1h + `}]},` +
				`{"role":"user","content":"Explain it."}]}`,
			want: map[string]string{},
		},
	} {
		t.Run(name, func(t *testing.T) {
			body, ctx := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
				source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1, claude: true, body: test.body,
			})
			if hasAutoCacheDiagnostic(ctx, "automatic_cache_default") {
				t.Fatalf("the Router supplied a directive beside the client's breakpoints: %+v", ctx.ProtocolDiagnostics)
			}
			got := messagesBreakpoints(t, body)
			if name == "inside a tool result" {
				// The breakpoint sits inside the tool_result block, below the
				// depth messagesBreakpoints reads; it must be the only one.
				if len(got) != 0 || bytes.Count(body, []byte("cache_control")) != 1 {
					t.Fatalf("dispatched %s, want only the client's breakpoint in the tool result", body)
				}
				return
			}
			if len(got) != len(test.want) {
				t.Fatalf("breakpoints %v in %s, want %v", got, body, test.want)
			}
			for place, value := range test.want {
				if got[place] != value {
					t.Fatalf("breakpoints %v in %s, want %v", got, body, test.want)
				}
			}
		})
	}
}

// A client whose own breakpoints use all four Messages allows is a client
// that placed breakpoints: the Router adds no directive, so none is added
// and the codec has nothing to drop.
func TestClaudeAutoCacheAddsNoneToAFullClient(t *testing.T) {
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
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == "cache_control" {
			t.Fatalf("a cache_control diagnostic for a client that placed its own: %+v", diagnostic)
		}
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

// The decision is made on the request that is dispatched, after the
// decision's tools plugin has run. A breakpoint on a tool the plugin removes
// is never sent, so it does not stop the automatic one: without it the turn
// would carry neither the client's breakpoint nor the Router's. A breakpoint
// on a tool that survives selection is sent, and nothing is added beside it.
func TestClaudeAutoCacheCountsBreakpointsAfterToolSelection(t *testing.T) {
	semanticSelection := false
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled: true, Mode: config.ToolsPluginModeFiltered,
		AllowTools: []string{"read_file"}, SemanticSelection: &semanticSelection,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name:    "tool-selection",
		Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginTools, Configuration: payload}},
	}
	const ephemeral1h = `{"type":"ephemeral","ttl":"1h"}`
	body := func(cachedTool string) string {
		tool := func(name string) string {
			cache := ""
			if name == cachedTool {
				cache = `,"cache_control":` + ephemeral1h
			}
			return `{"type":"function","function":{"name":"` + name + `","parameters":{"type":"object"}}` + cache + `}`
		}
		return `{"model":"m","tool_choice":"auto","messages":[{"role":"user","content":"Open main.go."}],` +
			`"tools":[` + tool("read_file") + `,` + tool("write_file") + `]}`
	}
	for name, test := range map[string]struct {
		cachedTool string
		want       map[string]string
		generated  bool
	}{
		"breakpoint on a removed tool": {"write_file", map[string]string{"messages[0][0]": `{"type":"ephemeral"}`}, true},
		"breakpoint on a kept tool":    {"read_file", map[string]string{"tools[0]": ephemeral1h}, false},
	} {
		t.Run(name, func(t *testing.T) {
			router, logicalModel := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
			params := router.Config.ModelConfig[logicalModel]
			params.Publisher = "anthropic"
			router.Config.ModelConfig[logicalModel] = params
			ctx := &RequestContext{
				Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
				RequestID: "claude-auto-cache-tools", TraceContext: context.Background(),
				VSRSelectedDecision: decision,
			}
			request, immediate := router.prepareProtocolRequest([]byte(body(test.cachedTool)), ctx)
			if immediate != nil || request == nil {
				t.Fatalf("ingress refused the request: %+v", ctx.ImmediateProtocolError)
			}
			response, routeErr := router.handleEntrypointModelRouting(
				request, "entrypoint", decision.Name, entropy.ReasoningDecision{}, logicalModel, ctx,
			)
			if routeErr != nil {
				t.Fatal(routeErr)
			}
			dispatched := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			if !bytes.Contains(dispatched, []byte(`"read_file"`)) || bytes.Contains(dispatched, []byte(`"write_file"`)) {
				t.Fatalf("the tools plugin did not filter the tools: %s", dispatched)
			}
			got := messagesBreakpoints(t, dispatched)
			if len(got) != len(test.want) {
				t.Fatalf("breakpoints %v in %s, want %v", got, dispatched, test.want)
			}
			for place, value := range test.want {
				if got[place] != value {
					t.Fatalf("breakpoints %v in %s, want %v", got, dispatched, test.want)
				}
			}
			if hasAutoCacheDiagnostic(ctx, "automatic_cache_default") != test.generated {
				t.Fatalf("generated diagnostic = %v, want %v: %+v", !test.generated, test.generated, ctx.ProtocolDiagnostics)
			}
		})
	}
}

// A metadata-only Claude model -- a card with its publisher and no backend --
// is dispatched by the external gateway path, which builds its own dispatch.
// It meets the routed path at finalizeProviderDispatchResponse, where the
// rule is marked, so a Responses request converted to Messages for it gets
// the generated breakpoint too.
func TestClaudeAutoCacheOnTheExternalGatewayPath(t *testing.T) {
	const model = "claude-gateway"
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			model: {APIFormat: config.APIFormatAnthropic, Publisher: "anthropic"},
		}},
	}}
	if !router.usesExternalGatewayDispatch(model) {
		t.Fatal("the metadata-only model is not dispatched by the external gateway path")
	}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "claude-auto-cache-gateway", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(responsesNoDirective), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused the request: %+v", ctx.ImmediateProtocolError)
	}
	response, err := router.handleExternalGatewayModelRouting(request, model, ctx)
	if err != nil {
		t.Fatal(err)
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	got := messagesBreakpoints(t, body)
	if ctx.TargetFormat != llmprotocol.AnthropicMessagesV1 || len(got) != 1 ||
		got["messages[2][0]"] != `{"type":"ephemeral"}` {
		t.Fatalf("breakpoints %v in %s, want one 5m breakpoint on the last block", got, body)
	}
	if !hasAutoCacheDiagnostic(ctx, "automatic_cache_default") {
		t.Fatalf("no automatic_cache_default diagnostic in %+v", ctx.ProtocolDiagnostics)
	}
}

func hasRetentionDiagnostic(ctx *RequestContext, action llmprotocol.DiagnosticAction, reason string) bool {
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == "prompt_cache_retention" && diagnostic.Action == action && diagnostic.Reason == reason {
			return true
		}
	}
	return false
}

// prompt_cache_retention states how long the client wants its prefix kept.
// 24h is longer than Anthropic keeps one, so the breakpoint gets Anthropic's
// longest ttl, 1h, and the shortening is recorded as an approximation.
// in_memory matches the 5m default, so no ttl is sent. A value OpenAI does
// not define keeps the default and is recorded as dropped. The generated
// diagnostic names prompt_cache_key when the client sent one, else
// prompt_cache_retention.
func TestClaudeAutoCacheHonoursPromptCacheRetention(t *testing.T) {
	responses := func(members string) string {
		return `{"model":"m","instructions":"You are a coding agent.",` + members + autoCacheConversation + `}`
	}
	const chat24h = `{"model":"m","prompt_cache_retention":"24h","messages":[` +
		`{"role":"system","content":"You are a coding agent."},{"role":"user","content":"Open main.go."}]}`
	const ttl1h, ttlDefault = `{"type":"ephemeral","ttl":"1h"}`, `{"type":"ephemeral"}`
	for name, test := range map[string]struct {
		source      llmprotocol.WireFormat
		body        string
		place       string
		breakpoint  string
		generated   string
		approximate bool
		unknown     bool
	}{
		"Responses 24h": {
			llmprotocol.OpenAIResponsesV1, responses(`"prompt_cache_retention":"24h",`),
			"messages[2][0]", ttl1h, "automatic_cache_prompt_cache_retention", true, false,
		},
		"Chat 24h": {
			llmprotocol.OpenAIChatV1, chat24h,
			"messages[0][0]", ttl1h, "automatic_cache_prompt_cache_retention", true, false,
		},
		"Responses in_memory": {
			llmprotocol.OpenAIResponsesV1, responses(`"prompt_cache_retention":"in_memory",`),
			"messages[2][0]", ttlDefault, "automatic_cache_prompt_cache_retention", false, false,
		},
		"Responses 24h with prompt_cache_key": {
			llmprotocol.OpenAIResponsesV1, responses(`"prompt_cache_key":"session-1","prompt_cache_retention":"24h",`),
			"messages[2][0]", ttl1h, "automatic_cache_prompt_cache_key", true, false,
		},
		"Responses unknown retention": {
			llmprotocol.OpenAIResponsesV1, responses(`"prompt_cache_retention":"7d",`),
			"messages[2][0]", ttlDefault, "automatic_cache_prompt_cache_retention", false, true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			body, ctx := dispatchClaudeAutoCache(t, claudeAutoCacheCase{
				source: test.source, target: llmprotocol.AnthropicMessagesV1, claude: true, body: test.body,
			})
			got := messagesBreakpoints(t, body)
			if len(got) != 1 || got[test.place] != test.breakpoint {
				t.Fatalf("breakpoints %v in %s, want %s on %s", got, body, test.breakpoint, test.place)
			}
			if !hasAutoCacheDiagnostic(ctx, test.generated) {
				t.Fatalf("no %s diagnostic in %+v", test.generated, ctx.ProtocolDiagnostics)
			}
			if hasRetentionDiagnostic(ctx, llmprotocol.DiagnosticApproximated, "prompt_cache_retention_24h_as_1h_ttl") != test.approximate {
				t.Fatalf("approximated retention diagnostic present = %v, want %v: %+v", !test.approximate, test.approximate, ctx.ProtocolDiagnostics)
			}
			if hasRetentionDiagnostic(ctx, llmprotocol.DiagnosticDropped, "prompt_cache_retention_unknown_value") != test.unknown {
				t.Fatalf("unknown retention diagnostic present = %v, want %v: %+v", !test.unknown, test.unknown, ctx.ProtocolDiagnostics)
			}
		})
	}
}

// metadataOnlyGatewayConfig declares three metadata-only models, as an
// external-gateway deployment does: no backend, so the provider model id is
// all the router knows of where each one is served. The aliases carry no
// vendor prefix.
const metadataOnlyGatewayConfig = `
version: v0.3
listeners: []
providers:
  models:
    - name: claude-sonnet
      provider_model_id: anthropic/claude-sonnet-4.5
      api_format: anthropic
    - name: house-model
      provider_model_id: anthropic/claude-sonnet-4.5
      api_format: anthropic
    - name: minimax-messages
      provider_model_id: minimax/minimax-m2
      api_format: anthropic
routing:
  modelCards:
    - name: claude-sonnet
    - name: house-model
      publisher: Acme
    - name: minimax-messages
global:
  router:
    model_selection:
      enabled: false
`

// A metadata-only model has no backend to resolve its provider model
// through, so armFamily reads the vendor from its provider_model_id. An
// unprefixed alias bound to anthropic/... is Claude, and a Responses
// request to it through the external gateway path gets the generated
// breakpoint. A card's publisher still wins over the provider model id,
// and a non-Claude model gets no breakpoint.
func TestClaudeAutoCacheFamilyOfAMetadataOnlyModel(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(metadataOnlyGatewayConfig))
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		model      string
		family     string
		breakpoint bool
	}{
		"Claude by provider_model_id":     {"claude-sonnet", "anthropic", true},
		"publisher wins":                  {"house-model", "acme", false},
		"non-Claude by provider_model_id": {"minimax-messages", "minimax", false},
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{Config: cfg}
			if !router.usesExternalGatewayDispatch(test.model) {
				t.Fatalf("%s is not dispatched by the external gateway path", test.model)
			}
			if got := router.armFamily(test.model); got != test.family {
				t.Fatalf("armFamily(%q) = %q, want %q", test.model, got, test.family)
			}
			ctx := &RequestContext{
				Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
				RequestID: "claude-auto-cache-metadata", TraceContext: context.Background(),
			}
			request, immediate := router.prepareProtocolRequest([]byte(responsesNoDirective), ctx)
			if immediate != nil || request == nil {
				t.Fatalf("ingress refused the request: %+v", ctx.ImmediateProtocolError)
			}
			response, err := router.handleExternalGatewayModelRouting(request, test.model, ctx)
			if err != nil {
				t.Fatal(err)
			}
			body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			if ctx.TargetFormat != llmprotocol.AnthropicMessagesV1 {
				t.Fatalf("target format = %q, want Messages", ctx.TargetFormat)
			}
			got := messagesBreakpoints(t, body)
			if test.breakpoint {
				if len(got) != 1 || got["messages[2][0]"] != `{"type":"ephemeral"}` {
					t.Fatalf("breakpoints %v in %s, want one on the last block", got, body)
				}
			} else if containsCacheControl(body) {
				t.Fatalf("a non-Claude model was dispatched %s", body)
			}
		})
	}
}

// A model with a backend is read through the id that backend is sent, not
// its default provider model id: the fallback is for a model with no
// backend to resolve.
func TestArmFamilyPrefersTheResolvedBackendModel(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	params := router.Config.ModelConfig[logicalModel]
	params.ExternalModelIDs = map[string]string{"vllm": "deepseek/deepseek-v4", "default": "anthropic/claude-sonnet-4.5"}
	router.Config.ModelConfig[logicalModel] = params
	if got := router.armFamily(logicalModel); got != "deepseek" {
		t.Fatalf("armFamily(%q) = %q, want the backend's deepseek", logicalModel, got)
	}
}
