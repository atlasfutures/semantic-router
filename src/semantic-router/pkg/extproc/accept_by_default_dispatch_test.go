package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// dispatchAnthropicForTest renders a Claude Code body for a Chat arm the way
// dispatch does: the backend projection and codec (encodeDispatchRequest),
// then the provider adapter. A codec-only test would pass while the extproc
// projection still refused.
func dispatchAnthropicForTest(
	t *testing.T,
	router *OpenAIRouter,
	model string,
	useReasoning bool,
	profile *config.ProviderProfile,
	body string,
) (map[string]json.RawMessage, *RequestContext) {
	t.Helper()
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	request.Model = "vendor/" + model
	request.Generation++
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &request, ProtocolEnvelope: envelope, RequestModel: model,
	}
	dispatch := &providerDispatch{
		logicalModel: model, upstreamModel: "vendor/" + model, profile: profile,
		targetFormat: llmprotocol.OpenAIChatV1, decisionName: "arc", useReasoning: useReasoning,
	}
	router.bindDispatchProviderFacts(dispatch, ctx)
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode refused the turn: %v", err)
	}
	adapted, err := router.adaptProviderRequest(encoded, dispatch, ctx)
	if err != nil {
		t.Fatalf("adapter refused the turn: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(adapted, &wire); err != nil {
		t.Fatalf("provider body: %v\n%s", err, adapted)
	}
	return wire, ctx
}

func hasDiagnostic(ctx *RequestContext, field string) bool {
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == field {
			return true
		}
	}
	return false
}

// US-003e: Claude Code's thinking disabled (with effort high) routed to an
// OpenRouter Chat arm is honoured, never refused, whether or not the arm's
// model has a reasoning family, and the off-signal is the one OpenRouter's
// providers read: reasoning.enabled false, with no bound and no effort.
func TestClaudeCodeThinkingDisabledReachesAnOpenRouterChatArm(t *testing.T) {
	body := `{"model":"auto","max_tokens":32000,"thinking":{"type":"disabled"},` +
		`"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
	for name, tc := range map[string]struct {
		router *OpenAIRouter
		model  string
	}{
		"qwen3 family":     {newChatTemplateArmReasoningRouter(), "qwen3-model"},
		"no family":        {&OpenAIRouter{Config: &config.RouterConfig{}}, "unfamilied-model"},
		"reasoning family": {newArmReasoningRouter(), "gpt-5-mini"},
	} {
		t.Run(name, func(t *testing.T) {
			wire, _ := dispatchAnthropicForTest(t, tc.router, tc.model, true, openRouterProviderProfile(), body)
			var reasoning struct {
				Enabled   *bool  `json:"enabled"`
				MaxTokens *int64 `json:"max_tokens"`
				Effort    string `json:"effort"`
			}
			if err := json.Unmarshal(wire["reasoning"], &reasoning); err != nil ||
				reasoning.Enabled == nil || *reasoning.Enabled {
				t.Fatalf("no reasoning.enabled false: %s", wire["reasoning"])
			}
			if reasoning.MaxTokens != nil || reasoning.Effort != "" {
				t.Fatalf("the off-signal travelled with a reasoning request: %s", wire["reasoning"])
			}
			if effort, present := wire["reasoning_effort"]; present {
				t.Fatalf("reasoning_effort travelled beside the off-signal: %s", effort)
			}
		})
	}
}

// US-003g: a context_management edit a Chat arm cannot apply is dropped and
// counted, never refused; the keep-all no-op leaves no diagnostic.
func TestClaudeCodeContextManagementIsDroppedForAChatArm(t *testing.T) {
	for name, tc := range map[string]struct {
		edit       string
		diagnostic bool
	}{
		"tool clearing edit": {`{"edits":[{"type":"clear_tool_uses_20250919"}]}`, true},
		"keep-all no-op":     {`{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`, false},
	} {
		t.Run(name, func(t *testing.T) {
			body := `{"model":"auto","max_tokens":1024,"context_management":` + tc.edit +
				`,"messages":[{"role":"user","content":"hi"}]}`
			wire, ctx := dispatchAnthropicForTest(t, newArmReasoningRouter(), "gpt-5-mini", true, openRouterProviderProfile(), body)
			if _, present := wire["context_management"]; present {
				t.Fatalf("context_management reached the Chat provider: %s", wire["context_management"])
			}
			if got := hasDiagnostic(ctx, "context_management"); got != tc.diagnostic {
				t.Fatalf("context_management diagnostic = %v, want %v: %+v", got, tc.diagnostic, ctx.ProtocolDiagnostics)
			}
		})
	}
}

// US-003h: a summarized thinking display a Chat arm cannot produce is dropped
// and counted, never refused.
func TestClaudeCodeSummarizedThinkingDisplayIsDroppedForAChatArm(t *testing.T) {
	body := `{"model":"auto","max_tokens":32000,"thinking":{"type":"adaptive","display":"summarized"},` +
		`"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
	_, ctx := dispatchAnthropicForTest(t, newArmReasoningRouter(), "gpt-5-mini", true, openRouterProviderProfile(), body)
	if !hasDiagnostic(ctx, "reasoning_display") {
		t.Fatalf("no reasoning_display diagnostic: %+v", ctx.ProtocolDiagnostics)
	}
}

// US-004: a per-message reasoning effort reaches an OpenRouter Chat provider
// as configuration_update; a Chat backend that is not OpenRouter omits it.
func TestPerMessageEffortReachesOpenRouterChatAsConfigurationUpdate(t *testing.T) {
	for name, tc := range map[string]struct {
		profile *config.ProviderProfile
		want    bool
	}{
		"openrouter": {openRouterProviderProfile(), true},
		"plain vllm": {&config.ProviderProfile{Type: "vllm"}, false},
	} {
		t.Run(name, func(t *testing.T) {
			request := llmprotocol.Request{
				Model: "vendor/m", Generation: 1,
				Messages: []llmprotocol.Message{
					{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}}},
					{Role: llmprotocol.RoleSystem, ReasoningEffort: "low"},
					{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go on"}}},
				},
			}
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
				SemanticRequest: &request, RequestModel: "m",
			}
			dispatch := &providerDispatch{logicalModel: "m", profile: tc.profile, targetFormat: llmprotocol.OpenAIChatV1}
			(&OpenAIRouter{}).bindDispatchProviderFacts(dispatch, ctx)
			encoded, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if got := strings.Contains(string(encoded), `"configuration_update"`); got != tc.want {
				t.Fatalf("configuration_update sent = %v, want %v: %s", got, tc.want, encoded)
			}
		})
	}
}
