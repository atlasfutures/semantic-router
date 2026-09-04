package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A thinking-off arm has to state the off-signal in the control the provider
// reads, and on OpenRouter that control is reasoning.enabled.
//
// chat_template_kwargs.enable_thinking is a vLLM chat-template argument. The
// providers OpenRouter picks for the ARC arms do not read it. Measured against
// OpenRouter 2026-09-04 on deepseek-v4-pro, deepseek-v4-flash, mimo-v2.5-pro,
// qwen3.6-35b-a3b and glm-5.2, a body carrying only that flag reasoned anyway:
// every arm spent its whole 64-token budget on reasoning, returned empty
// content and finished on "length". The same five with reasoning.enabled false
// returned reasoning_tokens 0 and content. So a thinking-off arm today is
// billed at the output rate for hidden reasoning, answers slower, and returns
// nothing at all when the budget is small.
func TestThinkingOffArmStatesTheOffSignalOpenRouterReads(t *testing.T) {
	claudeCodeBody := `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],` +
		`"max_completion_tokens":32000,"reasoning":{"max_tokens":32000},"reasoning_effort":"high"}`

	t.Run("the off-signal travels", func(t *testing.T) {
		body := reasoningBodyForProfile(t, claudeCodeBody, false, openRouterProviderProfile())
		if enabled := reasoningEnabledOf(t, body); enabled == nil || *enabled {
			t.Fatalf("a thinking-off arm carried no off-signal: %v", body["reasoning"])
		}
		if bound := reasoningBoundOf(t, body); bound != nil {
			t.Fatalf("the off-signal travelled with a bound: %v", body["reasoning"])
		}
		assertNoReasoningEffort(t, body)
	})

	t.Run("the chat-template flag stays beside it", func(t *testing.T) {
		router := newChatTemplateArmReasoningRouter()
		encoded, err := router.setReasoningModeToRequestBodyForModelAndProvider(
			[]byte(claudeCodeBody), "qwen3-model", false,
			router.Config.GetDecisionByName("arc"), openRouterProviderProfile(), nil,
		)
		require.NoError(t, err)
		var body map[string]interface{}
		require.NoError(t, json.Unmarshal(encoded, &body))

		if enabled := reasoningEnabledOf(t, body); enabled == nil || *enabled {
			t.Fatalf("a thinking-off arm carried no off-signal: %v", body["reasoning"])
		}
		kwargs, present := body["chat_template_kwargs"].(map[string]interface{})
		if !present || kwargs["enable_thinking"] != false {
			t.Fatalf("the chat-template off-flag was lost: %v", body["chat_template_kwargs"])
		}
	})

	// reasoning is OpenRouter's object. Every other OpenAI-compatible backend
	// would see an unknown member.
	t.Run("only OpenRouter gets it", func(t *testing.T) {
		body := reasoningBodyForProfile(t, claudeCodeBody, false,
			&config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"})
		if _, present := body["reasoning"]; present {
			t.Fatalf("an OpenRouter-only member reached another backend: %v", body)
		}
	})

	// The off-signal is the thinking-off arm's alone. A thinking-on arm keeps
	// the shape that already works: the bound, and nothing beside it.
	t.Run("a thinking-on arm keeps its bound", func(t *testing.T) {
		body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
			`"max_completion_tokens":4096}`)
		bound := reasoningBoundOf(t, body)
		if bound == nil || *bound != 4096 {
			t.Fatalf("the thinking-on bound changed: %v", body["reasoning"])
		}
		if enabled := reasoningEnabledOf(t, body); enabled != nil {
			t.Fatalf("the thinking-on arm carried an enabled flag: %v", body["reasoning"])
		}
	})
}

// reasoningEnabledOf reads the off-signal OpenRouter documents, or nil when the
// request states none.
func reasoningEnabledOf(t *testing.T, body map[string]interface{}) *bool {
	t.Helper()
	reasoning, present := body["reasoning"].(map[string]interface{})
	if !present {
		return nil
	}
	enabled, present := reasoning["enabled"].(bool)
	if !present {
		return nil
	}
	return &enabled
}

// newChatTemplateArmReasoningRouter is an arm whose reasoning family speaks the
// vLLM chat-template flag, which is what the ARC model cards configure.
func newChatTemplateArmReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"qwen3": {Type: "chat_template_kwargs", Parameter: "enable_thinking"},
			},
		},
		[]config.Decision{
			reasoningDecision("arc", "", 0, "qwen3-model", boolPtr(false), ""),
		},
		map[string]config.ModelParams{"qwen3-model": {ReasoningFamily: "qwen3"}},
	)
}
