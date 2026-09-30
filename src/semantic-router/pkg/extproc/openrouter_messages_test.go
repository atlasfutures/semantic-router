//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const openRouterMessagesConfig = `
version: v0.3
providers:
  defaults:
    model: claude
  models:
    - name: claude
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
      provider_preferences:
        order:
          - anthropic
        allow_fallbacks: false
      backend_refs:
        - name: openrouter-claude
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: OPENROUTER_MESSAGES_TEST_KEY
routing:
  modelCards:
    - name: claude
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: claude
`

// An OpenRouter model card can bind Anthropic Messages: the turn goes to
// OpenRouter's /api/v1/messages with its bearer key, and carries the provider
// pin and session_id, which OpenRouter's Messages API reads as its Chat API
// does.
func TestOpenRouterMessagesDispatch(t *testing.T) {
	t.Setenv("OPENROUTER_MESSAGES_TEST_KEY", "public-openrouter-test-key")
	cfg, err := config.ParseYAMLBytes([]byte(openRouterMessagesConfig))
	if err != nil {
		t.Fatalf("an OpenRouter Messages card did not load: %v", err)
	}
	if format := cfg.GetModelAPIFormat("claude"); format != config.APIFormatAnthropic {
		t.Fatalf("api format = %q, want anthropic", format)
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}

	request := testNeutralRequest("auto", "fix the failing test")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "default",
		Plugins: []config.DecisionPlugin{{
			Type:          "request_params",
			Configuration: config.MustStructuredPayload(map[string]interface{}{"send_upstream_session_id": true}),
		}},
	}
	ctx.VSRRaylineARC = &selection.RaylineARCTrace{EpisodeIDHash: raylinearc.HashEpisodeID("user:conversation")}
	response, err := router.handleEntrypointModelRouting(
		request, "auto", "default", entropy.ReasoningDecision{}, "claude", ctx,
	)
	if err != nil {
		t.Fatalf("routing: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	headersByName := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := headersByName[":path"]; got != "/api/v1/messages" {
		t.Fatalf(":path = %q, want /api/v1/messages", got)
	}
	if got := headersByName["Authorization"]; got != "Bearer public-openrouter-test-key" {
		t.Fatalf("authorization = %q, want the OpenRouter bearer key", got)
	}
	if _, present := headersByName["x-api-key"]; present || headersByName["X-Api-Key"] != "" {
		t.Fatal("an Anthropic x-api-key header was sent to OpenRouter")
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("provider body: %v", err)
	}
	if string(body["model"]) != `"anthropic/claude-opus-5"` {
		t.Fatalf("model = %s", body["model"])
	}
	if _, present := body["max_tokens"]; !present {
		t.Fatalf("the body is not a Messages body: %s", common.GetBodyMutation().GetBody())
	}
	if string(body["provider"]) != `{"order":["anthropic"],"allow_fallbacks":false}` {
		t.Fatalf("provider pin = %s", body["provider"])
	}
	if want := `"` + raylinearc.HashEpisodeID("user:conversation") + `"`; string(body["session_id"]) != want {
		t.Fatalf("session_id = %s, want %s", body["session_id"], want)
	}
	if !reflect.DeepEqual(ctx.DispatchedProviderOrder, []string{"anthropic"}) {
		t.Fatalf("recorded provider order = %v", ctx.DispatchedProviderOrder)
	}
}

// OpenRouter's Messages response adds members Anthropic's does not: the
// serving provider, routing metadata and cost. They are dropped, not fatal.
func TestOpenRouterMessagesResponseReachesAChatClient(t *testing.T) {
	upstream := `{"id":"msg_1","type":"message","role":"assistant","model":"anthropic/claude-opus-5",` +
		`"content":[{"type":"text","text":"Fixed."}],"stop_reason":"end_turn","stop_sequence":null,` +
		`"provider":"Anthropic","openrouter_metadata":{"provider":"Anthropic"},` +
		`"usage":{"input_tokens":12,"output_tokens":3,"cost":0.0004,"is_byok":false,` +
		`"cost_details":{"upstream_inference_cost":0.0004}}}`
	result, err := protocolcodec.NewBuiltinEngine().TranslateResponse(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(upstream), nil)
	if err != nil {
		t.Fatalf("an OpenRouter Messages response failed to decode: %v", err)
	}
	var chat struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(result.Body, &chat); err != nil || len(chat.Choices) != 1 || chat.Choices[0].Message.Content != "Fixed." {
		t.Fatalf("chat response = %s (err %v)", result.Body, err)
	}
}
