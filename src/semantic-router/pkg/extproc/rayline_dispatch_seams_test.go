package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

const qualifiedToolCallChat = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"model-fallback-1","choices":[{"index":0,` +
	`"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{}"}}]},` +
	`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// A fallback candidate is encoded with its own provider facts, not the
// primary's: an OpenRouter Chat candidate receives the per-message effort as
// configuration_update, and the primary's hosted-tool admission does not
// travel to a candidate whose card admits none.
func TestFallbackCandidateEncodesWithItsOwnProviderFacts(t *testing.T) {
	router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	cfg.ProviderProfiles["prof-openrouter"] = config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}
	cfg.VLLMEndpoints[1].ProviderProfileName = "prof-openrouter"

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.DispatchHostedTools = []string{"web_search"} // the primary's admission
	ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages,
		llmprotocol.Message{Role: llmprotocol.RoleSystem, ReasoningEffort: "low"},
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go on"}}},
	)
	var sent []byte
	router.fallbackCaller = func(_ context.Context, _ string, body []byte, _ map[string]string) ([]byte, int, error) {
		sent = body
		return []byte(`{"id":"c","object":"chat.completion","created":1,"model":"model-fallback-1","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), http.StatusOK, nil
	}
	if resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"down"}}`), ctx); resp.GetImmediateResponse() == nil {
		t.Fatalf("fallback did not answer: %#v", resp)
	}
	if !bytes.Contains(sent, []byte(`"configuration_update"`)) {
		t.Fatalf("the OpenRouter candidate lost the per-message effort: %s", sent)
	}
	if len(ctx.DispatchHostedTools) != 0 {
		t.Fatalf("the primary's hosted-tool admission travelled to the candidate: %v", ctx.DispatchHostedTools)
	}
}

// A namespaced call from a fallback candidate comes back to the Responses
// client with its namespace, as it does on the primary path.
func TestFallbackCandidateRestoresToolNamespaces(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
	ctx.SemanticRequest.Tools = namespacedToolRequest().Tools
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		return []byte(qualifiedToolCallChat), http.StatusOK, nil
	}
	resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"down"}}`), ctx)
	body := resp.GetImmediateResponse().GetBody()
	if !bytes.Contains(body, []byte(`"name":"spawn_agent","namespace":"multi_agent_v1"`)) {
		t.Fatalf("the fallback call did not come back with its namespace: %s", body)
	}
}

// A shadow call gets the same provider step as primary dispatch: an
// OpenRouter shadow target carries its model's provider pin.
func TestShadowDispatchCarriesTheShadowModelsProviderPin(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	router.Config.ProviderProfiles["shadow-openrouter"] = config.ProviderProfile{Type: "openrouter", BaseURL: backend.server.URL}
	for i := range router.Config.VLLMEndpoints {
		if router.Config.VLLMEndpoints[i].Name == "shadow-backend" {
			router.Config.VLLMEndpoints[i].ProviderProfileName = "shadow-openrouter"
		}
	}
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: []string{"pinned-provider"}}
	router.Config.ModelConfig[shadowTestModel] = shadowParams

	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	if backend.requestCount() != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", backend.requestCount())
	}
	var wire struct {
		Provider struct {
			Order []string `json:"order"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(backend.bodies[0], &wire); err != nil {
		t.Fatalf("decode shadow request: %v", err)
	}
	if len(wire.Provider.Order) != 1 || wire.Provider.Order[0] != "pinned-provider" {
		t.Fatalf("the shadow body carries no provider pin: %s", backend.bodies[0])
	}
}

// A looper answer to a Responses client restores namespaced calls, buffered
// and streamed.
func TestLooperResponseRestoresToolNamespaces(t *testing.T) {
	stream := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"multi_agent_v1__spawn_agent\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	for name, response := range map[string]*looper.Response{
		"buffered": {Body: []byte(qualifiedToolCallChat), Model: "m", ContentType: "application/json"},
		"streamed": {Body: []byte(stream), Model: "m", ContentType: "text/event-stream"},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, SemanticRequest: namespacedToolRequest(), TraceContext: t.Context()}
			_, _, body, err := (&OpenAIRouter{}).prepareLooperResponse(response, ctx)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(body, []byte(`"namespace":"multi_agent_v1"`)) {
				t.Fatalf("the looper call did not come back with its namespace: %s", body)
			}
		})
	}
}
