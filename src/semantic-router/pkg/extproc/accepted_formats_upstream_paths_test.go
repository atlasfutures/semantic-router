package extproc

import (
	"context"
	"net/http"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

const acceptedFormatsChatAnswer = `{"id":"c","object":"chat.completion","created":1,"model":"model-fallback-2","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// A fallback candidate picks its target format by the same accepted_formats
// rule as primary dispatch (#123): a Chat client reaches a Messages-first
// candidate that also accepts Chat in Chat. Without accepted_formats the
// candidate keeps its single api_format, and a Chat answer is not what it
// returns.
func TestFallbackCandidateFollowsAcceptedFormats(t *testing.T) {
	for name, tc := range map[string]struct {
		accepted []string
		want     bool
	}{
		"accepts the client's Chat": {[]string{config.APIFormatAnthropic, config.APIFormatOpenAI}, true},
		"single api_format":         {nil, false},
	} {
		t.Run(name, func(t *testing.T) {
			router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
			params := cfg.ModelConfig["model-fallback-2"]
			params.AcceptedFormats = tc.accepted
			cfg.ModelConfig["model-fallback-2"] = params
			ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-2"})
			ctx.UpstreamStatusCode = 503
			var sent []byte
			router.fallbackCaller = func(_ context.Context, _ string, body []byte, _ map[string]string) ([]byte, int, error) {
				sent = body
				return []byte(acceptedFormatsChatAnswer), http.StatusOK, nil
			}
			resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"down"}}`), ctx)
			gotChat := !strings.Contains(string(sent), `"max_tokens"`)
			served := resp.GetImmediateResponse() != nil && resp.GetImmediateResponse().GetStatus().GetCode() == typev3.StatusCode_OK
			if gotChat != tc.want || served != tc.want {
				t.Fatalf("Chat dispatch = %v, served = %v, want %v: %s", gotChat, served, tc.want, sent)
			}
		})
	}
}

// A shadow target follows the same rule: a Chat client's shadow to a
// Messages-first model that also accepts Chat goes out on the Chat path.
func TestShadowTargetFollowsAcceptedFormats(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.APIFormat = config.APIFormatAnthropic
	shadowParams.AcceptedFormats = []string{config.APIFormatAnthropic, config.APIFormatOpenAI}
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
		// Ingress stamps the client format on the neutral request
		// (prepareProtocolRequest); the shadow job reads it from there.
		ctx.SemanticRequest.Trusted.SourceFormat = ctx.SourceFormat
	})
	waitForShadow(t, router)
	if backend.requestCount() != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", backend.requestCount())
	}
	if !strings.HasSuffix(backend.uris[0], "/chat/completions") {
		t.Fatalf("shadow went to %q, want the Chat path the client's format selects", backend.uris[0])
	}
}
