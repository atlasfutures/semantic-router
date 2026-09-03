//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const (
	upstreamSessionArm      = "arc/model@thinking-off"
	upstreamSessionRawID    = "user-42:conversation-7"
	upstreamSessionDecision = "rayline-arc-dev"
)

// OpenRouter pins a conversation's turns to one provider endpoint when the
// request carries a stable session_id, so the provider-side prompt cache
// survives the turn boundary. The router already derives a stable episode id
// per conversation; this sends it.
func TestUpstreamSessionIDReachesOpenRouterWhenEnabled(t *testing.T) {
	body := upstreamSessionRoutedBody(t, upstreamSessionRouter("https://openrouter.ai/api/v1"), true, upstreamSessionRawID)

	want := raylinearc.HashEpisodeID(upstreamSessionRawID)
	if got := body["session_id"]; got != want {
		t.Fatalf("session_id = %#v, want the episode id hash %q", got, want)
	}
}

// The raw header is userId:conversationId. Sending it would hand the provider
// an account identifier, so only the hash may leave the cell.
func TestUpstreamSessionIDNeverCarriesTheRawSessionHeader(t *testing.T) {
	raw := upstreamSessionRouteBytes(t, upstreamSessionRouter("https://openrouter.ai/api/v1"), true, upstreamSessionRawID)

	if bytes.Contains(raw, []byte(upstreamSessionRawID)) {
		t.Fatalf("dispatched body carries the raw session header:\n%s", raw)
	}
	if bytes.Contains(raw, []byte("user-42")) {
		t.Fatalf("dispatched body carries the user id:\n%s", raw)
	}
}

// A request with no x-rayline-session header has no episode identity, so
// there is nothing to pin and the field must be absent rather than empty.
func TestUpstreamSessionIDIsAbsentWithoutAnEpisode(t *testing.T) {
	body := upstreamSessionRoutedBody(t, upstreamSessionRouter("https://openrouter.ai/api/v1"), true, "")

	if _, present := body["session_id"]; present {
		t.Fatalf("session_id is present with no episode: %#v", body["session_id"])
	}
}

// Default off. A decision that does not ask for the field must serialize the
// bytes it serialized before the option existed.
func TestUpstreamSessionIDIsAbsentWhenTheOptionIsOff(t *testing.T) {
	router := upstreamSessionRouter("https://openrouter.ai/api/v1")
	off := upstreamSessionRouteBytes(t, router, false, upstreamSessionRawID)

	if bytes.Contains(off, []byte("session_id")) {
		t.Fatalf("session_id appears with the option off:\n%s", off)
	}
}

// session_id is an OpenRouter field. Another OpenAI-compatible backend would
// see an unknown member, and some of them refuse one.
func TestUpstreamSessionIDStaysOffNonOpenRouterBackends(t *testing.T) {
	body := upstreamSessionRoutedBody(t, upstreamSessionRouter("http://127.0.0.1:8000/v1"), true, upstreamSessionRawID)

	if _, present := body["session_id"]; present {
		t.Fatalf("session_id reached a non-OpenRouter backend: %#v", body["session_id"])
	}
}

// The field is sent upstream only. The neutral request the response path
// renders from must never learn about it.
func TestUpstreamSessionIDIsNotWrittenBackToTheNeutralRequest(t *testing.T) {
	router := upstreamSessionRouter("https://openrouter.ai/api/v1")
	request := testNeutralRequest("auto", "pin this conversation")
	ctx := upstreamSessionContext(request, true, upstreamSessionRawID)

	upstreamSessionRoute(t, router, request, ctx)

	if ctx.SemanticRequest.Metadata["session_id"] != "" {
		t.Fatalf("neutral request metadata carries session_id: %#v", ctx.SemanticRequest.Metadata)
	}
	if ctx.SemanticRequest.Unmodeled != nil {
		if _, present := ctx.SemanticRequest.Unmodeled.Fields["session_id"]; present {
			t.Fatalf("neutral request unmodeled fields carry session_id")
		}
	}
}

func upstreamSessionRoutedBody(
	t *testing.T,
	router *OpenAIRouter,
	enabled bool,
	rawEpisodeID string,
) map[string]any {
	t.Helper()
	request := testNeutralRequest("auto", "pin this conversation")
	ctx := upstreamSessionContext(request, enabled, rawEpisodeID)
	return decodeUpstreamSessionBody(t, upstreamSessionRoute(t, router, request, ctx))
}

func upstreamSessionRouteBytes(
	t *testing.T,
	router *OpenAIRouter,
	enabled bool,
	rawEpisodeID string,
) []byte {
	t.Helper()
	request := testNeutralRequest("auto", "pin this conversation")
	ctx := upstreamSessionContext(request, enabled, rawEpisodeID)
	return upstreamSessionRoute(t, router, request, ctx)
}

func upstreamSessionRoute(
	t *testing.T,
	router *OpenAIRouter,
	request *llmprotocol.Request,
	ctx *RequestContext,
) []byte {
	t.Helper()
	response, err := router.handleEntrypointModelRouting(
		request, "auto", upstreamSessionDecision, entropy.ReasoningDecision{}, upstreamSessionArm, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func upstreamSessionContext(
	request *llmprotocol.Request,
	enabled bool,
	rawEpisodeID string,
) *RequestContext {
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	configuration := map[string]interface{}{}
	if enabled {
		configuration["send_upstream_session_id"] = true
	}
	ctx.VSRSelectedDecision = &config.Decision{
		Name: upstreamSessionDecision,
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: config.MustStructuredPayload(configuration)},
		},
	}
	if rawEpisodeID != "" {
		ctx.VSRRaylineARC = &selection.RaylineARCTrace{
			EpisodeIDHash: raylinearc.HashEpisodeID(rawEpisodeID),
		}
	}
	return ctx
}

func upstreamSessionRouter(baseURL string) *OpenAIRouter {
	arm := config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatOpenAI,
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: upstreamSessionArm,
			ModelConfig:  map[string]config.ModelParams{upstreamSessionArm: arm},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: "openai", BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

func decodeUpstreamSessionBody(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	return body
}
