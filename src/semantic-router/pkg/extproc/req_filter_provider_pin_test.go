//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const (
	providerPinArm      = "arc/model@thinking-off"
	providerPinDecision = "rayline-arc-dev"
)

// The wire bytes a pinned arm has to produce. OpenRouter reads the provider
// object as its routing instruction, so the order and the fallback verdict
// travel together: an order with fallbacks still allowed pins nothing.
// https://openrouter.ai/docs/features/provider-routing, read 2026-09-05.
const providerPinWireBytes = `"provider":{"order":["deepinfra","together"],"allow_fallbacks":false}`

// A pinned arm names its providers on the dispatched body, and names only
// what the config set. The keys the operator left unset must be absent rather
// than sent empty: OpenRouter reads an empty only list as "no provider may
// serve this", which would refuse every turn.
func TestPinnedArmSendsTheProviderObjectOpenRouterReads(t *testing.T) {
	raw := providerPinRouteBytes(t, providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences()))

	if !bytes.Contains(raw, []byte(providerPinWireBytes)) {
		t.Fatalf("the pinned arm did not send the provider object:\n%s", raw)
	}
	pin := providerPinObject(t, raw)
	for _, unset := range []string{"only", "ignore", "require_parameters", "data_collection"} {
		if _, present := pin[unset]; present {
			t.Fatalf("an unset provider preference reached the wire (%s): %#v", unset, pin)
		}
	}
}

// provider is OpenRouter's object. Every other OpenAI-compatible backend
// would see an unknown member, and some of them refuse one.
func TestProviderPinStaysOffNonOpenRouterBackends(t *testing.T) {
	body := providerPinRoutedBody(t, providerPinRouter("http://127.0.0.1:8000/v1", providerPinPreferences()))

	if _, present := body["provider"]; present {
		t.Fatalf("an OpenRouter-only member reached another backend: %#v", body["provider"])
	}
}

// Default off. An arm without the key must serialize the bytes it serialized
// before the key existed.
func TestProviderPinIsAbsentWhenTheArmDoesNotPin(t *testing.T) {
	raw := providerPinRouteBytes(t, providerPinRouter("https://openrouter.ai/api/v1", nil))

	if bytes.Contains(raw, []byte(`"provider"`)) {
		t.Fatalf("an unpinned arm carried a provider object:\n%s", raw)
	}
}

// The routing record names the providers the request actually pinned, read
// back off the rendered body rather than recomputed from the arm's config,
// which is the same rule the reasoning controls follow.
func TestRoutingDecisionNamesThePinnedProviders(t *testing.T) {
	logs := captureLogs(t)
	router := providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences())
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ProcessingStartTime = time.Now()
	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", providerPinArm, providerPinDecision, false)

	providerPinRoute(t, router, request, ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	if order := fmt.Sprint(fields["provider_order"]); order != "[deepinfra together]" {
		t.Fatalf("the record did not name the pinned providers: %#v", fields["provider_order"])
	}
}

// A turn that pins nothing must not gain a field naming providers it never
// asked for.
func TestRoutingDecisionOmitsProviderOrderWhenNothingIsPinned(t *testing.T) {
	logs := captureLogs(t)
	router := providerPinRouter("https://openrouter.ai/api/v1", nil)
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ProcessingStartTime = time.Now()
	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", providerPinArm, providerPinDecision, false)

	providerPinRoute(t, router, request, ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	if _, present := fields["provider_order"]; present {
		t.Fatalf("the record named providers on an unpinned turn: %#v", fields)
	}
}

// The pin and the CP9o reasoning object are two members of one body. Writing
// the pin must keep whatever the reasoning mutation already put there.
func TestProviderPinSitsBesideTheReasoningObject(t *testing.T) {
	router := providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences())
	dispatch := &providerDispatch{
		logicalModel: providerPinArm,
		decisionName: providerPinDecision,
		targetFormat: llmprotocol.OpenAIChatV1,
		profile:      openRouterProviderProfile(),
	}
	reasoned := []byte(`{"model":"` + providerPinArm + `","reasoning":{"max_tokens":1024}}`)

	pinned, err := applyProviderPreferences(reasoned, dispatch, router.Config)
	if err != nil {
		t.Fatalf("applyProviderPreferences: %v", err)
	}
	if !bytes.Contains(pinned, []byte(providerPinWireBytes)) {
		t.Fatalf("the pin did not travel beside the reasoning object:\n%s", pinned)
	}
	if !bytes.Contains(pinned, []byte(`"reasoning":{"max_tokens":1024}`)) {
		t.Fatalf("the pin overwrote the reasoning object:\n%s", pinned)
	}
}

// providerPinObject reads the provider object off the dispatched bytes, so a
// claim about what the pin omits is a claim about that object rather than about
// any substring of the body.
func providerPinObject(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var body struct {
		Provider map[string]any `json:"provider"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	if body.Provider == nil {
		t.Fatalf("the dispatched body carries no provider object:\n%s", raw)
	}
	return body.Provider
}

func providerPinPreferences() *config.OpenRouterProviderPreferences {
	allowFallbacks := false
	return &config.OpenRouterProviderPreferences{
		Order:          []string{"deepinfra", "together"},
		AllowFallbacks: &allowFallbacks,
	}
}

func providerPinRoutedBody(t *testing.T, router *OpenAIRouter) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(providerPinRouteBytes(t, router), &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	return body
}

func providerPinRouteBytes(t *testing.T, router *OpenAIRouter) []byte {
	t.Helper()
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	return providerPinRoute(t, router, request, ctx)
}

func providerPinRoute(
	t *testing.T,
	router *OpenAIRouter,
	request *llmprotocol.Request,
	ctx *RequestContext,
) []byte {
	t.Helper()
	response, err := router.handleEntrypointModelRouting(
		request, "auto", providerPinDecision, entropy.ReasoningDecision{}, providerPinArm, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func providerPinRouter(baseURL string, preferences *config.OpenRouterProviderPreferences) *OpenAIRouter {
	arm := config.ModelParams{
		PreferredEndpoints:  []string{"backend"},
		APIFormat:           config.APIFormatOpenAI,
		ProviderPreferences: preferences,
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: providerPinArm,
			ModelConfig:  map[string]config.ModelParams{providerPinArm: arm},
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
