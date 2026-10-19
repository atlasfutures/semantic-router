//go:build !windows && cgo

package extproc

import (
	"bytes"
	"net/http"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// withArmServiceTier sets an arm's configured service tier on a test router.
func withArmServiceTier(router *OpenAIRouter, arm string, tier string) {
	params := router.Config.ModelConfig[arm]
	params.ServiceTier = tier
	router.Config.ModelConfig[arm] = params
}

// A configured arm sends its tier as OpenRouter's top-level service_tier,
// and the dispatch records the tier the body carried.
func TestConfiguredArmSendsItsServiceTierToOpenRouter(t *testing.T) {
	router := providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences())
	withArmServiceTier(router, providerPinArm, "default")
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	raw := providerPinRoute(t, router, request, ctx)

	if !bytes.Contains(raw, []byte(`"service_tier":"default"`)) {
		t.Fatalf("the configured tier did not reach the wire:\n%s", raw)
	}
	if ctx.DispatchedServiceTier != "default" {
		t.Fatalf("dispatched service tier = %q, want default", ctx.DispatchedServiceTier)
	}
}

// Default off, and OpenRouter only: an arm without the key, or a configured
// arm on another backend, keeps the body it had before the key existed.
func TestServiceTierIsAbsentForUnconfiguredArmsAndOtherBackends(t *testing.T) {
	unconfigured := providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences())
	if raw := providerPinRouteBytes(t, unconfigured); bytes.Contains(raw, []byte("service_tier")) {
		t.Fatalf("an arm without a configured tier carried one:\n%s", raw)
	}
	elsewhere := providerPinRouter("http://127.0.0.1:8000/v1", providerPinPreferences())
	withArmServiceTier(elsewhere, providerPinArm, "default")
	if raw := providerPinRouteBytes(t, elsewhere); bytes.Contains(raw, []byte("service_tier")) {
		t.Fatalf("an OpenRouter-only tier reached another backend:\n%s", raw)
	}
}

// The configured tier wins over a client's, for configured arms only: the
// arm's price is defined for its tier. An unconfigured arm forwards the
// client's tier untouched.
func TestConfiguredServiceTierReplacesTheClientsOnlyOnConfiguredArms(t *testing.T) {
	router := providerPinRouter("https://openrouter.ai/api/v1", nil)
	dispatch := &providerDispatch{
		profile: &config.ProviderProfile{Type: "openrouter"}, logicalModel: providerPinArm,
	}
	clientBody := []byte(`{"model":"m","service_tier":"priority"}`)

	ctx := &RequestContext{}
	forwarded, err := applyServiceTier(clientBody, dispatch, router.Config, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(forwarded, clientBody) || ctx.ClientServiceTier != nil {
		t.Fatalf("an unconfigured arm changed the client's tier: %s", forwarded)
	}
	if got := dispatchedServiceTier(forwarded); got != "priority" {
		t.Fatalf("forwarded tier = %q, want the client's", got)
	}

	withArmServiceTier(router, providerPinArm, "default")
	ctx = &RequestContext{}
	replaced, err := applyServiceTier(clientBody, dispatch, router.Config, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got := dispatchedServiceTier(replaced); got != "default" {
		t.Fatalf("configured arm sent tier %q, want default:\n%s", got, replaced)
	}
	if string(ctx.ClientServiceTier) != `"priority"` {
		t.Fatalf("the client's replaced tier was not kept: %s", ctx.ClientServiceTier)
	}
}

// Every llm_usage line names the tier the dispatched body requested beside
// the tier the upstream served, and null when the body carried none.
func TestUsageLineNamesTheRequestedServiceTier(t *testing.T) {
	for name, test := range map[string]struct {
		tier string
		want interface{}
	}{
		"requested": {"default", "default"},
		"absent":    {"", nil},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			response := usageAttributionResponse("OpenAI", nil)
			response.UpstreamServiceTier = "flex"
			ctx := &RequestContext{
				RequestID: "rt_tier-requested", RequestModel: "openai/gpt-luna", SemanticResponse: response,
				DispatchedServiceTier: test.tier,
			}

			router.reportNonStreamingUsage(ctx, time.Second, router.takeNeutralResponseUsage(ctx))

			fields := findLogEvent(t, logs, "llm_usage")
			if value, present := fields["requested_service_tier"]; !present || value != test.want {
				t.Fatalf("requested_service_tier = %v (present %v), want %v", value, present, test.want)
			}
			if fields["upstream_service_tier"] != "flex" {
				t.Fatalf("upstream_service_tier = %v, want the served tier", fields["upstream_service_tier"])
			}
		})
	}
}

// The standard re-send carries the twin's configured tier, never the flex
// attempt's, and each usage line names the tier its own call requested. A
// twin without a tier gets the client's (here none) back, not flex.
func TestFlexResendCarriesTheTwinsServiceTier(t *testing.T) {
	for name, test := range map[string]struct {
		twinTier   string
		wantResent interface{}
	}{
		"twin configured":   {"default", "default"},
		"twin unconfigured": {"", nil},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
			router := flexResendRouter(twin.server.URL)
			withArmServiceTier(router, flexResendArm, "flex")
			withArmServiceTier(router, flexResendTwin, test.twinTier)
			ctx, flexBody := routeFlexTurn(t, router, flexResendArm, false)
			if got := dispatchedServiceTier(flexBody); got != "flex" {
				t.Fatalf("flex dispatch tier = %q, want flex:\n%s", got, flexBody)
			}

			answerHeaders(t, router, ctx, http.StatusTooManyRequests)

			calls := twin.calls()
			if len(calls) != 1 {
				t.Fatalf("the twin was called %d times, want once", len(calls))
			}
			resent := dispatchedServiceTier(calls[0].body)
			if want, _ := test.wantResent.(string); resent != want {
				t.Fatalf("re-send tier = %q, want %q:\n%s", resent, want, calls[0].body)
			}
			lines := usageLines(logs)
			if len(lines) != 2 {
				t.Fatalf("got %d llm_usage lines, want 2", len(lines))
			}
			if lines[0]["requested_service_tier"] != "flex" {
				t.Fatalf("flex attempt requested_service_tier = %v", lines[0]["requested_service_tier"])
			}
			if lines[1]["requested_service_tier"] != test.wantResent {
				t.Fatalf("re-send requested_service_tier = %v, want %v",
					lines[1]["requested_service_tier"], test.wantResent)
			}
		})
	}
}
