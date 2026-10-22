//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// The endpoint pick end to end through the extproc phases: an ARC turn on an
// arm pinning several OpenRouter endpoints is sent to the session's pick, and
// a failure before any byte is re-sent on the next endpoints of the
// rendezvous order, against a fake OpenRouter that answers per endpoint.

const (
	endpointPickArm      = "arc/model@thinking-off"
	endpointPickDecision = "rayline-arc-dev"
	endpointPickModelID  = "vendor/model"
	endpointPickKey      = "9b74c9897bac770ffc029102a200c5de"
)

var endpointPickTags = []string{"amazon-bedrock", "anthropic", "azure/global", "claude-on-aws", "google-vertex/global"}

// endpointUpstream is a fake OpenRouter that answers each call by the one
// endpoint its provider object pins, and records the endpoints it was sent.
type endpointUpstream struct {
	server   *httptest.Server
	mu       sync.Mutex
	pinned   []string
	bodies   [][]byte
	statuses map[string]int
}

func newEndpointUpstream(t *testing.T, statuses map[string]int) *endpointUpstream {
	t.Helper()
	upstream := &endpointUpstream{statuses: statuses}
	upstream.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received, _ := io.ReadAll(r.Body)
		var wire struct {
			Provider struct {
				Order []string `json:"order"`
			} `json:"provider"`
		}
		_ = json.Unmarshal(received, &wire)
		tag := strings.Join(wire.Provider.Order, ",")
		upstream.mu.Lock()
		upstream.pinned = append(upstream.pinned, tag)
		upstream.bodies = append(upstream.bodies, received)
		status, ok := upstream.statuses[tag]
		upstream.mu.Unlock()
		if !ok {
			status = http.StatusOK
		}
		w.Header().Set("content-type", "application/json")
		w.WriteHeader(status)
		if status != http.StatusOK {
			_, _ = io.WriteString(w, `{"error":{"message":"endpoint unavailable","code":503}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"gen-1","object":"chat.completion","created":1,"model":"vendor/model",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"served on `+tag+`"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	}))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func (upstream *endpointUpstream) calls() []string {
	upstream.mu.Lock()
	defer upstream.mu.Unlock()
	return append([]string(nil), upstream.pinned...)
}

// endpointPickRouter is a cell with one arm pinning endpointPickTags, with no
// fallbacks, on an OpenRouter backend at baseURL.
func endpointPickRouter(baseURL string) *OpenAIRouter {
	noFallbacks := false
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: endpointPickArm,
			ModelConfig: map[string]config.ModelParams{
				endpointPickArm: {
					PreferredEndpoints: []string{"openrouter"},
					APIFormat:          config.APIFormatOpenAI,
					ProviderPreferences: &config.OpenRouterProviderPreferences{
						Order: append([]string(nil), endpointPickTags...), AllowFallbacks: &noFallbacks,
					},
					AccessKeys:       map[string]string{"openrouter": "sk-test"},
					ExternalModelIDs: map[string]string{"vllm": endpointPickModelID},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "openrouter", Address: "127.0.0.1", Port: 1, ProviderProfileName: "openrouter",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"openrouter": {Type: "openrouter", BaseURL: baseURL + "/api/v1"},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// routeEndpointTurn dispatches one ARC turn with the given session key and
// data mode, and returns its context and the body the provider hop was sent.
func routeEndpointTurn(t *testing.T, router *OpenAIRouter, key, privacyMode string) (*RequestContext, []byte) {
	t.Helper()
	request := testNeutralRequest("auto", "route this turn")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.StartTime = time.Now()
	ctx.VSRRaylineARC = &selection.RaylineARCTrace{EpisodeIDHash: "episode"}
	ctx.RaylineARCSessionKey, ctx.RaylineARCSessionKeySource = key, sessionKeySourceHeader
	ctx.RaylineARCPrivacyMode = privacyMode
	response, err := router.handleEntrypointModelRouting(
		request, "auto", endpointPickDecision, entropy.ReasoningDecision{}, endpointPickArm, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	return ctx, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func providerObject(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var wire struct {
		Provider map[string]interface{} `json:"provider"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode body: %v\n%s", err, body)
	}
	return wire.Provider
}

// An arm with several endpoints is sent to the session's pick alone, with no
// OpenRouter fallback, so the endpoint that serves is the pick.
func TestARCTurnIsPinnedToTheSessionsEndpoint(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"), endpointPickKey, "")

	if !slices.Equal(ctx.RaylineARCEndpointOrder, order) || ctx.RaylineARCDispatchedEndpoint != order[0] {
		t.Fatalf("order %v endpoint %q, want %v", ctx.RaylineARCEndpointOrder, ctx.RaylineARCDispatchedEndpoint, order)
	}
	pin := providerObject(t, body)
	want := map[string]interface{}{
		"order": []interface{}{order[0]}, "only": []interface{}{order[0]}, "allow_fallbacks": false,
	}
	if got, _ := json.Marshal(pin); !sameJSON(t, got, mustJSON(t, want)) {
		t.Fatalf("provider object %s, want %s", got, mustJSON(t, want))
	}
	if ctx.RaylineARCEndpointResend == nil {
		t.Fatal("the picked dispatch staged no re-send")
	}
}

// A turn with no session key keeps the arm's pin as configured.
func TestTurnWithoutSessionKeyKeepsTheArmsPin(t *testing.T) {
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"), "", "")

	if ctx.RaylineARCEndpointOrder != nil || ctx.RaylineARCDispatchedEndpoint != "" || ctx.RaylineARCEndpointResend != nil {
		t.Fatalf("a keyless turn picked: %v %q", ctx.RaylineARCEndpointOrder, ctx.RaylineARCDispatchedEndpoint)
	}
	if order := providerObject(t, body)["order"].([]interface{}); len(order) != len(endpointPickTags) {
		t.Fatalf("the arm's order was changed: %v", order)
	}
}

// A private turn's pick keeps its data-use members on every endpoint.
func TestPrivateTurnPickKeepsItsDataMode(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newEndpointUpstream(t, map[string]int{order[0]: http.StatusServiceUnavailable})
	router := endpointPickRouter(upstream.server.URL)
	ctx, body := routeEndpointTurn(t, router, endpointPickKey, raylineARCPrivacyModeZDR)

	answerHeaders(t, router, ctx, http.StatusServiceUnavailable)
	for _, sent := range [][]byte{body, upstream.bodies[0]} {
		pin := providerObject(t, sent)
		if pin["data_collection"] != "deny" || pin["zdr"] != true || pin["allow_fallbacks"] != false {
			t.Fatalf("a private turn's endpoint lost its data mode: %v", pin)
		}
	}
	if got := upstream.calls(); !slices.Equal(got, order[1:2]) {
		t.Fatalf("re-sent on %v, want %v", got, order[1:2])
	}
}

// A picked endpoint's failure before any byte is re-sent on the next
// endpoint, which serves the client; the served endpoint is the header the
// gateway bills by, and each call writes its own usage line.
func TestPickedEndpointFailureIsResentOnTheNextEndpoint(t *testing.T) {
	logs := captureLogs(t)
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newEndpointUpstream(t, map[string]int{})
	router := endpointPickRouter(upstream.server.URL)
	ctx, _ := routeEndpointTurn(t, router, endpointPickKey, "")

	reply := answerHeaders(t, router, ctx, http.StatusServiceUnavailable)

	immediate := reply.GetImmediateResponse()
	if immediate == nil || int(immediate.GetStatus().GetCode()) != http.StatusOK {
		t.Fatalf("the failure was not answered by the next endpoint: %v", reply)
	}
	if !strings.Contains(string(immediate.GetBody()), "served on "+order[1]) {
		t.Fatalf("the client did not get the next endpoint's reply:\n%s", immediate.GetBody())
	}
	if got := upstream.calls(); !slices.Equal(got, order[1:2]) {
		t.Fatalf("re-sent on %v, want %v", got, order[1:2])
	}
	served := ""
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRServedEndpoint {
			served = string(option.GetHeader().GetRawValue())
		}
	}
	if served != order[1] {
		t.Fatalf("%s = %q, want %q", headers.VSRServedEndpoint, served, order[1])
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want one per call: %v", len(lines), lines)
	}
	assertFields(t, "refused pick", lines[0], map[string]interface{}{
		"endpoint_resend": endpointResendAttempt, "provider_endpoint": order[0],
		"failure_class": turnFailureUpstream5xx, "usage_source": usageSourceUnknown,
		"session_key_source": sessionKeySourceHeader,
	})
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"endpoint_resend": endpointResendServed, "provider_endpoint": order[1],
	})
	if tried := lines[1]["provider_endpoints_tried"]; !sameJSON(t, json.RawMessage(mustJSON(t, tried)), mustJSON(t, order[:2])) {
		t.Fatalf("provider_endpoints_tried = %v, want %v", tried, order[:2])
	}
	event := findLogEvent(t, logs, "endpoint_resend")
	if statuses := mustJSON(t, event["endpoint_statuses"]); !sameJSON(t, json.RawMessage(statuses), "[503,200]") {
		t.Fatalf("endpoint_statuses = %s", statuses)
	}
}

// At most two further endpoints are tried; the last one's failure is the
// client's.
func TestEndpointResendStopsAfterTwoFurtherEndpoints(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	statuses := map[string]int{}
	for _, tag := range order {
		statuses[tag] = http.StatusServiceUnavailable
	}
	upstream := newEndpointUpstream(t, statuses)
	router := endpointPickRouter(upstream.server.URL)
	ctx, _ := routeEndpointTurn(t, router, endpointPickKey, "")

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	if code := int(reply.GetImmediateResponse().GetStatus().GetCode()); code != http.StatusServiceUnavailable {
		t.Fatalf("client status = %d, want the last endpoint's 503", code)
	}
	if got := upstream.calls(); !slices.Equal(got, order[1:3]) {
		t.Fatalf("re-sent on %v, want %v", got, order[1:3])
	}
	if !slices.Equal(ctx.RaylineARCEndpointsTried, order[:3]) {
		t.Fatalf("tried %v, want %v", ctx.RaylineARCEndpointsTried, order[:3])
	}
}

// A request error is the request's: another endpoint would refuse it too, so
// it takes the ordinary path, as does a response that has begun.
func TestEndpointResendLeavesRequestErrorsAndServedResponsesAlone(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusOK} {
		upstream := newEndpointUpstream(t, map[string]int{})
		router := endpointPickRouter(upstream.server.URL)
		ctx, _ := routeEndpointTurn(t, router, endpointPickKey, "")

		reply := answerHeaders(t, router, ctx, status)

		if reply.GetImmediateResponse() != nil || len(upstream.calls()) != 0 {
			t.Fatalf("status %d was re-sent: %v", status, upstream.calls())
		}
	}
}

// The header names the gateway's key; without it the cell derives the same
// digest from the episode's user and conversation, a subagent's branch
// dropped, and says so.
func TestSessionKeyFallsBackToTheEpisodesConversation(t *testing.T) {
	want := raylinearc.SessionKeyDigest("uid-1", "conv-1")
	for _, tc := range []struct {
		headers map[string]string
		episode string
		key     string
		source  string
	}{
		{map[string]string{raylineARCSessionKeyHeader: " gw-key "}, "uid-1:conv-1", "gw-key", sessionKeySourceHeader},
		{nil, "uid-1:conv-1", want, sessionKeySourceFallback},
		{map[string]string{}, "uid-1:conv-1:agent-7", want, sessionKeySourceFallback},
		{nil, "opaque", raylinearc.HashEpisodeID("opaque"), sessionKeySourceFallback},
		{nil, "", "", sessionKeySourceNone},
	} {
		key, source := raylineARCSessionKey(tc.headers, tc.episode)
		if key != tc.key || source != tc.source {
			t.Fatalf("%v %q: key %q source %q, want %q %q", tc.headers, tc.episode, key, source, tc.key, tc.source)
		}
	}
}

// x-rayline-session-key is a gateway header: read only when the gateway is
// trusted, and never forwarded upstream.
func TestSessionKeyHeaderIsATrustedSignal(t *testing.T) {
	request := map[string]string{raylineARCSessionKeyHeader: "gw-key"}
	trusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}, request)
	if trusted[raylineARCSessionKeyHeader] != "gw-key" {
		t.Fatalf("trusted headers %v", trusted)
	}
	untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{}, request)
	if _, source := raylineARCSessionKey(untrusted, "uid-1:conv-1"); source != sessionKeySourceFallback {
		t.Fatalf("an untrusted session key was read: %s", source)
	}
	if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), raylineARCSessionKeyHeader) {
		t.Fatal("x-rayline-session-key is forwarded upstream")
	}
}

// The endpoints an arm may be picked among are its pin's, less what it rules out.
func TestArmEndpointsReadThePin(t *testing.T) {
	pin := &config.OpenRouterProviderPreferences{
		Order: []string{"a", "b", "a", " ", "c", "d"}, Only: []string{"a", "b", "d"}, Ignore: []string{"d"},
	}
	if got := armEndpoints(pin); !slices.Equal(got, []string{"a", "b"}) {
		t.Fatalf("endpoints %v", got)
	}
	if got := armEndpoints(&config.OpenRouterProviderPreferences{Only: []string{"x", "y"}}); !slices.Equal(got, []string{"x", "y"}) {
		t.Fatalf("only-list endpoints %v", got)
	}
}
