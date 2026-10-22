//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// The endpoint pick end to end through the extproc phases: an ARC turn on an
// arm whose pin reaches several of its model's endpoints in the routing-price
// feed is sent to the session's pick, and a failure before any byte is
// re-sent on the next endpoints of the rendezvous order, against a fake
// OpenRouter that answers per endpoint.

const (
	endpointPickArm      = "arc/model@thinking-off"
	endpointPickDecision = "rayline-arc-dev"
	endpointPickModelID  = "vendor/model"
	endpointPickKey      = "9b74c9897bac770ffc029102a200c5de"
	// endpointTransportFailure makes the fake upstream drop the connection
	// before any status.
	endpointTransportFailure = -1
)

// endpointPickTags are the model's endpoints in the feed; endpointPickSlugs
// is the arm's pin, whose bare slugs reach azure/global and
// google-vertex/global.
var (
	endpointPickTags  = []string{"amazon-bedrock", "anthropic", "azure/global", "claude-on-aws", "google-vertex/global"}
	endpointPickSlugs = []string{"amazon-bedrock", "anthropic", "azure", "claude-on-aws", "google-vertex"}
)

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
		if status == endpointTransportFailure {
			connection, _, _ := w.(http.Hijacker).Hijack()
			_ = connection.Close()
			return
		}
		w.Header().Set("content-type", "application/json")
		w.WriteHeader(status)
		if status != http.StatusOK {
			_, _ = fmt.Fprintf(w, `{"error":{"message":"endpoint unavailable","code":%d}}`, status)
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

// endpointPickRouter is a cell with one arm pinned to pin, with no
// fallbacks, on an OpenRouter backend at baseURL.
func endpointPickRouter(baseURL string, pin ...string) *OpenAIRouter {
	if len(pin) == 0 {
		pin = endpointPickSlugs
	}
	noFallbacks := false
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: endpointPickArm,
			ModelConfig: map[string]config.ModelParams{
				endpointPickArm: {
					PreferredEndpoints: []string{"openrouter"},
					APIFormat:          config.APIFormatOpenAI,
					ProviderPreferences: &config.OpenRouterProviderPreferences{
						Order: append([]string(nil), pin...), AllowFallbacks: &noFallbacks,
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

// endpointFeed is a routing-price cache whose feed carries the model's rows
// at tags, fetched once.
func endpointFeed(t *testing.T, tags ...string) *raylinearc.RoutingPriceCache {
	t.Helper()
	rows := make([]string, len(tags))
	for index, tag := range tags {
		rows[index] = fmt.Sprintf(`{"model": %q, "endpoint_tag": %q, "input": 3, "cache_read": 0.3,
			"cache_write": 3.75, "output": 15, "valid_from": null, "valid_until": null, "higher_tier": null}`,
			endpointPickModelID, tag)
	}
	body := `{"schema": "arc-routing-prices.v1", "store_version": "1", "as_of": "2026-10-14T16:00:00Z",
		"currency": "USD", "unit": "usd_per_million_tokens", "models": [` + strings.Join(rows, ",") + `]}`
	now := time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)
	cache := raylinearc.NewRoutingPriceCache(raylinearc.RoutingPriceCacheConfig{
		URL: newRoutingPriceFeed(t, []byte(body)).URL, TTL: time.Minute, RefreshAhead: 10 * time.Second,
		MaxAge: 10 * time.Minute, FetchTimeout: time.Second, Now: func() time.Time { return now },
	})
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	return cache
}

// endpointTurn is how one test turn is dispatched.
type endpointTurn struct {
	key     string
	privacy string
	feed    *raylinearc.RoutingPriceCache
	allowed privacyAllowedEndpoints
}

// routeEndpointTurn dispatches one ARC turn and returns its context and the
// body the provider hop was sent.
func routeEndpointTurn(t *testing.T, router *OpenAIRouter, turn endpointTurn) (*RequestContext, []byte) {
	t.Helper()
	request := testNeutralRequest("auto", "route this turn")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.StartTime = time.Now()
	ctx.VSRRaylineARC = &selection.RaylineARCTrace{EpisodeIDHash: "episode", SettlementPrices: turn.feed}
	ctx.RaylineARCSessionKey, ctx.RaylineARCSessionKeySource = turn.key, sessionKeySourceHeader
	ctx.RaylineARCPrivacyMode = turn.privacy
	ctx.RaylineARCPrivacyAllowedEndpoints = turn.allowed
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

func servedEndpointHeader(reply *ext_proc.ProcessingResponse) (string, bool) {
	for _, option := range reply.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRServedEndpoint {
			return string(option.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

// The candidates are the endpoints the pin's slugs reach in the feed, bare
// slugs included, and the turn is sent to the session's pick alone, with no
// OpenRouter fallback, so the endpoint that serves is the pick.
func TestARCTurnIsPinnedToTheSessionsReachableEndpoint(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	feed := endpointFeed(t, append([]string{"openai"}, endpointPickTags...)...)
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"), endpointTurn{key: endpointPickKey, feed: feed})

	if !slices.Equal(ctx.RaylineARCEndpointOrder, order) || ctx.RaylineARCDispatchedEndpoint != order[0] {
		t.Fatalf("order %v endpoint %q, want %v", ctx.RaylineARCEndpointOrder, ctx.RaylineARCDispatchedEndpoint, order)
	}
	if ctx.RaylineARCEndpointSource != endpointSourceFeed {
		t.Fatalf("endpoint_source = %q", ctx.RaylineARCEndpointSource)
	}
	want := mustJSON(t, map[string]interface{}{
		"order": []interface{}{order[0]}, "only": []interface{}{order[0]}, "allow_fallbacks": false,
	})
	if got, _ := json.Marshal(providerObject(t, body)); !sameJSON(t, got, want) {
		t.Fatalf("provider object %s, want %s", got, want)
	}
	if ctx.RaylineARCEndpointResend == nil {
		t.Fatal("the picked dispatch staged no re-send")
	}
}

// Without a routing-price feed the pin's strings are not endpoint records:
// the arm's pin goes out as configured, and says so.
func TestFeedlessCellSendsTheArmsPinUnpicked(t *testing.T) {
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"), endpointTurn{key: endpointPickKey})

	if ctx.RaylineARCEndpointOrder != nil || ctx.RaylineARCDispatchedEndpoint != "" || ctx.RaylineARCEndpointResend != nil {
		t.Fatalf("a feedless cell picked: %v %q", ctx.RaylineARCEndpointOrder, ctx.RaylineARCDispatchedEndpoint)
	}
	if ctx.RaylineARCEndpointSource != endpointSourceNone {
		t.Fatalf("endpoint_source = %q, want none", ctx.RaylineARCEndpointSource)
	}
	if order := providerObject(t, body)["order"].([]interface{}); len(order) != len(endpointPickSlugs) {
		t.Fatalf("the arm's order was changed: %v", order)
	}
}

// A turn with no session key keeps the arm's pin as configured, and names no
// endpoint when OpenRouter may choose among several.
func TestTurnWithoutSessionKeyKeepsTheArmsPin(t *testing.T) {
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"),
		endpointTurn{feed: endpointFeed(t, endpointPickTags...)})

	if ctx.RaylineARCEndpointOrder != nil || ctx.RaylineARCDispatchedEndpoint != "" || ctx.RaylineARCEndpointResend != nil {
		t.Fatalf("a keyless turn picked: %v %q", ctx.RaylineARCEndpointOrder, ctx.RaylineARCDispatchedEndpoint)
	}
	if order := providerObject(t, body)["order"].([]interface{}); len(order) != len(endpointPickSlugs) {
		t.Fatalf("the arm's order was changed: %v", order)
	}
}

// An arm sent with one configured endpoint tag is named; a single bare slug
// the feed says reaches several endpoints is not, unless the cell picked.
func TestSoleConfiguredEndpointIsNamedOnlyWhenItIsOne(t *testing.T) {
	for _, tc := range []struct {
		name string
		feed *raylinearc.RoutingPriceCache
		want string
	}{
		{"feedless single tag", nil, "google-vertex"},
		{"feed reaches one", endpointFeed(t, "google-vertex", "anthropic"), "google-vertex"},
		{"feed reaches several", endpointFeed(t, "google-vertex", "google-vertex/global"), ""},
	} {
		ctx, _ := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1", "google-vertex"), endpointTurn{feed: tc.feed})
		if ctx.RaylineARCDispatchedEndpoint != tc.want {
			t.Fatalf("%s: endpoint %q, want %q", tc.name, ctx.RaylineARCDispatchedEndpoint, tc.want)
		}
	}
}

// A private turn's candidates are kept to the endpoints its gateway allows
// the arm, and its pick keeps its data-use members on every endpoint.
func TestPrivateTurnPicksOnlyAllowedEndpoints(t *testing.T) {
	allowedTags := []string{"anthropic", "azure/global", "google-vertex/global"}
	order := raylinearc.PickEndpoint(endpointPickKey, allowedTags)
	upstream := newEndpointUpstream(t, map[string]int{order[0]: http.StatusServiceUnavailable})
	router := endpointPickRouter(upstream.server.URL)
	allowed := privacyAllowedEndpoints{present: true, arms: map[string][]string{endpointPickArm: allowedTags}}
	ctx, body := routeEndpointTurn(t, router, endpointTurn{
		key: endpointPickKey, privacy: raylineARCPrivacyModeZDR, feed: endpointFeed(t, endpointPickTags...), allowed: allowed,
	})

	if !slices.Equal(ctx.RaylineARCEndpointOrder, order) {
		t.Fatalf("order %v, want %v", ctx.RaylineARCEndpointOrder, order)
	}
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

// A private turn its gateway narrowed to one endpoint is pinned to it, even
// with no session key to pick by.
func TestPrivateTurnNarrowedToOneEndpointIsPinnedToIt(t *testing.T) {
	allowed := privacyAllowedEndpoints{present: true, arms: map[string][]string{endpointPickArm: {"anthropic"}}}
	ctx, body := routeEndpointTurn(t, endpointPickRouter("http://127.0.0.1:1"), endpointTurn{
		privacy: raylineARCPrivacyModeZDR, feed: endpointFeed(t, endpointPickTags...), allowed: allowed,
	})

	if pin := providerObject(t, body); !sameJSON(t, mustJSONRaw(t, pin["only"]), `["anthropic"]`) {
		t.Fatalf("provider object %v, want only anthropic", pin)
	}
	if ctx.RaylineARCDispatchedEndpoint != "anthropic" || ctx.RaylineARCEndpointResend != nil {
		t.Fatalf("endpoint %q resend %v", ctx.RaylineARCDispatchedEndpoint, ctx.RaylineARCEndpointResend)
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
	ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, endpointPickTags...)})

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
	if served, _ := servedEndpointHeader(reply); served != order[1] {
		t.Fatalf("%s = %q, want %q", headers.VSRServedEndpoint, served, order[1])
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want one per call: %v", len(lines), lines)
	}
	assertFields(t, "refused pick", lines[0], map[string]interface{}{
		"endpoint_resend": endpointResendAttempt, "provider_endpoint": order[0],
		"failure_class": turnFailureUpstream5xx, "usage_source": usageSourceUnknown,
		"session_key_source": sessionKeySourceHeader, "endpoint_source": endpointSourceFeed,
	})
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"endpoint_resend": endpointResendServed, "provider_endpoint": order[1],
	})
	if tried := lines[1]["provider_endpoints_tried"]; !sameJSON(t, mustJSONRaw(t, tried), mustJSON(t, order[:2])) {
		t.Fatalf("provider_endpoints_tried = %v, want %v", tried, order[:2])
	}
	event := findLogEvent(t, logs, "endpoint_resend")
	if statuses := mustJSONRaw(t, event["endpoint_statuses"]); !sameJSON(t, statuses, "[503,200]") {
		t.Fatalf("endpoint_statuses = %s", statuses)
	}
}

// A provider-side 404 and a dropped connection mid-chain move on to the next
// endpoint, as the gateway's retriesOnNextEndpoint does.
func TestEndpointResendMovesOnPastA404AndATransportFailure(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	for _, failure := range []int{http.StatusNotFound, endpointTransportFailure} {
		upstream := newEndpointUpstream(t, map[string]int{order[1]: failure})
		router := endpointPickRouter(upstream.server.URL)
		ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, endpointPickTags...)})

		reply := answerHeaders(t, router, ctx, http.StatusNotFound)

		if code := int(reply.GetImmediateResponse().GetStatus().GetCode()); code != http.StatusOK {
			t.Fatalf("failure %d: client status = %d, want the third endpoint's 200", failure, code)
		}
		if got := upstream.calls(); !slices.Equal(got, order[1:3]) {
			t.Fatalf("failure %d: re-sent on %v, want %v", failure, got, order[1:3])
		}
		if served, _ := servedEndpointHeader(reply); served != order[2] {
			t.Fatalf("failure %d: served endpoint %q, want %q", failure, served, order[2])
		}
	}
}

// At most two further endpoints are tried; the last one's failure is the
// client's, and names no served endpoint.
func TestEndpointResendStopsAfterTwoFurtherEndpoints(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	statuses := map[string]int{}
	for _, tag := range order {
		statuses[tag] = http.StatusServiceUnavailable
	}
	upstream := newEndpointUpstream(t, statuses)
	router := endpointPickRouter(upstream.server.URL)
	ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, endpointPickTags...)})

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
	if served, present := servedEndpointHeader(reply); present {
		t.Fatalf("a failed turn named a served endpoint: %q", served)
	}
}

// A request error is the request's: another endpoint would refuse it too, so
// it takes the ordinary path, as does a response that has begun.
func TestEndpointResendLeavesRequestErrorsAndServedResponsesAlone(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity, http.StatusOK} {
		upstream := newEndpointUpstream(t, map[string]int{})
		router := endpointPickRouter(upstream.server.URL)
		ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, endpointPickTags...)})

		reply := answerHeaders(t, router, ctx, status)

		if reply.GetImmediateResponse() != nil || len(upstream.calls()) != 0 {
			t.Fatalf("status %d was re-sent: %v", status, upstream.calls())
		}
	}
}

func TestEndpointResendTriggerIsTheGatewaysRetrySet(t *testing.T) {
	for status, retried := range map[int]bool{
		0: true, 401: true, 402: true, 403: true, 404: true, 408: true, 429: true, 500: true, 502: true, 503: true, 504: true,
		200: false, 204: false, 400: false, 413: false, 422: false,
	} {
		if got := endpointResendTrigger(status) != ""; got != retried {
			t.Errorf("status %d re-sent = %v, want %v", status, got, retried)
		}
	}
}

// The allowed-endpoints header lists each allowed arm's tags; anything
// malformed allows no arm, and it is read only beside the allowed arms on a
// private turn.
func TestPrivacyAllowedEndpointsParse(t *testing.T) {
	headersWith := func(endpoints string) map[string]string {
		return map[string]string{
			raylineARCPrivacyAllowedArmsHeader:      "arm-a,arm-b",
			raylineARCPrivacyAllowedEndpointsHeader: endpoints,
		}
	}
	parsed := raylineARCPrivacyAllowedEndpoints(raylineARCPrivacyModeZDR,
		headersWith(" arm-a = Anthropic|azure/global , arm-b=xai/zdr|xai/zdr "))
	want := map[string][]string{"arm-a": {"anthropic", "azure/global"}, "arm-b": {"xai/zdr"}}
	if !parsed.present || parsed.malformed || fmt.Sprint(parsed.arms) != fmt.Sprint(want) {
		t.Fatalf("parsed %+v, want %v", parsed, want)
	}
	for _, mode := range []string{raylineARCPrivacyModeZDR, raylineARCPrivacyModePrivacy} {
		for _, raw := range []string{
			"arm-a", "=anthropic", "arm-a=", "arm-a=anthropic|", "arm-a=|anthropic", "arm-a=a=b",
			"arm-a=anthropic,arm-a=xai", "arm-a=anthropic,", ",arm-a=anthropic",
		} {
			if got := raylineARCPrivacyAllowedEndpoints(mode, headersWith(raw)); !got.present || !got.malformed {
				t.Errorf("%s %q parsed as well-formed: %+v", mode, raw, got)
			}
		}
	}
	for name, signal := range map[string]map[string]string{
		"no allowed arms": {raylineARCPrivacyAllowedEndpointsHeader: "arm-a=anthropic"},
		"no endpoints":    {raylineARCPrivacyAllowedArmsHeader: "arm-a"},
	} {
		if got := raylineARCPrivacyAllowedEndpoints(raylineARCPrivacyModeZDR, signal); got.present {
			t.Errorf("%s: read %+v", name, got)
		}
	}
	if got := raylineARCPrivacyAllowedEndpoints("", headersWith("arm-a=anthropic")); got.present {
		t.Errorf("a standard turn read the endpoints: %+v", got)
	}
}

// The endpoint list folds into the privacy mask: malformed excludes every
// arm, an allowed arm left out is excluded, and an OpenRouter arm none of
// whose reachable endpoints is allowed is excluded once the feed is known.
func TestPrivacyAllowedEndpointsMaskArms(t *testing.T) {
	router := endpointPickRouter("http://127.0.0.1:1")
	refs := []config.ModelRef{{Model: endpointPickArm}, {Model: "arc/direct"}}
	allowedBoth := []string{endpointPickArm, "arc/direct"}
	mask := func(endpoints privacyAllowedEndpoints) ([]bool, []*raylinearc.ArmEndpoints) {
		return router.withPrivacyEndpointConstraints(
			privacyExcludedArms(raylineARCPrivacyModeZDR, allowedBoth, refs), endpoints, refs,
		)
	}

	if excluded, _ := mask(privacyAllowedEndpoints{present: true, malformed: true}); !slices.Equal(excluded, []bool{true, true}) {
		t.Fatalf("a malformed list left arms in: %v", excluded)
	}
	excluded, constraints := mask(privacyAllowedEndpoints{present: true, arms: map[string][]string{endpointPickArm: {"xai/zdr"}}})
	if !slices.Equal(excluded, []bool{false, true}) {
		t.Fatalf("an allowed arm the list leaves out was kept: %v", excluded)
	}
	if constraints[0] == nil || constraints[0].Model != endpointPickModelID || constraints[1] != nil {
		t.Fatalf("constraints %v", constraints)
	}
	if excluded, constraints := mask(privacyAllowedEndpoints{}); !slices.Equal(excluded, []bool{false, false}) || constraints != nil {
		t.Fatalf("an absent list changed the mask: %v %v", excluded, constraints)
	}

	feed := endpointFeed(t, endpointPickTags...)
	settlement, _ := feed.Settlement()
	for name, allowed := range map[string][]string{"empty intersection": {"xai/zdr"}, "reachable": {"anthropic"}} {
		excluded, constraints := mask(privacyAllowedEndpoints{present: true, arms: map[string][]string{
			endpointPickArm: allowed, "arc/direct": {"anything"},
		}})
		arcContext := &selection.RaylineARCSelectionContext{PrivacyExcludedArms: slices.Clone(excluded), PrivacyArmEndpoints: constraints}
		withPrivacyEndpointExclusions(arcContext, settlement)
		if want := name == "empty intersection"; arcContext.PrivacyExcludedArms[0] != want || arcContext.PrivacyExcludedArms[1] {
			t.Fatalf("%s: mask %v", name, arcContext.PrivacyExcludedArms)
		}
		// Without a feed nothing more is masked.
		arcContext = &selection.RaylineARCSelectionContext{PrivacyExcludedArms: slices.Clone(excluded), PrivacyArmEndpoints: constraints}
		withPrivacyEndpointExclusions(arcContext, nil)
		if arcContext.PrivacyExcludedArms[0] {
			t.Fatalf("%s: a feedless cell masked by endpoints", name)
		}
	}
}

// The header names the gateway's key; without it the cell derives the same
// digest from the episode's user and conversation, a scope or branch
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
		{nil, "uid-1:conv-1:arc-swift-1.0", want, sessionKeySourceFallback},
		{nil, "opaque", raylinearc.HashEpisodeID("opaque"), sessionKeySourceFallback},
		{nil, "", "", sessionKeySourceNone},
	} {
		key, source := raylineARCSessionKey(tc.headers, tc.episode)
		if key != tc.key || source != tc.source {
			t.Fatalf("%v %q: key %q source %q, want %q %q", tc.headers, tc.episode, key, source, tc.key, tc.source)
		}
	}
}

// x-rayline-session-key and x-rayline-privacy-allowed-endpoints are gateway
// headers: read only when the gateway is trusted, and never forwarded
// upstream.
func TestEndpointHeadersAreTrustedSignals(t *testing.T) {
	request := map[string]string{raylineARCSessionKeyHeader: "gw-key", raylineARCPrivacyAllowedEndpointsHeader: "a=b"}
	trusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}, request)
	if trusted[raylineARCSessionKeyHeader] != "gw-key" || trusted[raylineARCPrivacyAllowedEndpointsHeader] != "a=b" {
		t.Fatalf("trusted headers %v", trusted)
	}
	untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{}, request)
	if _, source := raylineARCSessionKey(untrusted, "uid-1:conv-1"); source != sessionKeySourceFallback {
		t.Fatalf("an untrusted session key was read: %s", source)
	}
	for _, header := range []string{raylineARCSessionKeyHeader, raylineARCPrivacyAllowedEndpointsHeader} {
		if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), header) {
			t.Fatalf("%s is forwarded upstream", header)
		}
	}
}

func mustJSONRaw(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	return json.RawMessage(mustJSON(t, value))
}
