//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// The ARC Contributor (formerly Flex) re-send, end to end through the
// extproc phases: a flex arm is routed and dispatched, the cell's provider
// hop answers with a 429 or a pre-byte timeout, and the standard twin -- a
// fake OpenRouter here -- is called once with the same body under the
// standard provider object.

const (
	flexResendArm      = "luna-flex"
	flexResendTwin     = "luna"
	flexResendDecision = "arc-contributor"
	flexResendModelID  = "openai/gpt-luna"
)

const flexTwinChatReply = `{"id":"gen-1","object":"chat.completion","created":1,"model":"openai/gpt-luna",` +
	`"provider":"OpenAI","choices":[{"index":0,"message":{"role":"assistant","content":"served on standard"},` +
	`"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}`

type flexTwinRequest struct {
	path    string
	headers http.Header
	body    []byte
}

// flexTwin is a fake OpenRouter: it records every call and answers each with
// the configured status, content type and body.
type flexTwin struct {
	server      *httptest.Server
	mu          sync.Mutex
	requests    []flexTwinRequest
	status      int
	contentType string
	body        string
}

func newFlexTwin(t *testing.T, status int, contentType, body string) *flexTwin {
	t.Helper()
	twin := &flexTwin{status: status, contentType: contentType, body: body}
	twin.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received, _ := io.ReadAll(r.Body)
		twin.mu.Lock()
		twin.requests = append(twin.requests, flexTwinRequest{path: r.URL.Path, headers: r.Header.Clone(), body: received})
		twin.mu.Unlock()
		w.Header().Set("content-type", twin.contentType)
		w.WriteHeader(twin.status)
		_, _ = io.WriteString(w, twin.body)
	}))
	t.Cleanup(twin.server.Close)
	return twin
}

func (twin *flexTwin) calls() []flexTwinRequest {
	twin.mu.Lock()
	defer twin.mu.Unlock()
	return append([]flexTwinRequest(nil), twin.requests...)
}

// flexResendRouter is a cell with a flex arm pinned to openai/flex, its
// standard twin ignoring openai/flex, and a plain arm with no twin, all on one
// OpenRouter backend at baseURL. The twin is priced at twice the flex rate.
func flexResendRouter(baseURL string) *OpenAIRouter {
	noFallbacks := false
	arm := func(pin *config.OpenRouterProviderPreferences, twin, key string, prompt, completion float64) config.ModelParams {
		return config.ModelParams{
			PreferredEndpoints:  []string{"openrouter"},
			APIFormat:           config.APIFormatOpenAI,
			ProviderPreferences: pin,
			StandardTwin:        twin,
			AccessKeys:          map[string]string{"openrouter": key},
			// Keyed by endpoint type; an untyped endpoint is "vllm".
			ExternalModelIDs: map[string]string{"vllm": flexResendModelID},
			Pricing:          config.ModelPricing{Currency: "USD", PromptPer1M: prompt, CompletionPer1M: completion},
		}
	}
	flexPin := &config.OpenRouterProviderPreferences{Only: []string{"openai/flex"}, AllowFallbacks: &noFallbacks}
	standardPin := &config.OpenRouterProviderPreferences{
		Order: []string{"openai"}, AllowFallbacks: &noFallbacks, Ignore: []string{"openai/flex"},
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: flexResendArm,
			ModelConfig: map[string]config.ModelParams{
				flexResendArm:  arm(flexPin, flexResendTwin, "sk-flex-test", 1, 2),
				flexResendTwin: arm(standardPin, "", "sk-standard-test", 2, 4),
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "openrouter", Address: "127.0.0.1", Port: 1,
				ProviderProfileName: "openrouter",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"openrouter": {Type: "openrouter", BaseURL: baseURL + "/api/v1"},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// routeFlexTurn dispatches one turn to arm and returns the request context
// and the body the cell's provider hop was sent.
func routeFlexTurn(t *testing.T, router *OpenAIRouter, arm string, stream bool) (*RequestContext, []byte) {
	t.Helper()
	request := testNeutralRequest("auto", "route this turn")
	request.Stream = stream
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ExpectStreamingResponse = stream
	ctx.StartTime = time.Now()
	response, err := router.handleEntrypointModelRouting(
		request, "auto", flexResendDecision, entropy.ReasoningDecision{}, arm, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	return ctx, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func flexUpstreamHeaders(status int, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte(strconv.Itoa(status))},
			{Key: "content-type", RawValue: []byte(contentType)},
		}},
	}}
}

// answerHeaders delivers the cell's provider-hop response headers and returns
// what the Router answered.
func answerHeaders(t *testing.T, router *OpenAIRouter, ctx *RequestContext, status int) *ext_proc.ProcessingResponse {
	t.Helper()
	stream := NewMockStream(nil)
	if err := router.processResponseHeaders(stream, flexUpstreamHeaders(status, "application/json"), ctx); err != nil {
		t.Fatalf("processResponseHeaders: %v", err)
	}
	if len(stream.Responses) != 1 {
		t.Fatalf("the Router sent %d replies, want 1", len(stream.Responses))
	}
	return stream.Responses[0]
}

func usageLines(logs *observer.ObservedLogs) []map[string]interface{} {
	var lines []map[string]interface{}
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if name, _ := fields["event"].(string); name == "llm_usage" {
			lines = append(lines, fields)
		}
	}
	return lines
}

// withoutProvider decodes a body and drops its provider object, so two bodies
// can be compared on everything else.
func withoutProvider(t *testing.T, body []byte) (map[string]json.RawMessage, json.RawMessage) {
	t.Helper()
	var members map[string]json.RawMessage
	if err := json.Unmarshal(body, &members); err != nil {
		t.Fatalf("decode body: %v\n%s", err, body)
	}
	provider := members["provider"]
	delete(members, "provider")
	return members, provider
}

// A flex 429 is re-sent once on the twin: the same body byte for byte except
// the provider object, the twin's credential and path, and the twin's reply
// reaches the client. The refused attempt and the re-send each write a usage
// line, and the refused attempt is never a free call.
func TestFlex429IsResentOnceOnTheStandardTwin(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, flexBody := routeFlexTurn(t, router, flexResendArm, false)
	if ctx.FlexResend == nil {
		t.Fatalf("the flex arm's dispatch staged no re-send: %v", findLogEvent(t, logs, "flex_resend_unavailable"))
	}

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the 429 was not answered by the re-send: %v", reply)
	}
	if code := immediate.GetStatus().GetCode(); int(code) != http.StatusOK {
		t.Fatalf("client status = %d, want the twin's 200", code)
	}
	if !strings.Contains(string(immediate.GetBody()), "served on standard") {
		t.Fatalf("the client did not get the twin's reply:\n%s", immediate.GetBody())
	}

	calls := twin.calls()
	if len(calls) != 1 {
		t.Fatalf("the twin was called %d times, want exactly once", len(calls))
	}
	call := calls[0]
	if call.path != "/api/v1/chat/completions" {
		t.Fatalf("re-send path = %q", call.path)
	}
	// The twin's own credential, as its own dispatch would send.
	if got := call.headers.Get("Authorization"); got != "Bearer sk-standard-test" {
		t.Fatalf("re-send credential = %q", got)
	}
	flexMembers, flexProvider := withoutProvider(t, flexBody)
	twinMembers, twinProvider := withoutProvider(t, call.body)
	if !sameJSON(t, flexProvider, `{"only":["openai/flex"],"allow_fallbacks":false}`) {
		t.Fatalf("the flex dispatch was not pinned to flex: %s", flexProvider)
	}
	if !sameJSON(t, twinProvider, `{"order":["openai"],"allow_fallbacks":false,"ignore":["openai/flex"]}`) {
		t.Fatalf("the re-send did not carry the standard provider object: %s", twinProvider)
	}
	if len(flexMembers) != len(twinMembers) {
		t.Fatalf("the re-send changed the body's members:\nflex %s\ntwin %s", flexBody, call.body)
	}
	for name, value := range flexMembers {
		if !sameJSON(t, twinMembers[name], string(value)) {
			t.Fatalf("the re-send changed %q: flex %s, twin %s", name, value, twinMembers[name])
		}
	}
	if !sameJSON(t, twinMembers["model"], `"`+flexResendModelID+`"`) {
		t.Fatalf("the re-send changed the model: %s", twinMembers["model"])
	}

	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want one per attempt: %v", len(lines), lines)
	}
	attempt, resend := lines[0], lines[1]
	assertFields(t, "flex attempt", attempt, map[string]interface{}{
		"flex_resend":   flexResendAttempt,
		"model":         flexResendArm,
		"failure_class": turnFailureRateLimited,
		"usage_source":  usageSourceUnknown,
		"pricing":       usagePricingNoUsage,
	})
	if attempt["cost"] != nil || attempt["prompt_tokens"] != nil {
		t.Fatalf("the refused attempt was settled with counts or a cost nobody stated: %v", attempt)
	}
	assertFields(t, "standard re-send", resend, map[string]interface{}{
		"flex_resend":            flexResendStandard,
		"model":                  flexResendTwin,
		"pricing":                usagePricingPriced,
		"pricing_prompt_per_1m":  2.0,
		"provider_reported_cost": 0.00004,
		"upstream_attempts":      1,
	})
	if resend["failure_class"] != nil {
		t.Fatalf("the served re-send carried a failure class: %v", resend)
	}

	event := findLogEvent(t, logs, "flex_resend")
	assertFields(t, "flex_resend event", event, map[string]interface{}{
		"trigger":         turnFailureRateLimited,
		"flex_model":      flexResendArm,
		"standard_model":  flexResendTwin,
		"flex_status":     http.StatusTooManyRequests,
		"standard_status": http.StatusOK,
	})
}

// A timeout before any response byte (Envoy's 504 at the headers) is the
// other trigger. A twin that refuses too ends the turn with its refusal: there
// is no second re-send.
func TestFlexPreByteTimeoutIsResentOnceAndNeverAgain(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusTooManyRequests, "application/json",
		`{"error":{"message":"Rate limit exceeded","code":429}}`)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)

	reply := answerHeaders(t, router, ctx, http.StatusGatewayTimeout)

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the timeout was not answered by the re-send: %v", reply)
	}
	if code := immediate.GetStatus().GetCode(); int(code) != http.StatusTooManyRequests {
		t.Fatalf("client status = %d, want the twin's 429", code)
	}
	if calls := twin.calls(); len(calls) != 1 {
		t.Fatalf("the twin was called %d times, want exactly once", len(calls))
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "flex attempt", lines[0], map[string]interface{}{
		"flex_resend": flexResendAttempt, "failure_class": turnFailureTimeout,
		"usage_source": usageSourceUnknown, "pricing": usagePricingNoUsage,
	})
	if lines[0]["cost"] != nil {
		t.Fatalf("a timed-out flex attempt was settled as free: %v", lines[0])
	}
	assertFields(t, "standard re-send", lines[1], map[string]interface{}{
		"flex_resend": flexResendStandard, "failure_class": turnFailureRateLimited,
		"usage_source": usageSourceUnknown, "model": flexResendTwin,
	})
}

// A streamed turn's re-send streams from the twin and reaches the client as
// the whole event stream, settled from the stream's own usage.
func TestFlexResendOfAStreamedTurnServesTheTwinsStream(t *testing.T) {
	logs := captureLogs(t)
	sse := `data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"streamed on standard"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}` + "\n\n" +
		"data: [DONE]\n\n"
	twin := newFlexTwin(t, http.StatusOK, "text/event-stream", sse)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, true)

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the streamed 429 was not answered by the re-send: %v", reply)
	}
	if !strings.Contains(string(immediate.GetBody()), "streamed on standard") {
		t.Fatalf("the client did not get the twin's stream:\n%s", immediate.GetBody())
	}
	if contentType := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["content-type"]; !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("the re-sent stream travelled as %q", contentType)
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "standard re-send", lines[1], map[string]interface{}{
		"flex_resend": flexResendStandard, "streaming": true, "provider_reported_cost": 0.00004,
	})
}

// Default off: an arm without a twin answers a 429 exactly as before, and
// nothing is dialled.
func TestArmWithoutATwinIsNeverResent(t *testing.T) {
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)
	if ctx.FlexResend != nil {
		t.Fatal("an arm without a twin staged a re-send")
	}

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	if reply.GetImmediateResponse() != nil {
		t.Fatalf("a 429 on an arm without a twin was replaced: %v", reply)
	}
	if calls := twin.calls(); len(calls) != 0 {
		t.Fatalf("an arm without a twin dialled %d re-sends", len(calls))
	}
}

// Only a refusal or a pre-byte timeout re-sends. A flex call that completes
// with a 200 is served as it is, and one that states no usage is settled as
// unknown, never as a free call.
func TestFlex200WithoutUsageSettlesUnknownAndIsNotResent(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)

	if reply := answerHeaders(t, router, ctx, http.StatusOK); reply.GetImmediateResponse() != nil {
		t.Fatalf("a flex 200 was replaced: %v", reply)
	}
	noUsage := `{"id":"gen-3","object":"chat.completion","created":1,"model":"openai/gpt-luna",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"served on flex"},"finish_reason":"stop"}]}`
	stream := NewMockStream(nil)
	if err := router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(noUsage), EndOfStream: true},
	}, ctx); err != nil {
		t.Fatalf("processResponseBody: %v", err)
	}

	if calls := twin.calls(); len(calls) != 0 {
		t.Fatalf("a flex 200 dialled %d re-sends", len(calls))
	}
	lines := usageLines(logs)
	if len(lines) != 1 {
		t.Fatalf("got %d llm_usage lines, want 1: %v", len(lines), lines)
	}
	assertFields(t, "flex 200 without usage", lines[0], map[string]interface{}{
		"model": flexResendArm, "usage_source": usageSourceUnknown, "pricing": usagePricingNoUsage,
	})
	if lines[0]["cost"] != nil || lines[0]["flex_resend"] != nil {
		t.Fatalf("a usage-less flex 200 was settled with a cost or marked re-sent: %v", lines[0])
	}
}

// A twin that cannot be reached still ends the turn once: the client gets a
// 502 in its own format, and the re-send's line records the failure.
func TestFlexResendToAnUnreachableTwinFailsOnce(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)
	twin.server.Close()

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the 429 was not answered: %v", reply)
	}
	if code := immediate.GetStatus().GetCode(); int(code) != http.StatusBadGateway {
		t.Fatalf("client status = %d, want 502", code)
	}
	if !json.Valid(immediate.GetBody()) {
		t.Fatalf("the client was not told why in its own format:\n%s", immediate.GetBody())
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "standard re-send", lines[1], map[string]interface{}{
		"flex_resend": flexResendStandard, "failure_class": turnFailureUpstream5xx,
		"usage_source": usageSourceUnknown,
	})
	if event := findLogEvent(t, logs, "flex_resend"); event["standard_error"] == nil {
		t.Fatalf("the flex_resend event did not say why the re-send failed: %v", event)
	}
}

// The re-send must answer inside Envoy's ext_proc message_timeout, which the
// Router cannot read: the budget is stated, 55 s by default (below the 60 s
// the shipped Gateway API manifests set), and a negative one turns it off.
func TestFlexResendBudgetIsStatedAndConservative(t *testing.T) {
	router := flexResendRouter("http://127.0.0.1:1")
	if budget, skipped := router.flexResendBudget(); budget != 55*time.Second || skipped != "" {
		t.Fatalf("default budget = %v (%q), want 55s", budget, skipped)
	}
	router.Config.FlexResendBudgetSec = 20
	if budget, skipped := router.flexResendBudget(); budget != 20*time.Second || skipped != "" {
		t.Fatalf("configured budget = %v (%q), want 20s", budget, skipped)
	}
	router.Config.FlexResendBudgetSec = 2
	if _, skipped := router.flexResendBudget(); skipped != "budget_too_short" {
		t.Fatalf("a 2s budget was not skipped: %q", skipped)
	}
	router.Config.FlexResendBudgetSec = -1
	if _, skipped := router.flexResendBudget(); skipped != "disabled" {
		t.Fatalf("a negative budget did not turn the re-send off: %q", skipped)
	}
}

// A budget too short to answer in is not spent: the twin is never dialled,
// the 429 takes the ordinary path, and it settles there as unknown.
func TestFlexResendWithTooShortABudgetFallsBackToTheRefusal(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	router.Config.FlexResendBudgetSec = 2
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)

	if reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests); reply.GetImmediateResponse() != nil {
		t.Fatalf("a re-send ran on a budget too short for it: %v", reply)
	}
	stream := NewMockStream(nil)
	if err := router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(`{"error":{"message":"Rate limit exceeded","code":429}}`), EndOfStream: true},
	}, ctx); err != nil {
		t.Fatalf("processResponseBody: %v", err)
	}

	if calls := twin.calls(); len(calls) != 0 {
		t.Fatalf("the twin was dialled %d times on a skipped re-send", len(calls))
	}
	lines := usageLines(logs)
	if len(lines) != 1 {
		t.Fatalf("got %d llm_usage lines, want the refusal's one: %v", len(lines), lines)
	}
	assertFields(t, "refused flex call", lines[0], map[string]interface{}{
		"model": flexResendArm, "failure_class": turnFailureRateLimited,
		"usage_source": usageSourceUnknown, "pricing": usagePricingNoUsage,
	})
	if lines[0]["cost"] != nil {
		t.Fatalf("the refused call was settled with a cost: %v", lines[0])
	}
	findLogEvent(t, logs, "flex_resend_skipped")
}

// A re-sent stream whose body read fails after its usage frame is cut, not
// served: the usage it reported is kept, the turn is classed as a failure,
// and the selection is not committed as a success.
func TestFlexResendStreamCutAfterUsageIsNotCommitted(t *testing.T) {
	logs := captureLogs(t)
	frames := `data: {"id":"gen-4","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"cut on standard"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"gen-4","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}` + "\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Error("the test server cannot cut a connection")
			return
		}
		conn, buffered, err := hijacker.Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		// A chunked 200 that stops after the usage frame, with no final chunk.
		_, _ = buffered.WriteString("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
		_, _ = buffered.WriteString(strconv.FormatInt(int64(len(frames)), 16) + "\r\n" + frames + "\r\n")
		_ = buffered.Flush()
		_ = conn.Close()
	}))
	t.Cleanup(server.Close)
	router := flexResendRouter(server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, true)

	reply := answerHeaders(t, router, ctx, http.StatusTooManyRequests)

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the 429 was not answered: %v", reply)
	}
	if !strings.Contains(string(immediate.GetBody()), "error") {
		t.Fatalf("the cut stream did not end with an error:\n%s", immediate.GetBody())
	}
	// An aborted stream is what keeps the turn from committing: the
	// selection commits only a stream that was not aborted, and the cut
	// releases it (releaseCutTurn).
	if ctx.ResponseFailureClass != turnFailureUpstreamError || ctx.ResponseFailureDetail != "standard_resend_body_cut" {
		t.Fatalf("a cut re-send was classed %q (%q), want upstream_error standard_resend_body_cut",
			ctx.ResponseFailureClass, ctx.ResponseFailureDetail)
	}
	if !ctx.StreamingAborted {
		t.Fatal("a cut re-send was settled as a complete stream")
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "cut re-send", lines[1], map[string]interface{}{
		"flex_resend": flexResendStandard, "truncated": true, "provider_reported_cost": 0.00004,
	})
	if lines[1]["failure_class"] == nil {
		t.Fatalf("the cut re-send's line carries no failure class: %v", lines[1])
	}
}

// A 200 whose body read fails before any byte is the twin's failure, not a
// served empty reply.
func TestFlexResendNonStreamReadErrorIsAFailure(t *testing.T) {
	logs := captureLogs(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hijacker, _ := w.(http.Hijacker)
		conn, buffered, err := hijacker.Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		_, _ = buffered.WriteString("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 400\r\n\r\n{\"id\":")
		_ = buffered.Flush()
		_ = conn.Close()
	}))
	t.Cleanup(server.Close)
	router := flexResendRouter(server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)

	immediate := answerHeaders(t, router, ctx, http.StatusTooManyRequests).GetImmediateResponse()
	if immediate == nil {
		t.Fatal("the 429 was not answered")
	}
	if code := int(immediate.GetStatus().GetCode()); code < 500 {
		t.Fatalf("a twin body cut short was served with status %d", code)
	}
	lines := usageLines(logs)
	if len(lines) != 2 || lines[1]["failure_class"] == nil {
		t.Fatalf("the cut re-send was not settled as a failure: %v", lines)
	}
}

// sameJSON compares two JSON values by meaning, not by member order.
func sameJSON(t *testing.T, got json.RawMessage, want string) bool {
	t.Helper()
	var left, right interface{}
	if err := json.Unmarshal(got, &left); err != nil {
		return false
	}
	if err := json.Unmarshal([]byte(want), &right); err != nil {
		t.Fatalf("decode expected JSON: %v", err)
	}
	return reflect.DeepEqual(left, right)
}

func assertFields(t *testing.T, label string, fields map[string]interface{}, want map[string]interface{}) {
	t.Helper()
	for name, value := range want {
		if !looselyEqual(fields[name], value) {
			t.Fatalf("%s: %s = %#v, want %#v\n%v", label, name, fields[name], value, fields)
		}
	}
}

// looselyEqual compares a logged value with an expected one across the
// numeric types a structured log may hold.
func looselyEqual(got, want interface{}) bool {
	toFloat := func(value interface{}) (float64, bool) {
		switch typed := value.(type) {
		case int:
			return float64(typed), true
		case int64:
			return float64(typed), true
		case float64:
			return typed, true
		}
		return 0, false
	}
	if left, ok := toFloat(got); ok {
		right, ok := toFloat(want)
		return ok && left == right
	}
	return got == want
}
