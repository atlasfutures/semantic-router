package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func turnTimingEvents(logs *observer.ObservedLogs) []map[string]interface{} {
	var events []map[string]interface{}
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if fields["event"] == "rayline_arc_turn_timing" {
			events = append(events, fields)
		}
	}
	return events
}

func oneTurnTimingEvent(t *testing.T, logs *observer.ObservedLogs) map[string]interface{} {
	t.Helper()
	events := turnTimingEvents(logs)
	if len(events) != 1 {
		t.Fatalf("%d rayline_arc_turn_timing events, want 1: %v", len(events), events)
	}
	return events[0]
}

func requireTurnTimingFields(t *testing.T, fields map[string]interface{}, names ...string) {
	t.Helper()
	for _, name := range names {
		if _, ok := fields[name]; !ok {
			t.Errorf("%s missing from %v", name, fields)
		}
	}
}

// runPolicyTurn runs one Messages turn through the request phases and
// returns the immediate response, nil when the turn was routed.
func runPolicyTurn(t *testing.T, router *OpenAIRouter, episode, client string) *ext_proc.ImmediateResponse {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "timing-request", StartTime: time.Now(), TraceContext: context.Background()}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	return response.GetImmediateResponse()
}

// A hosted policy turn logs one rayline_arc_turn_timing line, with its
// cell-side spans and the policy call's HTTP trace, names and numbers only;
// a turn the service refuses still logs one, marked failed.
func TestPolicyTurnLogsOneTurnTimingLine(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, ""),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["think"].ActionID })
	const content = "fix the failing test in the timing secret-turn-text"
	client := `{"model":"auto","max_tokens":32000,"messages":[{"role":"user","content":"` + content + `"}]}`

	logs := captureLogs(t)
	if immediate := runPolicyTurn(t, router, "episode-timing-raw-id", client); immediate != nil {
		t.Fatalf("turn refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	fields := oneTurnTimingEvent(t, logs)
	requireTurnTimingFields(t, fields,
		"request_id", "episode_id_hash", "session_action", "headers_to_body_ms", "body_bytes", "parse_ms",
		"episode_load_ms", "decide_prepare_ms", "decide_marshal_ms", "decide_build_ms", "decide_request_bytes",
		"wrote_request_ms", "first_byte_ms", "http_ms", "decode_ms", "policy_ms", "reused_conn", "total_ms")
	if fields["surface"] != raylineARCTurnSurfaceHosted || fields["outcome"] != "selected" ||
		fields["request_id"] != "timing-request" || fields["policy_calls"] != int64(1) ||
		fields["body_bytes"] != int64(len(client)) || fields["policy_operation"] != "decide" {
		t.Fatalf("fields = %v", fields)
	}
	if fields["episode_id_hash"] != raylinearc.HashEpisodeID("episode-timing-raw-id") {
		t.Fatalf("episode_id_hash = %v", fields["episode_id_hash"])
	}
	if total, http := fields["total_ms"].(float64), fields["http_ms"].(float64); total < http {
		t.Fatalf("total_ms %v < http_ms %v", total, http)
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"episode-timing-raw-id", "secret-turn-text", "public-e2e-provider-key"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("the timing line carries %q: %s", secret, encoded)
		}
	}

	fake.failNext("backend_unavailable")
	logs = captureLogs(t)
	if immediate := runPolicyTurn(t, router, "episode-timing-failed", client); immediate == nil {
		t.Fatal("a refused decide was routed")
	}
	fields = oneTurnTimingEvent(t, logs)
	requireTurnTimingFields(t, fields, "failure_class", "http_ms", "first_byte_ms", "policy_ms", "parse_ms", "total_ms")
	if fields["outcome"] != "failed" || fields["surface"] != raylineARCTurnSurfaceHosted ||
		fields["failure_class"] == "selector" || fields["failure_class"] == "" {
		t.Fatalf("fields = %v", fields)
	}
	if _, present := fields["session_action"]; present {
		t.Fatalf("a failed turn logged a session action: %v", fields)
	}
}

// A /v1/routes consult records its codec parse, episode load and the commit
// it makes at decision time into the lookup's timing.
func TestRouteDecisionRecordsItsTurnTiming(t *testing.T) {
	selector := &stubARCSelector{arm: 1, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	timing := raylinearc.NewTurnTiming()
	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		raylinearc.WithTurnTiming(context.Background(), timing),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-timing"},
	); err != nil {
		t.Fatal(err)
	}
	fields := timing.Seal(time.Millisecond)
	requireTurnTimingFields(t, fields, "parse_ms", "episode_load_ms", "episode_commit_ms", "episode_id_hash")
	if fields["episode_id_hash"] != raylinearc.HashEpisodeID("session-timing") {
		t.Fatalf("episode_id_hash = %v", fields["episode_id_hash"])
	}
}

// A lookup that fails at selection still logs its one line, with its route id.
func TestRoutesLookupLogsOneTurnTimingLineOnFailure(t *testing.T) {
	router := routesRouter(true)
	ctx := routesContext(map[string]string{raylineRoutesRouteIDHeader: "rte_timing"})
	ctx.StartTime = time.Now()
	startRaylineARCTurnTiming(ctx, 64)
	logs := captureLogs(t)
	if response := router.handleRaylineRoutesAPI([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}]}`), ctx); response == nil {
		t.Fatal("no response")
	}
	fields := oneTurnTimingEvent(t, logs)
	if fields["surface"] != raylineARCTurnSurfaceRoutes || fields["outcome"] != "failed" ||
		fields["routing_decision_id"] != "rte_timing" || fields["body_bytes"] != int64(64) ||
		fields["status"] == int64(200) || fields["status"] == int64(0) {
		t.Fatalf("fields = %v", fields)
	}

	// A lookup refused before selection (a routing block on a cell that
	// serves none) logs its one line too, with the refusal's status.
	ctx = routesContext(nil)
	startRaylineARCTurnTiming(ctx, 1)
	logs = captureLogs(t)
	response := router.handleRaylineRoutesAPI([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{}}`), ctx)
	fields = oneTurnTimingEvent(t, logs)
	if fields["outcome"] != "failed" || fields["status"] != int64(immediateStatusCode(t, response)) {
		t.Fatalf("fields = %v", fields)
	}
}
