package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// signalsFixture is a decisions cell whose fake service knows and serves
// topic.operation, topic.domain and pii.payload: a name the router has
// never been told about, so nothing here is specific to one family.
func signalsFixture(t *testing.T) *policySelectorFixture {
	t.Helper()
	fixture := routesFixture(t)
	fixture.fake.knownSignals = []string{"topic.operation", "topic.domain", "pii.payload"}
	fixture.fake.servedSignals = []string{"topic.operation", "topic.domain", "pii.payload"}
	return fixture
}

func signalsControls(t *testing.T, model, routingJSON string) (*raylineRoutesRouting, *selection.RaylineARCRouteControls) {
	t.Helper()
	body := `{"model":"` + model + `","messages":[{"role":"user","content":"x"}],"routing":` + routingJSON + `}`
	routing, _, refusal := readRaylineRoutesRouting([]byte(body), nil)
	if refusal != nil {
		t.Fatalf("%s: refused %+v", routingJSON, refusal)
	}
	return routing, raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{ExplorationSalt: "salt"}, time.Now())
}

// withSignals adds a signals member to a routing block's JSON.
func withSignals(routingJSON, signals string) string {
	if routingJSON == "{}" {
		return `{"signals":` + signals + `}`
	}
	return strings.TrimSuffix(routingJSON, "}") + `,"signals":` + signals + `}`
}

func TestRoutingSignalsShape(t *testing.T) {
	routing, _ := signalsControls(t, "arc-1.0", `{"signals":["topic.operation","pii.payload"]}`)
	if strings.Join(routing.signals, ",") != "topic.operation,pii.payload" {
		t.Fatalf("signals %v", routing.signals)
	}
	// An empty array is no signals: the lookup is answered as today.
	if routing, _ := signalsControls(t, "arc-1.0", `{"signals":[]}`); routing.signals != nil {
		t.Fatalf("[] read as %v", routing.signals)
	}
	for name, signals := range map[string]string{
		"not an array": `"topic.operation"`, "null": `null`, "number": `[1]`, "empty name": `[""]`,
		"duplicate": `["topic.domain","topic.domain"]`, "object": `{"topic.domain":true}`,
	} {
		body := `{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{"signals":` + signals + `}}`
		if _, _, refusal := readRaylineRoutesRouting([]byte(body), nil); refusal == nil || refusal.code != "signals_invalid" || refusal.status != 400 {
			t.Errorf("routes %s: %+v", name, refusal)
		}
		if _, _, refusal := readRaylineGenerationRouting([]byte(body), generationPolicy(true, nil)); refusal == nil || refusal.code != "signals_invalid" {
			t.Errorf("generation %s: %+v", name, refusal)
		}
	}
	// Any well-shaped name passes the router: admitting it is the service's.
	generation, _, refusal := readRaylineGenerationRouting(
		[]byte(`{"model":"auto","messages":[],"routing":{"signals":["pii.payload"]}}`), generationPolicy(false, sageWeight()))
	if refusal != nil || strings.Join(generation.signals, ",") != "pii.payload" || !strings.Contains(generation.key(), "pii.payload") {
		t.Fatalf("generation %+v %+v", generation, refusal)
	}
}

// routing.signals reaches the decide request as signals, and the service's
// block comes back on the trace and the public answer exactly as written.
func TestRoutingSignalsReachTheServiceAndComeBack(t *testing.T) {
	fixture := signalsFixture(t)
	routing, controls := signalsControls(t, "arc-1.0", `{"signals":["topic.operation","pii.payload"]}`)
	result, err := selectWithControls(t, fixture, controls)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.receivedBodies()[0]
	if !bytes.Contains(sent, []byte(`"signals":["topic.operation","pii.payload"]`)) {
		t.Fatalf("decide request %s", sent)
	}
	want := fakeSignalBlock([]string{"topic.operation", "pii.payload"})
	if !bytes.Equal(result.RaylineARC.Signals, want) {
		t.Fatalf("trace signals %s, want %s", result.RaylineARC.Signals, want)
	}
	var payload raylineRoutesResponse
	if err := raylineRoutesAddResults(&payload, routing, signalsDecision(want)); err != nil {
		t.Fatal(err)
	}
	var answer map[string]json.RawMessage
	_ = json.Unmarshal(executionJSON(t, payload), &answer)
	if !bytes.Equal(answer["signals"], want) {
		t.Fatalf("answer signals %s", answer["signals"])
	}
	// Asked and not answered: withheld as inconsistent, never shown as if
	// the caller had not asked.
	if err := raylineRoutesAddResults(&raylineRoutesResponse{}, routing, signalsDecision(nil)); err == nil {
		t.Fatal("a lookup that asked for signals was answered without them")
	}
}

func signalsDecision(block json.RawMessage) routerruntime.RouteDecision {
	return routerruntime.RouteDecision{
		Results: []routerruntime.RouteResult{
			{Name: "swift", CostWeight: 10, SelectedWorker: "w", WorkerModel: "m/a"},
			{Name: "sage", CostWeight: 1 / 4.5, SelectedWorker: "w", WorkerModel: "m/a"},
			{Name: "custom", CostWeight: 1 / 2.0, SelectedWorker: "w", WorkerModel: "m/a"},
		},
		Signals: block,
	}
}

// Without routing.signals nothing changes end to end: on the Swift and Sage
// presets and tiers and on hybrid-default, the decide request carries no
// signals member and is byte for byte the signals request minus exactly
// that member, and the public answer is byte for byte the signals answer
// minus its top-level signals.
func TestRoutingSignalsAbsentIsByteIdentical(t *testing.T) {
	for _, tc := range []struct{ name, model, routing string }{
		{"presets", "arc-1.0", `{}`},
		{"swift", "arc-swift", `{"value_usd":0.1}`},
		{"sage", "arc-sage", `{"value_usd":4.5}`},
		{"hybrid-default", "arc-1.0", `{"policy":{"preset":"arc-policy.hybrid-default.v1"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const signals = `["topic.operation","topic.domain"]`
			decide := func(routingJSON string) ([]byte, *raylineRoutesRouting, *selection.RaylineARCTrace) {
				fixture := signalsFixture(t)
				routing, controls := signalsControls(t, tc.model, routingJSON)
				result, err := selectWithControls(t, fixture, controls)
				if err != nil {
					t.Fatalf("%s: select: %v", routingJSON, err)
				}
				return fixture.fake.receivedBodies()[0], routing, result.RaylineARC
			}
			plain, plainRouting, plainTrace := decide(tc.routing)
			signalled, signalledRouting, signalledTrace := decide(withSignals(tc.routing, signals))
			if bytes.Contains(plain, []byte(`"signals"`)) || plainTrace.Signals != nil {
				t.Fatalf("a request without signals sent or recorded them: %s", plain)
			}
			if stripped := bytes.Replace(signalled, []byte(`,"signals":`+signals), nil, 1); !bytes.Equal(stripped, plain) {
				t.Fatalf("decide bytes differ beyond the signals member\n got %s\nwant %s", stripped, plain)
			}
			if !plainControlsKeyUnchanged(t, tc.model, tc.routing) {
				t.Fatal("the coalescing key of a lookup without signals changed")
			}

			render := func(routing *raylineRoutesRouting, block json.RawMessage) []byte {
				decision := signalsDecision(block)
				if routing.valueUSD == nil {
					decision.Results = decision.Results[:2]
				}
				payload := raylineRoutesPayload("rte_x", "ckpt", decision, []string{}, 0)
				if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
					t.Fatal(err)
				}
				return executionJSON(t, payload)
			}
			plainAnswer := render(plainRouting, nil)
			if bytes.Contains(plainAnswer, []byte(`"signals"`)) {
				t.Fatalf("an answer without signals carries them: %s", plainAnswer)
			}
			// A block the service sent unasked is never shown.
			if leaked := render(plainRouting, signalledTrace.Signals); !bytes.Equal(leaked, plainAnswer) {
				t.Fatalf("an unasked block reached the answer: %s", leaked)
			}
			signalledAnswer := render(signalledRouting, signalledTrace.Signals)
			var members map[string]json.RawMessage
			if err := json.Unmarshal(signalledAnswer, &members); err != nil || members["signals"] == nil {
				t.Fatalf("signals answer %s", signalledAnswer)
			}
			stripped := bytes.Replace(signalledAnswer, append([]byte(`,"signals":`), members["signals"]...), nil, 1)
			if !bytes.Equal(stripped, plainAnswer) {
				t.Fatalf("answer bytes differ beyond signals\n got %s\nwant %s", stripped, plainAnswer)
			}
		})
	}
}

// plainControlsKeyUnchanged checks a lookup without signals keys exactly as
// the key's members before signals existed.
func plainControlsKeyUnchanged(t *testing.T, model, routingJSON string) bool {
	t.Helper()
	_, controls := signalsControls(t, model, routingJSON)
	return !strings.Contains(raylineRoutesRoutingControlsKey(controls), "signals")
}

// The service's refusals reach the caller with the service's reason as the
// code: unknown and unserved names as 400, unloadable heads as 503. The
// router keeps no list of its own.
func TestRoutingSignalsServiceRefusals(t *testing.T) {
	router := routesRouter(true)
	for _, tc := range []struct {
		name, signals, code string
		status              int
		broken              bool
	}{
		{"unknown", `["topic.mood"]`, "signal_not_supported", 400, false},
		{"not served", `["pii.payload"]`, "signal_not_admitted", 400, false},
		{"heads unavailable", `["topic.domain"]`, "signal_heads_unavailable", 503, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := signalsFixture(t)
			fixture.fake.servedSignals = []string{"topic.operation", "topic.domain"}
			fixture.fake.signalHeadsBroken = tc.broken
			_, controls := signalsControls(t, "arc-1.0", `{"signals":`+tc.signals+`}`)
			_, err := selectWithControls(t, fixture, controls)
			mapped := selectionFailureError(err)
			var refusal *routeSignalRefusal
			if !errors.As(mapped, &refusal) {
				t.Fatalf("err = %v", mapped)
			}
			status, _, code := routesErrorCode(t, router.raylineRoutesFailure(routesContext(nil), context.Background(), "rte_x", mapped))
			if status != tc.status || code != tc.code {
				t.Fatalf("answered %d %s, want %d %s", status, code, tc.status, tc.code)
			}
		})
	}
	// An unsupported_request that is not about signals keeps its class.
	other := &raylinearc.PolicyServiceError{Class: "unsupported_request", Detail: map[string]any{"reason": "projection_refused"}}
	if signalRefusalClass(other) != "" {
		t.Fatal("a projection refusal read as a signal refusal")
	}
}

// generationSignalsTurn is generationTurn returning the request context, so
// the turn's trace and response headers can be read.
func generationSignalsTurn(t *testing.T, router *OpenAIRouter, episode, path, client string) (*ext_proc.ImmediateResponse, []byte, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "signals-e2e-" + episode, StartTime: time.Now(),
		TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"}, {Key: ":path", Value: path},
			{Key: "content-type", Value: "application/json"}, {Key: "x-rayline-session", Value: episode},
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
	if response.GetImmediateResponse() != nil {
		return router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse(), nil, ctx
	}
	return nil, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), ctx
}

// On a generation turn routing.signals reaches the decide request, never the
// provider, and the block comes back in x-vsr-signals; without it the decide
// request carries no signals and no header is written. A refusal is the
// service's reason in the client's own envelope.
func TestGenerationRoutingSignals(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage,
		[]string{actions["think"].ActionID, actions["off"].ActionID, actions["claude"].ActionID, actions["claude-off"].ActionID})
	fake.knownSignals = []string{"topic.operation", "topic.domain", "pii.payload"}
	fake.servedSignals = []string{"topic.operation", "pii.payload"}
	router, err := NewOpenAIRouter(writeGenerationRoutingConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["think"].ActionID })
	chat := func(routing string) string {
		return `{"model":"auto","max_completion_tokens":1024,` + routing +
			`"messages":[{"role":"user","content":"fix the failing test"}]}`
	}

	immediate, provider, ctx := generationSignalsTurn(t, router, "signals-asked", "/v1/chat/completions",
		chat(`"routing":{"signals":["topic.operation","pii.payload"]},`))
	if immediate != nil {
		t.Fatalf("refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	if bytes.Contains(provider, []byte("signals")) || bytes.Contains(provider, []byte(`"routing"`)) {
		t.Fatalf("routing reached the provider: %s", provider)
	}
	sent := fake.receivedBodies()
	if !bytes.Contains(sent[len(sent)-1], []byte(`"signals":["topic.operation","pii.payload"]`)) {
		t.Fatalf("decide request %s", sent[len(sent)-1])
	}
	want := fakeSignalBlock([]string{"topic.operation", "pii.payload"})
	if got := raylineGenerationSignalsHeader(ctx); got != string(want) {
		t.Fatalf("x-vsr-signals %q, want %s", got, want)
	}
	if applied, unapplied := raylineGenerationRoutingStatus(ctx); strings.Join(applied, ",") != "signals" || len(unapplied) != 0 {
		t.Fatalf("applied %v unapplied %v", applied, unapplied)
	}

	immediate, _, ctx = generationSignalsTurn(t, router, "signals-absent", "/v1/chat/completions", chat(""))
	if immediate != nil {
		t.Fatalf("refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	sent = fake.receivedBodies()
	if bytes.Contains(sent[len(sent)-1], []byte(`"signals"`)) || raylineGenerationSignalsHeader(ctx) != "" ||
		ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.Signals != nil {
		t.Fatalf("a turn without signals sent or answered them: %s", sent[len(sent)-1])
	}

	for _, tc := range []struct {
		name, path, client, code string
		status                   uint32
		broken                   bool
	}{
		{"unknown on messages", "/v1/messages", `{"model":"auto","max_tokens":1024,"routing":{"signals":["topic.mood"]},` +
			`"messages":[{"role":"user","content":"x"}]}`, "signal_not_supported", 400, false},
		{"not served on chat", "/v1/chat/completions", chat(`"routing":{"signals":["topic.domain"]},`), "signal_not_admitted", 400, false},
		{"heads unavailable on chat", "/v1/chat/completions", chat(`"routing":{"signals":["topic.operation"]},`),
			"signal_heads_unavailable", 503, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake.mu.Lock()
			fake.signalHeadsBroken = tc.broken
			fake.mu.Unlock()
			immediate, provider, _ := generationSignalsTurn(t, router, "signals-"+strings.ReplaceAll(tc.name, " ", "-"), tc.path, tc.client)
			if immediate == nil || provider != nil || uint32(immediate.GetStatus().GetCode()) != tc.status {
				t.Fatalf("immediate %v provider %s", immediate, provider)
			}
			var envelope struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(immediate.GetBody(), &envelope); err != nil || envelope.Error.Code != tc.code {
				t.Fatalf("refusal body %s, want code %s", immediate.GetBody(), tc.code)
			}
		})
	}
}
