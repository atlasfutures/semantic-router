package execution

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

const hybridPreset = "arc-policy.hybrid-default.v1"

var (
	localActionID = strings.Repeat("c", 64)
	cloudActionID = strings.Repeat("d", 64)
)

// hybridV2Config binds a local Qwen fixture and a model-pin cloud fixture.
func hybridV2Config(t *testing.T, localURL, cloudURL string) Config {
	t.Helper()
	config := testConfig(t, localURL, false)
	config.AllowExperimentalControls = true
	config.Decisions.APIMode = DecisionsExecutionContextV2
	config.Decisions.OfferSHA256 = ""
	config.Decisions.Routing = json.RawMessage(`{"policy":{"preset":"` + hybridPreset + `"}}`)
	config.Deployments[0].ContextWindow = 1000
	config.Deployments[0].MaxOutputTokens = 200
	cloud := modelPinConfig(t, cloudURL)
	cloud.Deployments[0].ID = "cloud-kimi"
	cloud.Deployments[0].ContextWindow = 65536
	cloud.Deployments[0].MaxOutputTokens = 4096
	cloud.Actions[0].ID, cloud.Actions[0].Worker, cloud.Actions[0].DeploymentID = cloudActionID, "kimi-k3", "cloud-kimi"
	config.Deployments = append(config.Deployments, cloud.Deployments[0])
	config.Actions = append(config.Actions, cloud.Actions[0])
	return config
}

// fixedCounter is a qualified counter stand-in. Each call pops the next count,
// repeating the last one, so a test can make the final prepared prompt grow.
type fixedCounter struct {
	identity cachecontext.PromptIdentity
	mu       sync.Mutex
	counts   []int
	calls    int
}

func (counter *fixedCounter) Identity() cachecontext.PromptIdentity { return counter.identity }

func (counter *fixedCounter) Count(context.Context, cachecontext.PreparedPrompt) (cachecontext.PromptMeasurement, error) {
	counter.mu.Lock()
	defer counter.mu.Unlock()
	index := min(counter.calls, len(counter.counts)-1)
	counter.calls++
	return cachecontext.PromptMeasurement{PromptTokens: counter.counts[index], Provenance: "provider_reported"}, nil
}

type v2Lookup struct {
	Execution contextV2
	Policy    routingPolicy
	Raw       map[string]json.RawMessage
}

type v2Decisions struct {
	mu      sync.Mutex
	lookups []v2Lookup
	// choose returns the action the fixture policy selects.
	choose func(contextV2) string
	// mutate edits the response before it is written.
	mutate func(map[string]any)
	// failNext answers that many lookups with HTTP 500.
	failNext int
}

func (fixture *v2Decisions) snapshot() []v2Lookup {
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	return append([]v2Lookup(nil), fixture.lookups...)
}

func v2Server(t *testing.T, config Config, fixture *v2Decisions, counts ...int) (*Server, *httptest.Server) {
	t.Helper()
	t.Setenv("TEST_DECISIONS_KEY", "fixture-hosted-key-never-for-local")
	actions := map[string]Action{}
	for _, action := range config.Actions {
		actions[action.ID] = action
	}
	decisions := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Routing struct {
				Execution contextV2     `json:"execution"`
				Policy    routingPolicy `json:"policy"`
			} `json:"routing"`
		}
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Error(err)
		}
		var rawBody map[string]json.RawMessage
		_ = json.Unmarshal(raw, &rawBody)
		fixture.mu.Lock()
		fixture.lookups = append(fixture.lookups, v2Lookup{Execution: body.Routing.Execution, Policy: body.Routing.Policy, Raw: rawBody})
		index := len(fixture.lookups) - 1
		fail := fixture.failNext > 0
		if fail {
			fixture.failNext--
		}
		fixture.mu.Unlock()
		if fail {
			http.Error(w, "unavailable", 500)
			return
		}
		sent := body.Routing.Execution
		id := fixture.choose(sent)
		action := actions[id]
		excluded := []map[string]any{}
		decidedBy := "schedule"
		for _, target := range sent.Targets {
			if target.Capacity != nil && !target.Capacity.Fits {
				excluded = append(excluded, map[string]any{"action_id": target.ActionID, "reason": *target.Capacity.Reason})
				if sent.LastServed != nil && sent.LastServed.ActionID == target.ActionID {
					decidedBy = "capacity_handoff"
				}
			}
		}
		response := map[string]any{"route_id": fmt.Sprintf("route-%d", index), "model": action.Model, "worker": action.Worker, "policy_action_id": id,
			"identity": map[string]any{"package_sha256": config.PackageSHA256, "action_id": id, "policy_action_id": id, "control_id": action.ControlID, "arm_id": nil, "thinking_controls_sha256": config.RegistrySHA256, "offer_sha256": nil, "prices": map[string]any{"basis": "catalog", "price_sha256": nil, "exclusive": nil}},
			"receipt":  map[string]any{"action_id": id, "control_id": action.ControlID, "decided_by": map[string]string{"model": decidedBy, "thinking": "schedule"}, "eligible": map[string]any{"model": nil, "thinking": "all_turns"}, "held_from": nil, "excluded": excluded, "basis": sent.Event.Basis},
			"routing":  map[string]any{"execution": map[string]any{"status": "applied", "version": executionContextV2, "turn": sent.Turn, "pinned": map[string]string{"package_sha256": "matched", "thinking_controls_sha256": "matched", "offer_sha256": "unpinned"}}, "policy": map[string]any{"status": "applied", "preset": body.Routing.Policy.Preset}, "unapplied_keys": []string{}}}
		if fixture.mutate != nil {
			fixture.mutate(response)
		}
		writeJSON(w, 200, response)
	}))
	t.Cleanup(decisions.Close)
	config.Decisions.URL = decisions.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(config, registry, provider, testCoreToken)
	if err != nil {
		t.Fatal(err)
	}
	if len(counts) > 0 {
		counter := &fixedCounter{identity: promptIdentity(config.Deployments[0]), counts: counts}
		if err := server.RegisterPromptCounter(config.Deployments[0].ID, counter); err != nil {
			t.Fatal(err)
		}
	}
	httpServer := httptest.NewServer(server)
	t.Cleanup(httpServer.Close)
	return server, httpServer
}

type upstreamFixture struct {
	mu     sync.Mutex
	bodies []map[string]json.RawMessage
	// reject answers with llama.cpp's context refusal while positive; fail
	// answers with an unrelated 503 while positive.
	reject int
	fail   int
}

func (fixture *upstreamFixture) count() int {
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	return len(fixture.bodies)
}

func chatUpstream(t *testing.T, fixture *upstreamFixture, cloud bool) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		fixture.mu.Lock()
		fixture.bodies = append(fixture.bodies, body)
		reject, fail := fixture.reject > 0, fixture.fail > 0 && fixture.reject == 0
		if reject {
			fixture.reject--
		}
		if fail {
			fixture.fail--
		}
		fixture.mu.Unlock()
		if fail {
			w.WriteHeader(503)
			_, _ = io.WriteString(w, `{"error":{"code":503,"message":"loading model","type":"unavailable_error"}}`)
			return
		}
		if reject {
			w.WriteHeader(400)
			_, _ = io.WriteString(w, `{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error","n_prompt_tokens":1200,"n_ctx":1000}}`)
			return
		}
		if cloud {
			modelPinHeaders(w.Header())
		}
		_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
	}))
	t.Cleanup(server.Close)
	return server
}

const v2Tools = `"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}}]`

func chatTurn(messages ...string) string {
	return `{"model":"arc-1.0","max_tokens":64,` + v2Tools + `,"messages":[` + strings.Join(messages, ",") + `]}`
}

const (
	userTask   = `{"role":"user","content":"Read the file."}`
	toolCall   = `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]}`
	toolResult = `{"role":"tool","tool_call_id":"call_1","content":"file contents"}`
	answer     = `{"role":"assistant","content":"ready"}`
	followUp   = `{"role":"user","content":"Now summarise it."}`
)

func TestV2EnvelopeCarriesEventsLastServedHistoryAndTargets(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture, 300)
	for index, body := range []string{chatTurn(userTask), chatTurn(userTask, toolCall, toolResult), chatTurn(userTask, toolCall, toolResult, answer, followUp)} {
		if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "v2-envelope"); status != 200 {
			t.Fatalf("turn %d status %d: %s", index, status, response)
		}
	}
	lookups := fixture.snapshot()
	if len(lookups) != 3 {
		t.Fatalf("got %d lookups", len(lookups))
	}
	for index, want := range []string{EventTaskStart, EventToolResult, EventUserMessage} {
		got := lookups[index].Execution
		if got.Version != executionContextV2 || got.Event.Kind == nil || *got.Event.Kind != want || got.Event.Basis != BasisMarkerInferred {
			t.Fatalf("turn %d event %+v", index, got.Event)
		}
		if lookups[index].Policy.Preset != hybridPreset {
			t.Fatalf("turn %d policy %+v", index, lookups[index].Policy)
		}
		if got.Turn.Attempt != 1 || got.Turn.RetryOf != nil || got.Turn.Reconsult != nil || got.Turn.Epoch == "" {
			t.Fatalf("turn %d identity %+v", index, got.Turn)
		}
		if got.Hold != nil || got.Expect == nil || got.Expect.PackageSHA256 == nil || *got.Expect.PackageSHA256 != config.PackageSHA256 || got.Expect.OfferSHA256 != nil {
			t.Fatalf("turn %d hold/expect %+v %+v", index, got.Hold, got.Expect)
		}
		if len(got.Targets) != 2 || got.Targets[0].ActionID != localActionID || got.Targets[0].Placement != "local" || got.Targets[1].Placement != "cloud" || got.Targets[1].Capacity != nil {
			t.Fatalf("turn %d targets %+v", index, got.Targets)
		}
		capacity := got.Targets[0].Capacity
		if capacity == nil || !capacity.Fits || *capacity.Required != 300+64 || *capacity.Available != 1000 || *capacity.AvailableSource != "configured" || *capacity.Units != "provider:fixture-worker" || capacity.Provenance != "local_measured" {
			t.Fatalf("turn %d local capacity %+v", index, capacity)
		}
	}
	if lookups[0].Execution.LastServed != nil || len(lookups[0].Execution.History) != 0 {
		t.Fatal("first turn claimed a served action")
	}
	last := lookups[2].Execution
	if last.LastServed == nil || last.LastServed.ActionID != localActionID || last.LastServed.Worker != "fixture-qwen" || last.LastServed.Message == nil || *last.LastServed.Message != 3 {
		t.Fatalf("last_served %+v", last.LastServed)
	}
	if len(last.History) != 2 || last.History[0].Message != 1 || last.History[1].Message != 3 {
		t.Fatalf("history %+v", last.History)
	}
	if last.Targets[0].Cache == nil || last.Targets[0].Cache.ContextEpoch == nil || *last.Targets[0].Cache.ContextEpoch != last.Turn.Epoch || last.Targets[1].Cache != nil {
		t.Fatalf("cache evidence %+v", last.Targets)
	}
	if local.count() != 3 || cloud.count() != 0 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestV2ExplicitEventIsStrippedAndPrecedesMarkers(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "explicit"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	body := `{"model":"arc-1.0","max_tokens":64,` + v2Tools + `,"rayline_event":{"kind":"compaction","compaction":{"summary_message":0,"retained_from":1,"new":true}},"messages":[{"role":"user","content":"Summary of earlier work."},` + answer + `,` + followUp + `]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "explicit"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	event := lookups[1].Execution.Event
	if event.Kind == nil || *event.Kind != EventCompaction || event.Basis != BasisExplicitClient || event.Compaction == nil || !event.Compaction.New || *event.Compaction.RetainedFrom != 1 {
		t.Fatalf("explicit compaction event %+v", event)
	}
	if _, leaked := lookups[1].Raw["rayline_event"]; leaked {
		t.Fatal("event body field reached Decisions")
	}
	local.mu.Lock()
	_, leaked := local.bodies[1]["rayline_event"]
	local.mu.Unlock()
	if leaked {
		t.Fatal("event body field reached the provider")
	}
	if lookups[1].Execution.Turn.Epoch == lookups[0].Execution.Turn.Epoch || lookups[1].Execution.LastServed == nil {
		t.Fatal("compaction must start a new epoch and keep last_served")
	}
	for name, mutate := range map[string]func(*http.Request){
		"unknown kind": func(r *http.Request) { r.Header.Set(eventHeader, `{"kind":"timer"}`) },
		"both carriers": func(r *http.Request) {
			r.Header.Set(eventHeader, `{"kind":"user_message"}`)
		},
		"compaction outside body": func(r *http.Request) {
			r.Header.Set(eventHeader, `{"kind":"compaction","compaction":{"summary_message":9,"retained_from":null,"new":true}}`)
		},
	} {
		t.Run(name, func(t *testing.T) {
			payload := chatTurn(userTask)
			if name == "both carriers" {
				payload = `{"rayline_event":{"kind":"user_message"},` + payload[1:]
			}
			request, _ := http.NewRequest(http.MethodPost, httpServer.URL+"/v1/chat/completions", strings.NewReader(payload))
			request.Header.Set("Authorization", "Bearer "+testCoreToken)
			request.Header.Set("X-Rayline-Session-Id", "invalid-"+name)
			mutate(request)
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			if response.StatusCode != 400 {
				t.Fatalf("status %d", response.StatusCode)
			}
		})
	}
}

func TestV2CapacityFactHandsOffWithoutCoreReselection(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	// The fixture policy prefers local whenever the core does not report it infeasible.
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}}
	// First turn fits; the tool-loop continuation outgrows the 1000-token allocation.
	_, httpServer := v2Server(t, config, fixture, 300, 300, 990)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "spill"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	request, _ := http.NewRequest(http.MethodPost, httpServer.URL+"/v1/chat/completions", strings.NewReader(chatTurn(userTask, toolCall, toolResult)))
	request.Header.Set("Authorization", "Bearer "+testCoreToken)
	request.Header.Set("X-Rayline-Session-Id", "spill")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != 200 || response.Header.Get("X-Rayline-Decided-By") != "capacity_handoff/schedule" || response.Header.Get("X-Rayline-Executed-Action-Id") != cloudActionID {
		t.Fatalf("handoff status %d headers %v", response.StatusCode, response.Header)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 {
		t.Fatalf("a pre-decision fact needs no re-consult; got %d lookups", len(lookups))
	}
	spill := lookups[1].Execution
	capacity := spill.Targets[0].Capacity
	if capacity == nil || capacity.Fits || *capacity.Reason != CapacityContext || *capacity.Required != 990+64 || spill.LastServed == nil || spill.LastServed.ActionID != localActionID || *spill.Event.Kind != EventToolResult {
		t.Fatalf("spill envelope %+v %+v", capacity, spill.LastServed)
	}
	if local.count() != 1 || cloud.count() != 1 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestV2PreDispatchVetoReconsultsWithTheFact(t *testing.T) {
	// The first dispatch fails before output; the identical client retry keeps
	// the decision but re-checks the final prepared request, which no longer
	// fits. The core vetoes with the fact and executes the service's new choice.
	local, cloud := &upstreamFixture{fail: 1}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}}
	_, httpServer := v2Server(t, config, fixture, 300, 990)
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "veto"); status != 503 {
		t.Fatalf("first attempt status %d", status)
	}
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "veto"); status != 200 {
		t.Fatalf("retry status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 {
		t.Fatalf("got %d lookups", len(lookups))
	}
	first, second := lookups[0].Execution, lookups[1].Execution
	if second.Turn.ID != first.Turn.ID || second.Turn.Attempt != 2 || second.Turn.Reconsult == nil || *second.Turn.Reconsult != "veto" || second.Turn.RetryOf == nil || *second.Turn.RetryOf != "route-0" {
		t.Fatalf("veto turn %+v", second.Turn)
	}
	if *second.Event.Kind != *first.Event.Kind || second.Targets[0].Capacity.Fits || *second.Targets[0].Capacity.Required != 990+64 {
		t.Fatalf("veto envelope %+v", second.Targets[0].Capacity)
	}
	if local.count() != 1 || cloud.count() != 1 {
		t.Fatalf("vetoed request reached local again: local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestV2RuntimeContextRefusalReconsultsBeforeOutput(t *testing.T) {
	local, cloud := &upstreamFixture{reject: 1}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}}
	_, httpServer := v2Server(t, config, fixture, 300)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "runtime"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 || *lookups[1].Execution.Turn.Reconsult != "capacity" || lookups[1].Execution.Targets[0].Capacity.Provenance != "provider_reported" {
		t.Fatalf("runtime refusal re-consult %+v", lookups)
	}
	if local.count() != 1 || cloud.count() != 1 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestV2ServiceThatReselectsAnInfeasibleTargetFailsVisibly(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture, 990)
	status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "stubborn")
	if status != http.StatusBadGateway || !strings.Contains(string(response), "does not fit") {
		t.Fatalf("status %d: %s", status, response)
	}
	if local.count() != 0 || cloud.count() != 0 {
		t.Fatal("core dispatched despite an infeasible selection")
	}
}

func TestV2SidecallDoesNotAdvanceBranchState(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	server, httpServer := v2Server(t, config, fixture)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "side"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	side := `{"model":"arc-1.0","max_tokens":64,"rayline_event":{"kind":"sidecall"},"messages":[{"role":"user","content":"Title this conversation."}]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", side, "side"); status != 200 {
		t.Fatalf("sidecall status %d: %s", status, response)
	}
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask, toolCall, toolResult), "side"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	if len(lookups[1].Execution.History) != 0 || lookups[1].Execution.LastServed == nil {
		t.Fatalf("sidecall envelope %+v", lookups[1].Execution)
	}
	if lookups[2].Execution.Turn.Epoch != lookups[0].Execution.Turn.Epoch || len(lookups[2].Execution.History) != 1 {
		t.Fatal("sidecall changed the main branch's epoch or attribution")
	}
	state, err := server.states.acquire(t.Context(), "side", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	if state.completed != 2 {
		t.Fatalf("completed %d, want 2 (sidecall excluded)", state.completed)
	}
}

func TestV2CompactionCarriesRetainedAttribution(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	for _, body := range []string{chatTurn(userTask), chatTurn(userTask, toolCall, toolResult)} {
		if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "compact"); status != 200 {
			t.Fatalf("status %d: %s", status, response)
		}
	}
	// Pi's shape: summary at 0, retained tail (the tool call, its result and
	// the reply that turn produced) from 1, then the new user message.
	body := `{"model":"arc-1.0","max_tokens":64,` + v2Tools + `,"rayline_event":{"kind":"compaction","compaction":{"summary_message":0,"retained_from":1,"new":true}},"messages":[{"role":"user","content":"Summary."},` + toolCall + `,` + toolResult + `,` + answer + `,` + followUp + `]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "compact"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	history := fixture.snapshot()[2].Execution.History
	if len(history) != 2 || history[0].Message != 1 || history[1].Message != 3 {
		t.Fatalf("retained history %+v", history)
	}
}

func TestV2ResponseChecksRefuseUnappliedOrUnexplainedDecisions(t *testing.T) {
	for name, mutate := range map[string]func(map[string]any){
		"missing receipt": func(response map[string]any) { delete(response, "receipt") },
		"receipt names another action": func(response map[string]any) {
			response["receipt"].(map[string]any)["action_id"] = cloudActionID
		},
		"unknown decided_by": func(response map[string]any) {
			response["receipt"].(map[string]any)["decided_by"] = map[string]string{"model": "periodic", "thinking": "schedule"}
		},
		"pin not matched": func(response map[string]any) {
			response["routing"].(map[string]any)["execution"].(map[string]any)["pinned"] = map[string]any{"package_sha256": "mismatched", "thinking_controls_sha256": "matched", "offer_sha256": nil}
		},
		"execution unsupported": func(response map[string]any) {
			response["routing"].(map[string]any)["execution"].(map[string]any)["status"] = "unsupported"
		},
		"policy not echoed": func(response map[string]any) {
			delete(response["routing"].(map[string]any), "policy")
		},
		"turn not echoed": func(response map[string]any) {
			response["routing"].(map[string]any)["execution"].(map[string]any)["turn"] = map[string]any{"id": "other", "attempt": 1}
		},
		"v1 version": func(response map[string]any) {
			response["routing"].(map[string]any)["execution"].(map[string]any)["version"] = "arc-execution-context.v1"
		},
	} {
		t.Run(name, func(t *testing.T) {
			local, cloud := &upstreamFixture{}, &upstreamFixture{}
			config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
			fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }, mutate: mutate}
			_, httpServer := v2Server(t, config, fixture)
			if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "refuse"); status != http.StatusBadGateway {
				t.Fatalf("status %d: %s", status, response)
			}
			if local.count() != 0 {
				t.Fatal("dispatched an unvalidated decision")
			}
		})
	}
}

func TestV2ConfigRequiresAnExplicitPublishedPolicy(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	base := hybridV2Config(t, "http://127.0.0.1:1/v1/chat/completions", "http://127.0.0.1:2")
	base.Decisions.URL = "http://127.0.0.1:3/v1/routes"
	for name, test := range map[string]struct {
		mode    string
		routing string
		ok      bool
	}{
		"default preset":             {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1"}}`, true},
		"David's schedule with V":    {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"schedule":{"model":["cache_boundaries","user_messages"],"thinking":["all_turns"]}}},"value_usd":0.1}`, true},
		"package default":            {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.package-default.v1"}}`, true},
		"prefer local":               {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v1"}}`, true},
		"prefer local threshold":     {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":0.65}}}}`, true},
		"threshold out of range":     {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":1}}}}`, false},
		"prefer local relative":      {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v2"}}`, true},
		"relative delta":             {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v2","overrides":{"selection":{"delta":0.1}}}}`, true},
		"relative delta zero":        {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v2","overrides":{"selection":{"delta":0}}}}`, true},
		"delta out of range":         {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v2","overrides":{"selection":{"delta":1.5}}}}`, false},
		"threshold on relative":      {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v2","overrides":{"selection":{"min_p_pass":0.5}}}}`, false},
		"delta on fixed":             {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":0.5,"delta":0.1}}}}`, false},
		"absent policy":              {DecisionsExecutionContextV2, `{}`, false},
		"policy forwarded in v1":     {DecisionsExecutionContextV1, `{"policy":{"preset":"arc-policy.hybrid-default.v1"}}`, true},
		"policy refused in current":  {DecisionsCurrent, `{"policy":{"preset":"arc-policy.hybrid-default.v1"}}`, false},
		"unpublished preset":         {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.always-local.v1"}}`, false},
		"unknown class":              {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"schedule":{"model":["every_5"]}}}}`, false},
		"selection override":         {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"selection":{"min_p_pass":0.5}}}}`, false},
		"unknown override":           {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"selector":"utility"}}}`, false},
		"unacknowledged routing key": {DecisionsExecutionContextV2, `{"policy":{"preset":"arc-policy.hybrid-default.v1"},"lookahead":1}`, false},
	} {
		t.Run(name, func(t *testing.T) {
			config := base
			config.Decisions.APIMode = test.mode
			config.Decisions.OfferSHA256 = strings.Repeat("b", 64)
			config.Decisions.Routing = json.RawMessage(test.routing)
			if strings.Contains(test.routing, "value_usd") {
				config.Decisions.Result = "custom"
			}
			_, err := NewHTTPDecisions(config)
			if (err == nil) != test.ok {
				t.Fatalf("ok=%v err=%v", test.ok, err)
			}
		})
	}
}

func TestV2CompactionRetryAfterFailureKeepsRetainedAttribution(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	for _, body := range []string{chatTurn(userTask), chatTurn(userTask, toolCall, toolResult)} {
		if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "compact-retry"); status != 200 {
			t.Fatalf("status %d: %s", status, response)
		}
	}
	compaction := func(followUp string) string {
		return `{"model":"arc-1.0","max_tokens":64,` + v2Tools + `,"rayline_event":{"kind":"compaction","compaction":{"summary_message":0,"retained_from":1,"new":true}},"messages":[{"role":"user","content":"Summary."},` + toolCall + `,` + toolResult + `,` + answer + `,` + followUp + `]}`
	}
	local.mu.Lock()
	local.fail = 1
	local.mu.Unlock()
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/chat/completions", compaction(followUp), "compact-retry"); status != 503 {
		t.Fatalf("first compaction attempt status %d", status)
	}
	// The harness retries with a changed body (for example a re-sent prompt);
	// the core decides anew and must still carry the retained attribution.
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", compaction(`{"role":"user","content":"Now summarise it, please."}`), "compact-retry"); status != 200 {
		t.Fatalf("retry status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	history := lookups[len(lookups)-1].Execution.History
	if len(history) != 2 || history[0].Message != 1 || history[1].Message != 3 {
		t.Fatalf("retained history after retry %+v", history)
	}
}

func TestV2OptionalExplicitEventDoesNotStopWireInference(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	server, httpServer := v2Server(t, config, fixture)
	explicit := `{"rayline_event":{"kind":"task_start"},` + chatTurn(userTask)[1:]
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", explicit, "optional"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	// A later request without an explicit event is read from the wire as usual.
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask, toolCall, toolResult), "optional"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	// A harness side call (Chat system prompt literal) is recognised without a label.
	summary := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"system","content":"` + piSummarizationSystem + `"},{"role":"user","content":"<conversation>...</conversation>"}]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", summary, "optional"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	if got := []string{lookups[1].Execution.Event.kindName(), lookups[2].Execution.Event.kindName()}; got[0] != EventToolResult || got[1] != EventSidecall {
		t.Fatalf("events %v", got)
	}
	state, err := server.states.acquire(t.Context(), "optional", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	if state.completed != 2 || state.epoch != 0 {
		t.Fatalf("summary call advanced the main branch: completed=%d epoch=%d", state.completed, state.epoch)
	}
}

func TestV2CounterFailureIsRuntimeUnavailableForThatTarget(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterPromptCounter(config.Deployments[0].ID, failingCounter{identity: promptIdentity(config.Deployments[0])}); err != nil {
		t.Fatal(err)
	}
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "counter-down"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	capacity := fixture.snapshot()[0].Execution.Targets[0].Capacity
	if capacity == nil || capacity.Fits || capacity.Reason == nil || *capacity.Reason != CapacityRuntimeUnavailable || capacity.Required != nil {
		t.Fatalf("capacity %s", mustJSON(capacity))
	}
	if local.count() != 0 || cloud.count() != 1 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

type failingCounter struct{ identity cachecontext.PromptIdentity }

func (counter failingCounter) Identity() cachecontext.PromptIdentity { return counter.identity }
func (failingCounter) Count(context.Context, cachecontext.PreparedPrompt) (cachecontext.PromptMeasurement, error) {
	return cachecontext.PromptMeasurement{}, fmt.Errorf("llama.cpp /tokenize request failed: connection refused")
}

func TestRemapRetainedAttributionPrefersTheRetainedCopyOfRepeatedContent(t *testing.T) {
	// previous: user, A, tool, A, tool (A repeats). Retained tail is the last A, tool.
	previous := []string{"u", "a", "t", "a", "t"}
	current := []string{"summary", "a", "t", "reply", "next"}
	roles := []string{"user", "assistant", "tool", "assistant", "user"}
	ledger := []attributedTurn{{entry: Attribution{Message: 1, ActionID: "first"}}, {entry: Attribution{Message: 3, ActionID: "second"}}, {entry: Attribution{Message: 5, ActionID: "reply"}}}
	retained := 1
	got := remapRetainedAttribution(ledger, previous, current, roles, "", &CompactionEvent{SummaryMessage: 0, RetainedFrom: &retained, New: true})
	if len(got) != 2 || got[0].entry.ActionID != "second" || got[0].entry.Message != 1 || got[1].entry.ActionID != "reply" || got[1].entry.Message != 3 {
		t.Fatalf("remap %+v", got)
	}
}

func TestV2AppliedEchoWithoutPinsIsRefused(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }, mutate: func(response map[string]any) {
		delete(response["routing"].(map[string]any)["execution"].(map[string]any), "pinned")
	}}
	_, httpServer := v2Server(t, config, fixture)
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "no-pins"); status != http.StatusBadGateway || local.count() != 0 {
		t.Fatalf("status %d local %d", status, local.count())
	}
}
