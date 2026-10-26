package execution

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// The Rayline Local default: a custom local endpoint with no prompt counter
// (98,304-token window) and two cloud deployments with 1,000,000-token windows.
const (
	estimateLocalWindow = 98304
	estimateCloudWindow = 1_000_000
)

var secondCloudActionID = strings.Repeat("e", 64)

func capacityEstimateConfig(t *testing.T, localURL, cloudURL string, secondWindow int64) Config {
	t.Helper()
	config := hybridV2Config(t, localURL, cloudURL)
	config.Deployments[0].ContextWindow, config.Deployments[0].MaxOutputTokens = estimateLocalWindow, 8192
	config.Deployments[1].ContextWindow, config.Deployments[1].MaxOutputTokens = estimateCloudWindow, 32000
	second := config.Deployments[1]
	second.ID, second.ContextWindow = "cloud-second", secondWindow
	action := config.Actions[1]
	action.ID, action.Worker, action.DeploymentID = secondCloudActionID, "kimi-k3-second", "cloud-second"
	config.Deployments = append(config.Deployments, second)
	config.Actions = append(config.Actions, action)
	return config
}

// fitPolicy stands in for the service: it never offers a target the core
// reports fits:false, prefers local, and otherwise takes the last cloud target
// left (so a smaller-window cloud target is visibly skipped when it is excluded).
func fitPolicy(sent contextV2) string {
	chosen := ""
	for _, target := range sent.Targets {
		if target.Capacity != nil && !target.Capacity.Fits {
			continue
		}
		if target.Placement == "local" {
			return target.ActionID
		}
		chosen = target.ActionID
	}
	return chosen
}

// promptOfAbout builds a user message whose prepared request the estimator
// sizes at roughly tokens (bytes / 3 * 1.15).
func promptOfAbout(tokens int) string {
	filler := strings.Repeat("lorem ipsum ", int(float64(tokens)*3/1.15)/12)
	return `{"role":"user","content":"` + filler + `"}`
}

func bigTurn(messages ...string) string {
	return `{"model":"arc-1.0","max_tokens":4096,` + v2Tools + `,"messages":[` + strings.Join(messages, ",") + `]}`
}

func TestCapacityEstimateRoutesByWindowWithoutACounter(t *testing.T) {
	for _, tc := range []struct {
		name         string
		tokens       int
		secondWindow int64
		want         string
	}{
		{"50k stays local", 50_000, estimateCloudWindow, localActionID},
		{"120k hands off to cloud before local sees it", 120_000, estimateCloudWindow, secondCloudActionID},
		{"600k goes to a cloud window that fits", 600_000, estimateCloudWindow, secondCloudActionID},
		{"600k skips a cloud window that does not fit", 600_000, 200_000, cloudActionID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			local, cloud := &upstreamFixture{}, &upstreamFixture{}
			config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions", tc.secondWindow)
			fixture := &v2Decisions{choose: fitPolicy}
			_, httpServer := v2Server(t, config, fixture)
			status, header, response := executeWithHeaders(t, httpServer.URL, bigTurn(promptOfAbout(tc.tokens)), "estimate")
			if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != tc.want {
				t.Fatalf("status %d executed %q: %.300s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
			}
			lookups := fixture.snapshot()
			if len(lookups) != 1 {
				t.Fatalf("got %d lookups", len(lookups))
			}
			for _, target := range lookups[0].Execution.Targets {
				fact := target.Capacity
				if fact == nil || fact.Provenance != "caller_estimated" || fact.Required == nil || fact.Available == nil || *fact.AvailableSource != "configured" || fact.Fits != (*fact.Required <= *fact.Available) {
					t.Fatalf("target %s fact %s", target.ActionID[:1], mustJSON(fact))
				}
				if !fact.Fits && (fact.Reason == nil || *fact.Reason != CapacityContext) {
					t.Fatalf("an infeasible target needs its reason: %s", mustJSON(fact))
				}
				if target.ActionID == tc.want && !fact.Fits {
					t.Fatalf("selected target does not fit: %s", mustJSON(fact))
				}
			}
			wantLocal := 0
			if tc.want == localActionID {
				wantLocal = 1
			}
			if local.count() != wantLocal || cloud.count() != 1-wantLocal {
				t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
			}
		})
	}
}

func TestCapacityEstimateRefusesAPromptNoDeploymentHolds(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions", estimateCloudWindow)
	fixture := &v2Decisions{choose: fitPolicy}
	_, httpServer := v2Server(t, config, fixture)
	status, _, response := executeWithHeaders(t, httpServer.URL, bigTurn(promptOfAbout(1_100_000)), "too-big")
	if status != http.StatusBadRequest {
		t.Fatalf("status %d: %.300s", status, response)
	}
	var wire struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Error.Code != CodeContextExceedsAllDeployments {
		t.Fatalf("code %q", wire.Error.Code)
	}
	for _, part := range []string{"prompt is about", "(estimated)", "the largest is 1000000 tokens (kimi-k3)", "Compact or shorten the conversation"} {
		if !strings.Contains(wire.Error.Message, part) {
			t.Fatalf("message %q lacks %q", wire.Error.Message, part)
		}
	}
	if len(fixture.snapshot()) != 0 || local.count() != 0 || cloud.count() != 0 {
		t.Fatalf("refusal contacted Decisions=%d local=%d cloud=%d", len(fixture.snapshot()), local.count(), cloud.count())
	}
}

func TestCapacityEstimateServiceNoFeasibleTargetSaysCompact(t *testing.T) {
	// The local counter is down (runtime_unavailable, not a context fact), so
	// the core cannot refuse up front; the service then finds nothing fits.
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions", 200_000)
	config.Deployments[1].ContextWindow = 300_000
	fixture := &v2Decisions{choose: fitPolicy, refuse: func(contextV2) string { return "execution_no_feasible_target" }}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterPromptCounter(config.Deployments[0].ID, failingCounter{identity: promptIdentity(config.Deployments[0])}); err != nil {
		t.Fatal(err)
	}
	status, _, response := executeWithHeaders(t, httpServer.URL, bigTurn(promptOfAbout(600_000)), "service-refuses")
	if status != http.StatusBadRequest || !strings.Contains(string(response), CodeContextExceedsAllDeployments) || !strings.Contains(string(response), "the largest is 300000 tokens (kimi-k3)") {
		t.Fatalf("status %d: %.400s", status, response)
	}
	if len(fixture.snapshot()) != 1 || local.count() != 0 || cloud.count() != 0 {
		t.Fatalf("lookups=%d local=%d cloud=%d", len(fixture.snapshot()), local.count(), cloud.count())
	}
	// A refusal while some target fits stays the service's own failure.
	status, _, response = executeWithHeaders(t, httpServer.URL, bigTurn(promptOfAbout(50_000)), "service-refuses-small")
	if status != http.StatusBadGateway || !strings.Contains(string(response), "decision_failed") {
		t.Fatalf("status %d: %.400s", status, response)
	}
}

func TestCapacityEstimateHandoffOutsideCacheBoundariesNeverReturnsLocal(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions", estimateCloudWindow)
	// The service's model schedule is cache_boundaries: a tool-result turn is
	// not a model boundary, but a capacity handoff is still forced.
	fixture := &v2Decisions{choose: fitPolicy, mutate: func(response map[string]any) {
		response["receipt"].(map[string]any)["eligible"] = map[string]any{"model": "cache_boundaries", "thinking": "all_turns"}
	}}
	_, httpServer := v2Server(t, config, fixture)
	task := promptOfAbout(50_000)
	bigResult := `{"role":"tool","tool_call_id":"call_1","content":"` + strings.Repeat("lorem ipsum ", 15_217) + `"}`
	turns := []struct {
		body, executed, decidedBy, event string
	}{
		{bigTurn(task), localActionID, "schedule/schedule", EventTaskStart},
		{bigTurn(task, toolCall, bigResult), secondCloudActionID, "capacity_handoff/schedule", EventToolResult},
		{bigTurn(task, toolCall, bigResult, answer, followUp), secondCloudActionID, "schedule/schedule", EventUserMessage},
	}
	for index, turn := range turns {
		status, header, response := executeWithHeaders(t, httpServer.URL, turn.body, "spill-estimate")
		if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != turn.executed || header.Get("X-Rayline-Decided-By") != turn.decidedBy || !strings.HasPrefix(header.Get("X-Rayline-Event"), turn.event+"/") {
			t.Fatalf("turn %d status %d executed %q decided %q event %q: %.300s", index, status, header.Get("X-Rayline-Executed-Action-Id"), header.Get("X-Rayline-Decided-By"), header.Get("X-Rayline-Event"), response)
		}
	}
	lookups := fixture.snapshot()
	if len(lookups) != 3 {
		t.Fatalf("a pre-decision fact needs no re-consult; got %d lookups", len(lookups))
	}
	for index, lookup := range lookups[1:] {
		fact := lookup.Execution.Targets[0].Capacity
		if fact == nil || fact.Fits || *fact.Reason != CapacityContext || fact.Provenance != "caller_estimated" {
			t.Fatalf("turn %d local fact %s", index+1, mustJSON(fact))
		}
	}
	if served := lookups[2].Execution.LastServed; served == nil || served.ActionID != secondCloudActionID {
		t.Fatalf("last_served %+v", served)
	}
	if local.count() != 1 || cloud.count() != 2 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestEstimatePromptTokensChargesInlineMediaNotItsBytes(t *testing.T) {
	blob := strings.Repeat("iVBORw0KGgo", 30_000) // 330 kB of base64
	text := strings.Repeat("x", 3000)
	if got := estimatePromptTokens([]byte(text), nil); got != 1150 {
		t.Fatalf("text estimate %d", got)
	}
	messages := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + blob + `"}}`
	chat := `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + blob + `"}}`
	for name, body := range map[string]string{"messages": messages, "chat": chat} {
		got := estimatePromptTokens([]byte(body), nil)
		want := int64(float64(len(body)-len(blob))/3*1.15 + 4096*1.15)
		if got < want || got > want+2 {
			t.Fatalf("%s: estimate %d, want about %d", name, got, want)
		}
	}
	// A PDF keeps the conservative byte count: its pages can exceed one image.
	pdf := `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + blob + `"}}`
	if got := estimatePromptTokens([]byte(pdf), nil); got < int64(len(pdf)/3) {
		t.Fatalf("pdf was discounted: %d", got)
	}
	// The same payload quoted inside a string (a tool result) is text.
	escaped := `{"content":"{\"data\":\"` + blob + `\"} and \"data:image/png;base64,` + blob + `\""}`
	if got := estimatePromptTokens([]byte(escaped), nil); got < int64(len(escaped)/3) {
		t.Fatalf("escaped media was discounted: %d", got)
	}
	bytesPerToken, headroom, media := 4.0, 0.0, int64(100)
	override := &CapacityEstimate{BytesPerToken: &bytesPerToken, HeadroomRatio: &headroom, MediaTokens: &media}
	if got := estimatePromptTokens([]byte(text), override); got != 750 {
		t.Fatalf("override estimate %d", got)
	}
}

func TestCapacityEstimateConfigIsPerDeploymentAndValidated(t *testing.T) {
	var deployment Deployment
	if err := json.Unmarshal([]byte(`{"id":"local","capacity_estimate":{"bytes_per_token":2.5,"headroom_ratio":0.3,"media_tokens":8192}}`), &deployment); err != nil {
		t.Fatal(err)
	}
	if estimate := deployment.CapacityEstimate; estimate == nil || *estimate.BytesPerToken != 2.5 || *estimate.HeadroomRatio != 0.3 || *estimate.MediaTokens != 8192 {
		t.Fatalf("capacity_estimate %+v", deployment.CapacityEstimate)
	}
	t.Setenv("TEST_DECISIONS_KEY", "fixture-hosted-key-never-for-local")
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	bad := 0.0
	config := capacityEstimateConfig(t, "http://127.0.0.1:1/v1/chat/completions", "http://127.0.0.1:2/v1/chat/completions", estimateCloudWindow)
	config.Deployments[0].CapacityEstimate = &CapacityEstimate{BytesPerToken: &bad}
	config.Decisions.URL = "http://127.0.0.1:3/v1/routes"
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewServer(config, registry, provider, testCoreToken); err == nil || !strings.Contains(err.Error(), "capacity_estimate") {
		t.Fatalf("invalid estimate admitted: %v", err)
	}
}
