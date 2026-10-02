package extproc

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const subagentBillingSystem = `[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.287; cc_entrypoint=sdk-cli; cc_is_subagent=true;"},{"type":"text","text":"You are a search agent."}]`

// A subagent is its own episode only when the gateway keyed it on the
// harness's agent id; any other key (or none) keeps it a side call, as before.
func TestSubagentCallKindFollowsTheGatewayKeySource(t *testing.T) {
	system := json.RawMessage(subagentBillingSystem)
	for _, tc := range []struct {
		keySource  string
		wantKind   raylinearc.PolicyCallKind
		wantSource string
	}{
		{"agent", raylinearc.PolicyCallMain, callKindSourceClaudeSubagentOwn},
		{"task", raylinearc.PolicyCallSide, callKindSourceClaudeSubagent},
		{"role", raylinearc.PolicyCallSide, callKindSourceClaudeSubagent},
		{"", raylinearc.PolicyCallSide, callKindSourceClaudeSubagent},
	} {
		headers := map[string]string{raylineARCAgentKeySourceHeader: tc.keySource}
		kind, source := raylineARCPolicyCallKind(headers, policyFormatAnthropic, system, nil, nil, "e")
		if kind != tc.wantKind || source != tc.wantSource {
			t.Fatalf("key source %q: %s/%s, want %s/%s", tc.keySource, kind, source, tc.wantKind, tc.wantSource)
		}
	}
	// A subagent's own compaction helper stays a side call even when the
	// subagent is agent-keyed: only its conversation turns are promoted.
	compactionBody, _ := json.Marshal(map[string]any{"role": "user", "content": raylinearc.ClaudeCodeSummarizeDirective + "the work so far"})
	kind, source := raylineARCPolicyCallKind(map[string]string{raylineARCAgentKeySourceHeader: "agent"},
		policyFormatAnthropic, system, []json.RawMessage{compactionBody}, nil, "e")
	if kind != raylinearc.PolicyCallSide || source != callKindSourceClaudeCompactionRq {
		t.Fatalf("agent-keyed subagent compaction helper: %s/%s, want a side call", kind, source)
	}
	// The headers count only from a trusted gateway: a client cannot make
	// its own subagent an episode.
	untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{},
		map[string]string{raylineARCAgentKeySourceHeader: "agent"})
	if kind, _ := raylineARCPolicyCallKind(untrusted, policyFormatAnthropic, system, nil, nil, "e"); kind != raylinearc.PolicyCallSide {
		t.Fatalf("an untrusted key source made the subagent %s", kind)
	}
}

// The parent link is recorded hashed, as metadata; an unknown key source is
// reported as unknown.
func TestSubagentSignalRecordsTheParentLink(t *testing.T) {
	request := raylinearc.PolicyClientRequest{System: json.RawMessage(subagentBillingSystem)}
	signals := raylineARCPolicyTurnSignals(map[string]string{
		raylineARCAgentKeySourceHeader: "agent",
		raylineARCParentSessionHeader:  "user-1:conv-1",
		raylineARCParentAgentHeader:    "agent-parent",
	}, policyFormatAnthropic, request, nil, "e")
	if signals.Subagent == nil || !signals.Subagent.OwnEpisode() ||
		signals.Subagent.ParentEpisodeIDHash != raylinearc.HashEpisodeID("user-1:conv-1") ||
		signals.Subagent.ParentAgentIDHash != raylinearc.HashEpisodeID("agent-parent") {
		t.Fatalf("subagent signal = %+v", signals.Subagent)
	}
	unknown := raylineARCPolicyTurnSignals(map[string]string{}, policyFormatAnthropic, request, nil, "e")
	if unknown.Subagent == nil || unknown.Subagent.KeySource != "unknown" || unknown.Subagent.OwnEpisode() {
		t.Fatalf("unkeyed subagent signal = %+v", unknown.Subagent)
	}
	main := raylineARCPolicyTurnSignals(map[string]string{}, policyFormatAnthropic, raylinearc.PolicyClientRequest{}, nil, "e")
	if main.Subagent != nil {
		t.Fatalf("a main conversation request carried a subagent signal: %+v", main.Subagent)
	}
	// A codex subagent has no body claim; the gateway's key source alone
	// records its link, and it is served as the main turn it already was.
	codex := raylineARCPolicyTurnSignals(map[string]string{
		raylineARCAgentKeySourceHeader: "agent",
		raylineARCParentSessionHeader:  "user-1:root-thread",
	}, policyFormatResponses, raylinearc.PolicyClientRequest{}, nil, "e")
	if codex.Subagent == nil || codex.Subagent.ParentEpisodeIDHash != raylinearc.HashEpisodeID("user-1:root-thread") ||
		codex.CallKind == raylinearc.PolicyCallSide {
		t.Fatalf("codex subagent signals = %+v", codex)
	}
}

// End to end on a strict cell trusting its gateway: three concurrent
// subagents, each keyed on its agent id, are three episodes deciding at the
// same time as main turns, and the parent's episode is untouched.
func TestConcurrentSubagentsAreSeparateEpisodes(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	path := writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "          policy_service:\n"
	rendered := strings.Replace(string(raw), anchor, anchor+"            trust_turn_signal_headers: true\n", 1)
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.holdUntilConcurrent(3)

	agents := []string{"agent-a", "agent-b", "agent-c"}
	statuses := make([]int, len(agents))
	errs := make([]error, len(agents))
	contexts := make([]*RequestContext, len(agents))
	var wg sync.WaitGroup
	for i, agent := range agents {
		wg.Add(1)
		go func() {
			defer wg.Done()
			contexts[i], statuses[i], errs[i] = subagentTurn(router, "user-1:conv-1:"+agent, agent)
		}()
	}
	wg.Wait()
	for i := range agents {
		if errs[i] != nil || statuses[i] != 0 {
			t.Fatalf("subagent %s: status %d err %v, want forwarded", agents[i], statuses[i], errs[i])
		}
		// A main turn of its own episode: it holds that episode's
		// transaction and will commit its own ledger, unlike a side call.
		if transaction := contexts[i].RaylineARCTransaction; transaction == nil || transaction.sideCall {
			t.Fatalf("subagent %s was served as a side call, want a main turn of its own episode", agents[i])
		}
	}
	if peak := fake.peakConcurrency(); peak != len(agents) {
		t.Fatalf("%d decides in flight at once, want %d", peak, len(agents))
	}
	episodes := map[string]bool{}
	for _, decide := range fake.received() {
		episodes[decide.EpisodeIDHash] = true
	}
	parent := raylinearc.HashEpisodeID("user-1:conv-1")
	if len(episodes) != len(agents) || episodes[parent] {
		t.Fatalf("decide episodes = %v, want %d distinct and none the parent's", episodes, len(agents))
	}
	snapshots, ok := router.RaylineARCEpisodeStore.(raylinearc.EpisodeSnapshotStore)
	if !ok {
		t.Fatal("the episode store reads no snapshot")
	}
	if _, read, err := snapshots.Snapshot(context.Background(), parent, 4); err != nil || read.Version() != 0 {
		t.Fatalf("the parent episode changed: version %d err %v", read.Version(), err)
	}
}

// subagentTurn sends one Claude Code subagent request as the gateway does:
// the episode keyed on the agent id, the key source and parent stated.
func subagentTurn(router *OpenAIRouter, episode, agent string) (*RequestContext, int, error) {
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "subagent-" + agent,
		TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
			{Key: raylineARCAgentKeySourceHeader, Value: "agent"},
			{Key: raylineARCParentSessionHeader, Value: "user-1:conv-1"},
		}},
	}}
	if _, err := router.handleRequestHeaders(headers, ctx); err != nil {
		return ctx, 0, err
	}
	body := `{"model":"auto","max_tokens":1024,"system":` + subagentBillingSystem +
		`,"messages":[{"role":"user","content":"search for ` + agent + `"}]}`
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx)
	if err != nil {
		return ctx, 0, err
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return ctx, int(immediate.GetStatus().GetCode()), nil
	}
	return ctx, 0, nil
}
