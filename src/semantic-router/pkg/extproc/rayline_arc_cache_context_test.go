package extproc

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The decide request carries each recently served action's last call, read
// from the episode's warmth; an arm served over an hour ago is left out.
func TestPolicyDecideCarriesTheEpisodeCacheContext(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[1].ActionID })
	now := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	fixture.selector.now = func() time.Time { return now }
	prompt, read := 1000, 900
	state, _ := raylinearc.NewEpisodeState(2)
	state.Warmth[0] = &raylinearc.WorkerWarmth{LastUsed: now.Add(-30 * time.Second), LastInputTokens: 700, Call: &raylinearc.CallCache{
		ActionID: bindings[1].ActionID, TTLSeconds: 300, TTLSource: "anthropic cache_control ephemeral 5m",
		PromptTokens: &prompt, CacheReadTokens: &read,
	}}
	state.Warmth[1] = &raylinearc.WorkerWarmth{LastUsed: now.Add(-2 * time.Hour), LastInputTokens: 700, Call: &raylinearc.CallCache{
		ActionID: bindings[2].ActionID,
	}}

	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0].CacheContext
	if sent == nil {
		t.Fatal("the decide request carried no cache_context")
	}
	encoded, _ := json.Marshal(sent)
	want := `{"actions":[{"action_id":"` + bindings[1].ActionID + `","gap_s":30,"ttl_s":300,` +
		`"ttl_source":"anthropic cache_control ephemeral 5m","prompt_tokens":1000,"cache_read_tokens":900,"cache_write_tokens":null}]}`
	if string(encoded) != want {
		t.Fatalf("cache_context\n got %s\nwant %s", encoded, want)
	}
}

// A first turn sends an empty list: the field is on every decide.
func TestPolicyDecideSendsAnEmptyCacheContextOnAFreshEpisode(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	state, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select: %v", err)
	}
	if sent := fixture.fake.received()[0].CacheContext; sent == nil || sent.Actions == nil || len(sent.Actions) != 0 {
		t.Fatalf("cache_context = %+v, want an empty list", sent)
	}
}

// The turn's commit records the call on the served arm: its action, the
// wire TTL and the provider's counts, an unreported leg unknown.
func TestPolicyTurnCommitRecordsTheCall(t *testing.T) {
	newTransaction := func() *raylineARCEpisodeTransaction {
		state, _ := raylinearc.NewEpisodeState(2)
		return &raylineARCEpisodeTransaction{state: state, selectedArm: 1, serializedTokens: 640}
	}
	action := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

	transaction := newTransaction()
	transaction.call.ActionID = action
	transaction.stageWireCacheTTL(3600, "anthropic cache_control ephemeral 1h")
	transaction.stageCallUsage(responseUsageMetrics{
		promptTokens: 1200, promptTokensReported: true,
		cachedPromptTokens: 0, cachedPromptTokensReported: true,
	})
	next, err := transaction.nextState()
	if err != nil {
		t.Fatal(err)
	}
	call := next.Warmth[1].Call
	if call == nil || call.ActionID != action || call.TTLSeconds != 3600 || *call.PromptTokens != 1200 ||
		call.CacheReadTokens == nil || *call.CacheReadTokens != 0 || call.CacheWriteTokens != nil {
		t.Fatalf("recorded call = %+v", call)
	}
	if next.Warmth[1].LastInputTokens != 640 {
		t.Fatalf("LastInputTokens = %d", next.Warmth[1].LastInputTokens)
	}

	// Counts the Router estimated, or could not read, are unknown.
	for name, usage := range map[string]responseUsageMetrics{
		"estimated": {promptTokens: 50, promptTokensReported: true, estimated: true},
		"invalid":   {promptTokens: 50, promptTokensReported: true, invalid: true},
	} {
		transaction = newTransaction()
		transaction.call.ActionID = action
		transaction.stageCallUsage(responseUsageMetrics{promptTokens: 9, promptTokensReported: true})
		transaction.stageCallUsage(usage)
		next, err = transaction.nextState()
		if err != nil {
			t.Fatal(err)
		}
		if call := next.Warmth[1].Call; call.PromptTokens != nil || call.CacheReadTokens != nil || call.CacheWriteTokens != nil {
			t.Fatalf("%s usage recorded %+v", name, call)
		}
	}

	// No action (a turn outside policy mode) records no call.
	next, err = newTransaction().nextState()
	if err != nil {
		t.Fatal(err)
	}
	if next.Warmth[1].Call != nil {
		t.Fatalf("a turn with no action recorded %+v", next.Warmth[1].Call)
	}
}

// Only an Anthropic model's dispatched cache_control is its TTL; another
// model's is unstated, and a turn outside policy mode records nothing.
func TestPolicyTurnStagesTheWireTTLOfAnAnthropicDispatch(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`)
	router := &OpenAIRouter{}
	for model, want := range map[string]int{"anthropic/claude-sonnet-4.5": 3600, "openai/gpt-5": 0} {
		transaction := &raylineARCEpisodeTransaction{commitOnCompletion: true}
		transaction.call.TTLSeconds, transaction.call.TTLSource = 300, "stale"
		ctx := &RequestContext{RaylineARCDispatch: &raylinearc.WorkerManifest{ID: model}, RaylineARCTransaction: transaction}
		stageRaylineARCWireCacheTTL(router, body, ctx)
		if transaction.call.TTLSeconds != want || (want == 0) != (transaction.call.TTLSource == "") {
			t.Fatalf("%s staged %d %q, want %d", model, transaction.call.TTLSeconds, transaction.call.TTLSource, want)
		}
	}
	transaction := &raylineARCEpisodeTransaction{}
	stageRaylineARCWireCacheTTL(router, body, &RequestContext{
		RaylineARCDispatch: &raylinearc.WorkerManifest{ID: "anthropic/claude-sonnet-4.5"}, RaylineARCTransaction: transaction,
	})
	if transaction.call.TTLSeconds != 0 {
		t.Fatalf("an artifact-mode turn staged %d", transaction.call.TTLSeconds)
	}
}
