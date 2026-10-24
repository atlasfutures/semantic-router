package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func redactedResponse(data string) *llmprotocol.Response {
	return &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "redacted_thinking",
			Raw: []byte(`{"type":"redacted_thinking","data":"` + data + `"}`),
		}},
		{Kind: llmprotocol.ContentText, Text: "answer"},
	}}}}
}

// A committed turn records which worker issued the opaque reasoning its
// response held.
func TestCommittedTurnRecordsOpaqueReasoningProvenance(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	transaction.dispatchWorker = "gpt-worker"
	response := redactedResponse("gpt-blob")
	response.Output[0].Content = append(response.Output[0].Content, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Signature: "gpt-signature", Reasoning: llmprotocol.ReasoningScopeText})
	if err := transaction.commit(context.Background(), &RequestContext{SemanticResponse: response}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	next, reread, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Abort(context.Background(), next) }()
	if !reasoning.ReasoningIssuedElsewhere(reread.ReasoningProvenance, "gpt-blob", "claude") ||
		reasoning.ReasoningIssuedElsewhere(reread.ReasoningProvenance, "gpt-blob", "gpt-worker") ||
		!reasoning.ReasoningIssuedBy(reread.ReasoningProvenance, "gpt-signature", "gpt-worker") {
		t.Fatalf("provenance after commit = %+v", reread.ReasoningProvenance)
	}
}

// End to end: a block the episode recorded as another Messages worker's is
// dropped from the history sent to this one, and kept for its issuer.
func TestOpaqueReasoningReachesOnlyItsIssuer(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	history := `{"model":"auto","max_tokens":4096,"messages":[` +
		`{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":[{"type":"redacted_thinking","data":"issuer-blob"},{"type":"text","text":"Looking at it."}]},` +
		`{"role":"user","content":"go on"}]}`
	for _, tc := range []struct {
		action   string
		wantKept bool
	}{
		{"claude", false},
		{"claude-off", true},
	} {
		t.Run(tc.action, func(t *testing.T) {
			episode := "episode-provenance-" + tc.action
			seedOpaqueReasoningIssuer(t, router, episode, "claude-off", "issuer-blob")
			action := actions[tc.action].ActionID
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
			body := dispatchPolicyClientRequest(t, router, episode, "/v1/messages", history)
			if kept := strings.Contains(string(body["messages"]), "issuer-blob"); kept != tc.wantKept {
				t.Fatalf("worker %s: block kept = %v, want %v: %s", tc.action, kept, tc.wantKept, body["messages"])
			}
			if !strings.Contains(string(body["messages"]), "Looking at it.") {
				t.Fatalf("the assistant's visible text was lost: %s", body["messages"])
			}
		})
	}
}

// seedOpaqueReasoningIssuer commits an episode whose record says worker, as
// the router would dispatch it, issued the block holding data.
func seedOpaqueReasoningIssuer(t *testing.T, router *OpenAIRouter, episode, worker, data string) {
	t.Helper()
	dispatch, err := router.resolveProviderDispatch(worker, "", false, llmprotocol.AnthropicMessagesV1)
	if err != nil {
		t.Fatal(err)
	}
	issuer := router.opaqueReasoningIssuerFor(dispatch)
	if issuer == reasoning.ReasoningIssuerUnknown {
		t.Fatalf("worker %s has no fixed issuer in the test config", worker)
	}
	store := router.RaylineARCEpisodeStore
	lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), len(router.Config.Decisions[0].ModelRefs))
	if err != nil {
		t.Fatal(err)
	}
	state.ReasoningProvenance = reasoning.WithReasoningProvenance(state.ReasoningProvenance, issuer, []string{data})
	if err := store.Commit(context.Background(), lease, lease.Version(), state); err != nil {
		t.Fatal(err)
	}
}

// On OpenRouter the issuer is fixed only by a single pinned provider with no
// fallbacks; otherwise OpenRouter may serve the next turn elsewhere.
func TestOpaqueReasoningIssuerOnOpenRouter(t *testing.T) {
	on, off := true, false
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	dispatch := &providerDispatch{
		logicalModel: "gpt", upstreamModel: "openai/gpt-5.6",
		profile: &config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"},
	}
	if issuer := router.opaqueReasoningIssuerFor(dispatch); issuer != reasoning.ReasoningIssuerUnknown {
		t.Fatalf("unpinned OpenRouter issuer = %q, want unknown", issuer)
	}
	for _, tc := range []struct {
		pin  config.OpenRouterProviderPreferences
		want bool
	}{
		{config.OpenRouterProviderPreferences{Order: []string{"openai"}, AllowFallbacks: &off}, true},
		{config.OpenRouterProviderPreferences{Order: []string{"openai"}, AllowFallbacks: &on}, false},
		{config.OpenRouterProviderPreferences{Order: []string{"openai", "azure"}, AllowFallbacks: &off}, false},
	} {
		router.Config = routerConfigWithPin("gpt", tc.pin)
		known := router.opaqueReasoningIssuerFor(dispatch) != reasoning.ReasoningIssuerUnknown
		if known != tc.want {
			t.Fatalf("pin %+v: issuer known = %v, want %v", tc.pin, known, tc.want)
		}
	}
}

func routerConfigWithPin(model string, pin config.OpenRouterProviderPreferences) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{model: {ProviderPreferences: &pin}}
	return cfg
}

// A Messages-compatible endpoint can serve GPT. The dispatched payload must
// keep only that worker's recorded private reasoning, including when history
// alternates Claude and GPT. Unknown ciphertext is never proof of compatibility.
func TestMixedMessagesReasoningReachesOnlyItsRecordedIssuer(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	path := writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rendered := string(raw)
	swaps := [][2]string{
		{"    - name: claude-off\n      provider_model_id: anthropic/claude-opus-5\n      api_format: anthropic", "    - name: claude-off\n      provider_model_id: openai/gpt-6-luna\n      api_format: anthropic\n      provider_preferences:\n        order: [openai]\n        only: [openai]\n        allow_fallbacks: false"},
		{"        - name: anthropic-claude-off\n          base_url: https://api.anthropic.com\n          provider: anthropic", "        - name: anthropic-claude-off\n          base_url: https://openrouter.ai/api/v1\n          provider: openrouter"},
	}
	for _, swap := range swaps {
		if !strings.Contains(rendered, swap[0]) {
			t.Fatalf("fixture missing %q", swap[0])
		}
		rendered = strings.Replace(rendered, swap[0], swap[1], 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatal(err)
	}
	awaitPolicySelectorArmed(t, router)
	history := `{"model":"auto","max_tokens":4096,"messages":[
  {"role":"user","content":"fix the test"},
  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"claude-signed"},{"type":"redacted_thinking","data":"claude-opaque"},{"type":"text","text":"Claude visible answer"}]},
  {"role":"user","content":"continue"},
  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"gpt-signed"},{"type":"redacted_thinking","data":"gpt-opaque"},{"type":"text","text":"GPT visible answer"}]},
  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"claude-signed-alone"},{"type":"text","text":"Claude continuation"}]},
  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"gpt-signed-alone"},{"type":"text","text":"GPT continuation"}]},
  {"role":"user","content":"continue"},
  {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"unknown-signed"},{"type":"redacted_thinking","data":"unknown-opaque"},{"type":"text","text":"Unknown visible answer"},{"type":"tool_use","id":"tool_1","name":"read","input":{}}]},
  {"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":"tool output"}]}
 ]}`
	for _, tc := range []struct {
		target, keep, drop string
		unknownPrefix      bool
	}{
		{"claude", "claude", "gpt", false}, {"claude-off", "gpt", "claude", false},
		{"claude", "claude", "gpt", true}, {"claude-off", "gpt", "claude", true},
	} {
		t.Run(fmt.Sprintf("%s/unknown-prefix=%v", tc.target, tc.unknownPrefix), func(t *testing.T) {
			episode := fmt.Sprintf("mixed-reasoning-%s-%v", tc.target, tc.unknownPrefix)
			for _, origin := range []struct{ worker, prefix string }{{"claude", "claude"}, {"claude-off", "gpt"}} {
				for _, suffix := range []string{"signed", "opaque", "signed-alone"} {
					seedOpaqueReasoningIssuer(t, router, episode, origin.worker, origin.prefix+"-"+suffix)
				}
			}
			action := relaxedPolicyActions()[tc.target].ActionID
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
			request := history
			if tc.unknownPrefix {
				request = strings.Replace(request, `{"role":"user","content":"fix the test"},`, `{"role":"user","content":"fix the test"},{"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"unknown-prefix"},{"type":"text","text":"older answer"}]},`, 1)
			}
			body := dispatchPolicyClientRequest(t, router, episode, "/v1/messages", request)
			wire := string(body["messages"])
			for _, suffix := range []string{"signed", "opaque", "signed-alone"} {
				if kept := strings.Contains(wire, tc.keep+"-"+suffix); kept == tc.unknownPrefix {
					t.Fatalf("compatible block kept=%v after unknown prefix=%v: %s", kept, tc.unknownPrefix, wire)
				}
				if strings.Contains(wire, tc.drop+"-"+suffix) {
					t.Fatalf("foreign encrypted reasoning reached target: %s", wire)
				}
			}
			if strings.Contains(wire, "unknown-signed") || strings.Contains(wire, "unknown-opaque") {
				t.Fatalf("unrecorded ciphertext reached target: %s", wire)
			}
			for _, visible := range []string{"Claude visible answer", "GPT visible answer", "Unknown visible answer", "tool_1", "tool output"} {
				if !strings.Contains(wire, visible) {
					t.Fatalf("visible history or tool relationship lost %q: %s", visible, wire)
				}
			}
		})
	}
}

// The synthetic renderer corpus tests placement, so it declares a fixed
// provider and the earlier issuance of its thinking instead of relying on
// the old behavior that forwarded unrecorded ciphertext.
func fixturePinnedBody(t *testing.T, body []byte) []byte {
	t.Helper()
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	wire["provider"] = json.RawMessage(`{"order":["fixture"],"allow_fallbacks":false,"only":["fixture"]}`)
	encoded, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func seedMessagesHistoryIssuer(t *testing.T, router *OpenAIRouter, episode, worker string, body []byte) {
	t.Helper()
	var history struct {
		Messages []struct{ Content json.RawMessage }
	}
	if err := json.Unmarshal(body, &history); err != nil {
		t.Fatal(err)
	}
	for _, message := range history.Messages {
		var blocks []struct{ Type, Signature, Data string }
		if len(message.Content) == 0 || message.Content[0] != '[' {
			continue
		}
		if err := json.Unmarshal(message.Content, &blocks); err != nil {
			t.Fatal(err)
		}
		for _, block := range blocks {
			data := block.Signature
			if block.Type == "redacted_thinking" {
				data = block.Data
			}
			if data != "" {
				seedOpaqueReasoningIssuer(t, router, episode, worker, data)
			}
		}
	}
}
