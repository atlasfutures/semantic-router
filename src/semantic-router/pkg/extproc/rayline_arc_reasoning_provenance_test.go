package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	if err := transaction.commit(context.Background(), &RequestContext{SemanticResponse: redactedResponse("gpt-blob")}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	next, reread, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Abort(context.Background(), next) }()
	if !raylinearc.ReasoningIssuedElsewhere(reread.ReasoningProvenance, "gpt-blob", "claude") ||
		raylinearc.ReasoningIssuedElsewhere(reread.ReasoningProvenance, "gpt-blob", "gpt-worker") {
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
	if issuer == raylinearc.ReasoningIssuerUnknown {
		t.Fatalf("worker %s has no fixed issuer in the test config", worker)
	}
	store := router.RaylineARCEpisodeStore
	lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), 4)
	if err != nil {
		t.Fatal(err)
	}
	state.ReasoningProvenance = raylinearc.WithReasoningProvenance(nil, issuer, []string{data})
	if err := state.Commit(0, 10, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
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
	if issuer := router.opaqueReasoningIssuerFor(dispatch); issuer != raylinearc.ReasoningIssuerUnknown {
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
		known := router.opaqueReasoningIssuerFor(dispatch) != raylinearc.ReasoningIssuerUnknown
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
