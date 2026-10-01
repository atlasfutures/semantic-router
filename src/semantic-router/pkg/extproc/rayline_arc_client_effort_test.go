package extproc

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const clientEffortWorker = "effort@default"

// clientEffortRequest is a Claude Code turn whose client states a per-message
// effort on a content-less system message that exists only to carry it.
func clientEffortRequest() *llmprotocol.Request {
	return &llmprotocol.Request{
		Model: "auto", Generation: 1, Sampling: llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(1024)},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "fix it"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "looking"}}},
			{Role: llmprotocol.RoleSystem, ReasoningEffort: "max"},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go on"}}},
		},
	}
}

func clientEffortDecision(kind string) *config.Decision {
	switch kind {
	case "non-arc":
		return &config.Decision{Name: "plain"}
	case "arc":
		return &config.Decision{Name: "arc", Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{},
		}}
	}
	// ARC with a per_turn_effort lever bound to the dispatched worker.
	return &config.Decision{Name: "arc", Algorithm: &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType, OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{ThinkingLever: &config.RaylineARCThinkingLeverConfig{
			Enabled: true, Source: config.RaylineARCThinkingSourceRule, Level: "low",
			Workers: map[string]config.RaylineARCThinkingBindingConfig{clientEffortWorker: {
				Admission: config.RaylineARCThinkingAdmissionCertified, Lever: "per_turn_effort", Emit: "every_turn",
				Placements: []string{"system_before_governed_turn", "system_after_tool_run"},
				Levels: []config.RaylineARCThinkingLevelConfig{
					{Level: "low", Rank: -1, Effort: "low"},
					{Level: "base", Rank: 0, Effort: "high"},
				},
			}},
		}},
	}}
}

// dispatchClientEffort renders the turn for one target through the dispatch
// path (prepareProviderDispatch, encodeDispatchRequest, adaptProviderRequest).
func dispatchClientEffort(t *testing.T, target llmprotocol.WireFormat, kind string) []byte {
	t.Helper()
	router, model := routingTestRouterForFormat(target)
	if target == llmprotocol.OpenAIChatV1 {
		router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}
	}
	request := clientEffortRequest()
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.VSRSelectedDecision = clientEffortDecision(kind)
	if kind == "lever" {
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
		ctx.RaylineARCTransaction = transaction
		ctx.RaylineARCDispatch = &raylinearc.WorkerManifest{ID: clientEffortWorker}
		t.Cleanup(func() { _ = transaction.abort(context.Background(), "test") })
	}
	dispatch, err := router.prepareProviderDispatch(request, model, ctx.VSRSelectedDecision.Name, true, ctx)
	if err != nil {
		t.Fatalf("prepare dispatch: %v", err)
	}
	router.bindDispatchProviderFacts(dispatch, ctx)
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	body, err := router.adaptProviderRequest(encoded, dispatch, ctx)
	if err != nil {
		t.Fatalf("adapt: %v", err)
	}
	return body
}

// On a rayline_arc turn only the lever's per-message effort travels: the
// client's is cleared (and its carrier message dropped), under
// provider_default and with the lever on. A non-ARC decision keeps
// upstream's behaviour and forwards the client's effort.
func TestRaylineARCTurnSendsOnlyTheLeversPerMessageEffort(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			t.Run("provider_default sends no effort", func(t *testing.T) {
				body := dispatchClientEffort(t, target, "arc")
				if bytes.Contains(body, []byte(`"max"`)) || bytes.Contains(body, []byte("configuration_update")) {
					t.Fatalf("the client's per-message effort reached the provider: %s", body)
				}
			})
			t.Run("lever on sends only the lever's effort", func(t *testing.T) {
				body := dispatchClientEffort(t, target, "lever")
				if bytes.Contains(body, []byte(`"max"`)) {
					t.Fatalf("the client's per-message effort reached the provider: %s", body)
				}
				if !bytes.Contains(body, []byte(`"low"`)) {
					t.Fatalf("the lever's effort did not reach the provider: %s", body)
				}
			})
			t.Run("non-ARC forwards the client effort", func(t *testing.T) {
				body := dispatchClientEffort(t, target, "non-arc")
				if !bytes.Contains(body, []byte(`"max"`)) {
					t.Fatalf("a non-ARC decision lost the client's per-message effort: %s", body)
				}
			})
		})
	}
}

// A direct-model Messages request (no decision) gets no OpenRouter routing
// members; a decision's request does.
func TestMessagesProviderRoutingNeedsADecision(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}
	params := router.Config.ModelConfig[model]
	params.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: []string{"pinned"}}
	router.Config.ModelConfig[model] = params
	profile := router.Config.ProviderProfiles["provider"]
	body := []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	for name, tc := range map[string]struct {
		decision string
		want     bool
	}{
		"direct model": {"", false},
		"decision":     {"arc", true},
	} {
		t.Run(name, func(t *testing.T) {
			dispatch := &providerDispatch{logicalModel: model, profile: &profile, targetFormat: llmprotocol.AnthropicMessagesV1, decisionName: tc.decision}
			adapted, err := router.adaptProviderRequest(body, dispatch, &RequestContext{})
			if err != nil {
				t.Fatal(err)
			}
			if got := bytes.Contains(adapted, []byte(`"provider"`)); got != tc.want {
				t.Fatalf("routing members present = %v, want %v: %s", got, tc.want, adapted)
			}
		})
	}
}
