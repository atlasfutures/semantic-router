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

// The clearing runs before the lever, so the lever's own writes survive it:
// under #118's on_change_v1 emit, a return to neutral writes the neutral
// marker at the tail, and it travels while the client's per-message effort
// (and its content-less carrier) does not. Runs through applyDispatchDecision,
// where the two are ordered.
func TestClientEffortClearingLeavesTheLeversOnChangeV1MarkerInPlace(t *testing.T) {
	const marker = "Until the next steering instruction, use your normal judgement."
	e := newLeverEpisode(t, true)
	lever := e.decision.Algorithm.RaylineARC.ThinkingLever
	binding := lever.Workers[leverWorker]
	binding.Emit, binding.NeutralLevel, binding.NeutralText = "on_change_v1", "none", marker
	lever.Workers[leverWorker] = binding

	turn0 := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "fix it")}
	e.turn(leverWorker, turn0, true) // steer in force

	lever.Level = "none"
	turn1 := append(append([]llmprotocol.Message(nil), turn0...),
		leverText(llmprotocol.RoleAssistant, "done"),
		llmprotocol.Message{Role: llmprotocol.RoleSystem, ReasoningEffort: "max"},
		leverText(llmprotocol.RoleUser, "next"))
	lease, state, err := e.store.Prepare(context.Background(), e.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(e.store, lease, state, e.episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	t.Cleanup(func() { _ = transaction.abort(context.Background(), "test") })
	ctx := &RequestContext{
		Headers: map[string]string{}, VSRSelectedDecision: e.decision,
		RaylineARCDispatch: &raylinearc.WorkerManifest{ID: leverWorker}, RaylineARCTransaction: transaction,
	}
	request := &llmprotocol.Request{Model: "m", Generation: 1, Messages: turn1}
	router, _ := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	dispatch := &providerDispatch{logicalModel: "m", targetFormat: llmprotocol.OpenAIChatV1, decisionName: e.decision.Name}
	if _, err := router.applyDispatchDecision(request, dispatch, ctx); err != nil {
		t.Fatal(err)
	}
	tail := request.Messages[len(request.Messages)-1].Content
	if tail[len(tail)-1].Text != marker {
		t.Fatalf("the lever's neutral marker did not survive: %+v", request.Messages)
	}
	if ctx.RaylineARCThinking == nil || ctx.RaylineARCThinking.Written != "neutral_marker" {
		t.Fatalf("the lever did not write the marker: %+v", ctx.RaylineARCThinking)
	}
	for index, message := range request.Messages {
		if message.ReasoningEffort != "" || len(message.Content) == 0 {
			t.Fatalf("message %d still carries the client's effort: %+v", index, message)
		}
	}
}

// Only a content-less system message is a carrier. A content-less user or
// assistant message keeps its place and loses only the effort, so dropping it
// cannot join two same-role turns on Messages.
func TestClientEffortClearingDropsOnlyTheSystemCarrier(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a"}}},
		{Role: llmprotocol.RoleAssistant, ReasoningEffort: "max"},
		{Role: llmprotocol.RoleSystem, ReasoningEffort: "max"},
		{Role: llmprotocol.RoleUser, ReasoningEffort: "max"},
	}}
	ctx := &RequestContext{VSRSelectedDecision: clientEffortDecision("arc")}
	if !clearClientMessageEffortForARC(request, ctx) {
		t.Fatal("nothing was cleared")
	}
	roles := []llmprotocol.Role{}
	for _, message := range request.Messages {
		if message.ReasoningEffort != "" {
			t.Fatalf("an effort survived: %+v", message)
		}
		roles = append(roles, message.Role)
	}
	want := []llmprotocol.Role{llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleUser}
	if len(roles) != len(want) || roles[0] != want[0] || roles[1] != want[1] || roles[2] != want[2] {
		t.Fatalf("roles = %v, want %v (only the system carrier dropped)", roles, want)
	}
}

// Shadow requests on an ARC turn are copied from the request after
// applyDispatchDecision cleared the client's per-message effort, so a
// Messages shadow target (which renders per-message effort) receives none.
func TestShadowDispatchOnAnARCTurnCarriesNoClientEffort(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	router.Config.ProviderProfiles["shadow-anthropic"] = config.ProviderProfile{Type: "anthropic", BaseURL: backend.server.URL}
	for i := range router.Config.VLLMEndpoints {
		if router.Config.VLLMEndpoints[i].Name == "shadow-backend" {
			router.Config.VLLMEndpoints[i].ProviderProfileName = "shadow-anthropic"
		}
	}
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.APIFormat = config.APIFormatAnthropic
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
		ctx.VSRSelectedDecision.Algorithm = &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{},
		}
		ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages,
			llmprotocol.Message{Role: llmprotocol.RoleSystem, ReasoningEffort: "max"},
			llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go on"}}})
	})
	waitForShadow(t, router)
	if backend.requestCount() != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", backend.requestCount())
	}
	if bytes.Contains(backend.bodies[0], []byte("configuration_update")) || bytes.Contains(backend.bodies[0], []byte(`"max"`)) {
		t.Fatalf("the shadow body carries the client's per-message effort: %s", backend.bodies[0])
	}
}
