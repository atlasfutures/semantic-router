package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyDispatchContext is a request the policy service routed to action.
func policyDispatchContext(decision *config.Decision, action config.RaylineARCPolicyBinding) *RequestContext {
	return &RequestContext{
		Headers:             map[string]string{},
		VSRSelectedDecision: decision,
		VSRRaylineARC:       &selection.RaylineARCTrace{PolicyActionID: action.ActionID, ThinkingLevel: action.Level},
		RaylineARCDispatch:  &raylinearc.WorkerManifest{ID: action.Worker},
	}
}

func policyDecisionWithActions(actions ...config.RaylineARCPolicyBinding) *config.Decision {
	decision := policyTestDecision("")
	decision.Algorithm.RaylineARC.PolicyService.Bindings = actions
	return decision
}

func policyAction(worker, level, model string, effort *string, budget *int64, suffix string) config.RaylineARCPolicyBinding {
	return config.RaylineARCPolicyBinding{
		ActionID: config.RaylineARCPolicyActionID(model, effort, budget, suffix),
		Worker:   worker, Level: level, Model: model, Effort: effort, ReasoningMaxTokens: budget,
	}
}

// On Chat the action's native reasoning replaces whatever the router derived,
// on the thinking-off arm too: the derived bound would otherwise carry the
// client's allowance and drop the action's effort.
func TestPolicyActionReasoningReachesTheChatWire(t *testing.T) {
	budget := int64(4096)
	cases := map[string]struct {
		action       config.RaylineARCPolicyBinding
		useReasoning bool
		want         string
	}{
		"effort":      {policyAction("think", "none", "vendor/think", policyTestEffort("high"), nil, ""), true, `{"effort":"high"}`},
		"budget":      {policyAction("think", "none", "vendor/think", nil, &budget, ""), true, `{"max_tokens":4096}`},
		"null effort": {policyAction("think", "none", "vendor/think", nil, nil, ""), true, ``},
		// The thinking-off action keeps the off signal the router derived for
		// its use_reasoning:false worker, as prod sends it; the fixture body's
		// derived controls are left as they were.
		"thinking off":   {policyAction("off", "none", "vendor/off", policyTestEffort("none"), nil, ""), false, `{"max_tokens":32000}`},
		"steered effort": {policyAction("think", "up", "vendor/think", policyTestEffort("low"), nil, policyTestUp), true, `{"effort":"low"}`},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := policyDispatchContext(policyDecisionWithActions(test.action), test.action)
			dispatch := &providerDispatch{
				logicalModel: test.action.Worker, targetFormat: llmprotocol.OpenAIChatV1, useReasoning: test.useReasoning,
				profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
			}
			body, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatch, ctx)
			if err != nil {
				t.Fatal(err)
			}
			reasoning, effort := reasoningControls(t, body)
			if name == "thinking off" {
				if string(body) != derivedThinkingBody {
					t.Fatalf("the thinking-off action rewrote the derived body: %s", body)
				}
				return
			}
			if string(reasoning) != test.want || effort {
				t.Fatalf("reasoning = %s (reasoning_effort present: %v), want %s", reasoning, effort, test.want)
			}
			if ctx.RaylineARCWorkerThinking == nil || ctx.RaylineARCWorkerThinking.Level != raylineARCPolicyActionLevel {
				t.Fatalf("recorded base = %+v", ctx.RaylineARCWorkerThinking)
			}
		})
	}
}

// An OpenAI-compatible binding -- the catalog's top-level transport, even one
// that reaches OpenRouter -- carries the action's effort as reasoning_effort.
// A budget has no top-level shape, so it fails the turn rather than send
// another action.
func TestPolicyActionReasoningOnATopLevelTransport(t *testing.T) {
	topLevel := &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1", ReasoningTransport: "top_level_effort"}
	dispatchTo := func(worker string, useReasoning bool) *providerDispatch {
		return &providerDispatch{logicalModel: worker, targetFormat: llmprotocol.OpenAIChatV1, useReasoning: useReasoning, profile: topLevel}
	}
	effort := policyAction("think", "up", "think-trained", policyTestEffort("xhigh"), nil, policyTestUp)
	body, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatchTo("think", true),
		policyDispatchContext(policyDecisionWithActions(effort), effort))
	if err != nil {
		t.Fatal(err)
	}
	reasoning, _ := reasoningControls(t, body)
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if reasoning != nil || string(wire["reasoning_effort"]) != `"xhigh"` {
		t.Fatalf("reasoning = %s, reasoning_effort = %s", reasoning, wire["reasoning_effort"])
	}

	budget := int64(4096)
	budgeted := policyAction("think", "none", "think-trained", nil, &budget, "")
	if _, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatchTo("think", true),
		policyDispatchContext(policyDecisionWithActions(budgeted), budgeted)); !errors.Is(err, errPolicyActionFormat) {
		t.Fatalf("budget on a top-level transport: error = %v, want errPolicyActionFormat", err)
	}
}

// A transport that reads neither shape fails the turn.
func TestPolicyActionReasoningRefusesAProviderThatCannotCarryIt(t *testing.T) {
	action := policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, "")
	ctx := policyDispatchContext(policyDecisionWithActions(action), action)
	dispatch := &providerDispatch{
		logicalModel: "think", targetFormat: llmprotocol.OpenAIChatV1, useReasoning: true,
		profile: &config.ProviderProfile{Type: "vllm", BaseURL: "http://vllm.internal:8000/v1"},
	}
	if _, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatch, ctx); !errors.Is(err, errPolicyActionFormat) {
		t.Fatalf("error = %v, want errPolicyActionFormat", err)
	}
}

func TestPolicyActionReasoningOnTheMessagesRequest(t *testing.T) {
	budget := int64(2048)
	clientBudget := int64(16000)
	cases := map[string]struct {
		action     config.RaylineARCPolicyBinding
		wantMode   llmprotocol.ReasoningMode
		wantEffort string
		wantBudget *int64
	}{
		"effort":       {policyAction("think", "none", "vendor/think", policyTestEffort("medium"), nil, ""), llmprotocol.ReasoningModeAdaptive, "medium", nil},
		"budget":       {policyAction("think", "none", "vendor/think", nil, &budget, ""), llmprotocol.ReasoningModeEnabled, "", &budget},
		"both":         {policyAction("think", "none", "vendor/think", policyTestEffort("low"), &budget, ""), llmprotocol.ReasoningModeEnabled, "low", &budget},
		"thinking off": {policyAction("off", "none", "vendor/off", policyTestEffort("none"), nil, ""), llmprotocol.ReasoningModeDisabled, "", nil},
		"null effort":  {policyAction("think", "none", "vendor/think", nil, nil, ""), llmprotocol.ReasoningModeEnabled, "", &clientBudget},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := policyDispatchContext(policyDecisionWithActions(test.action), test.action)
			// What the router derived from the client: enabled thinking on the
			// client's budget, at the client's effort.
			request := &llmprotocol.Request{
				ReasoningMode: llmprotocol.ReasoningModeEnabled, ReasoningEffort: "max", ReasoningBudgetTokens: &clientBudget,
			}
			if _, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.AnthropicMessagesV1, ctx); err != nil {
				t.Fatal(err)
			}
			if request.ReasoningMode != test.wantMode || request.ReasoningEffort != test.wantEffort ||
				!sameInt64Pointer(request.ReasoningBudgetTokens, test.wantBudget) {
				t.Fatalf("request reasoning = %s / %q / %v", request.ReasoningMode, request.ReasoningEffort, request.ReasoningBudgetTokens)
			}
		})
	}
}

func TestPolicyActionReasoningLeavesOtherTurnsAlone(t *testing.T) {
	action := policyAction("think", "none", "vendor/think", policyTestEffort("medium"), nil, "")
	request := &llmprotocol.Request{ReasoningEffort: "max"}
	// A decision-only binding leaves dispatch to the modelRef.
	decisionOnly := action
	decisionOnly.Model, decisionOnly.Effort = "", nil
	if changed, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.AnthropicMessagesV1,
		policyDispatchContext(policyDecisionWithActions(decisionOnly), decisionOnly)); changed || err != nil || request.ReasoningEffort != "max" {
		t.Fatalf("decision-only binding changed=%v err=%v effort=%q", changed, err, request.ReasoningEffort)
	}
	// Responses has no reasoning budget, so a budget action cannot travel.
	budget := int64(4096)
	budgeted := policyAction("think", "none", "vendor/think", nil, &budget, "")
	if _, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.OpenAIResponsesV1,
		policyDispatchContext(policyDecisionWithActions(budgeted), budgeted)); !errors.Is(err, errPolicyActionFormat) {
		t.Fatalf("responses budget error = %v", err)
	}
}

// On Responses the action's effort replaces the client's reasoning.effort; a
// null or withheld effort sends none, and the thinking-off action sends the
// off signal whatever the client asked for.
func TestPolicyActionReasoningReachesTheResponsesRequest(t *testing.T) {
	cases := map[string]struct {
		action          config.RaylineARCPolicyBinding
		providerDefault bool
		wantEffort      string
	}{
		"effort":           {policyAction("think", "none", "vendor/think", policyTestEffort("high"), nil, ""), false, "high"},
		"null effort":      {policyAction("think", "none", "vendor/think", nil, nil, ""), false, ""},
		"provider default": {policyAction("think", "none", "vendor/think", policyTestEffort("high"), nil, ""), true, ""},
		"thinking off":     {policyAction("off", "none", "vendor/off", policyTestEffort("none"), nil, ""), false, "none"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			decision := policyDecisionWithActions(test.action)
			if test.providerDefault {
				decision.Algorithm.RaylineARC.PolicyService.DispatchEffort = config.RaylineARCPolicyDispatchEffortProviderDefault
			}
			// What the client sent, and a familyless worker leaves in place.
			request := &llmprotocol.Request{ReasoningEffort: "minimal"}
			if _, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.OpenAIResponsesV1,
				policyDispatchContext(decision, test.action)); err != nil {
				t.Fatal(err)
			}
			if request.ReasoningEffort != test.wantEffort {
				t.Fatalf("effort = %q, want %q", request.ReasoningEffort, test.wantEffort)
			}
		})
	}
}

// With source policy the lever writes the decided action's level. The
// package's neutral level is an empty suffix, which states nothing, so a
// return to it after a steer holds the steer in force -- the replayed item is
// still the last instruction the provider sees -- and the turn records that
// the level it served is not the level the decision asked for.
func TestPolicyLeverAppliesTheDecidedLevel(t *testing.T) {
	up := policyAction(leverWorker, "down", "vendor/glm", policyTestEffort("high"), nil, leverDown)
	neutral := policyAction(leverWorker, "none", "vendor/glm", policyTestEffort("high"), nil, "")
	decision := thinkingLeverDecision(true)
	lever := decision.Algorithm.RaylineARC.ThinkingLever
	lever.Source, lever.Level = config.RaylineARCThinkingSourcePolicy, ""
	binding := lever.Workers[leverWorker]
	binding.NeutralLevel = "none"
	lever.Workers[leverWorker] = binding
	decision.Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{
		Bindings: []config.RaylineARCPolicyBinding{up, neutral},
	}
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	turn := func(action config.RaylineARCPolicyBinding, messages []llmprotocol.Message) ([]llmprotocol.Message, *RequestContext) {
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
		transaction.markSelection(0, 10)
		ctx := policyDispatchContext(decision, action)
		ctx.RaylineARCTransaction = transaction
		request := &llmprotocol.Request{Model: "m", Messages: append([]llmprotocol.Message(nil), messages...)}
		if _, err := (&OpenAIRouter{}).applyRaylineARCThinkingLever(request, ctx); err != nil {
			t.Fatalf("apply: %v", err)
		}
		if err := transaction.commit(context.Background(), ctx); err != nil {
			t.Fatalf("commit: %v", err)
		}
		return request.Messages, ctx
	}

	turn0 := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "fix it")}
	provider0, ctx0 := turn(up, turn0)
	tail := provider0[0].Content[len(provider0[0].Content)-1].Text
	if tail != leverDown || ctx0.RaylineARCThinking.LevelRequested != "down" || ctx0.RaylineARCThinking.Source != "policy" {
		t.Fatalf("turn 0 tail %q trace %+v", tail, ctx0.RaylineARCThinking)
	}

	turn1 := append(append([]llmprotocol.Message(nil), turn0...),
		leverText(llmprotocol.RoleAssistant, "done"), leverText(llmprotocol.RoleUser, "next"))
	provider1, ctx1 := turn(neutral, turn1)
	if trace := ctx1.RaylineARCThinking; trace.LevelRequested != "none" || trace.LevelInForce != "down" ||
		trace.Emitted || trace.Skipped != "neutral_inexpressible" {
		t.Fatalf("turn 1 trace = %+v", trace)
	}
	if got := provider1[0].Content[len(provider1[0].Content)-1].Text; got != leverDown {
		t.Fatalf("turn 1 did not replay turn 0's steer: %q", got)
	}
}

func TestPolicyLeverFailsATurnWithNoDecidedAction(t *testing.T) {
	decision := thinkingLeverDecision(true)
	decision.Algorithm.RaylineARC.ThinkingLever.Source = config.RaylineARCThinkingSourcePolicy
	decision.Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{}
	ctx := &RequestContext{VSRSelectedDecision: decision, RaylineARCDispatch: &raylinearc.WorkerManifest{ID: leverWorker}}
	request := &llmprotocol.Request{Messages: leverMessages()}
	if _, err := (&OpenAIRouter{}).applyRaylineARCThinkingLever(request, ctx); !errors.Is(err, errPolicyLevelMissing) {
		t.Fatalf("error = %v, want errPolicyLevelMissing", err)
	}
}

// An action whose worker's provider cannot carry its reasoning stops the
// selector arming, rather than failing every turn that picks it.
func TestPolicyActionsMustBeCarriableByTheirProvider(t *testing.T) {
	budget := int64(4096)
	routerWith := func(apiFormat string, profile config.ProviderProfile) *config.RouterConfig {
		return &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"think": {PreferredEndpoints: []string{"backend"}, APIFormat: apiFormat}},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "provider", Port: 443, Type: profile.Type,
				ProviderProfileName: "profile",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{"profile": profile},
		}}
	}
	acceptingBoth := func(cfg *config.RouterConfig) *config.RouterConfig {
		params := cfg.ModelConfig["think"]
		params.AcceptedFormats = []string{config.APIFormatAnthropic, config.APIFormatOpenAI}
		cfg.ModelConfig["think"] = params
		return cfg
	}
	openRouter := config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}
	topLevel := config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1", ReasoningTransport: "top_level_effort"}
	chatTemplate := config.ProviderProfile{Type: "vllm", BaseURL: "http://vllm.internal:8000/v1"}
	anthropic := config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}
	effort := policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, "")
	budgeted := policyAction("think", "none", "think-trained", nil, &budget, "")
	off := policyAction("think", "none", "think-trained", policyTestEffort("none"), nil, "")
	withOffFamily := func(cfg *config.RouterConfig) *config.RouterConfig {
		params := cfg.ModelConfig["think"]
		params.ReasoningFamily = "gpt"
		cfg.ModelConfig["think"] = params
		cfg.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
			"gpt": {Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort", Disabled: "none"},
		}
		return cfg
	}
	cases := []struct {
		name    string
		cfg     *config.RouterConfig
		action  config.RaylineARCPolicyBinding
		carries bool
	}{
		{"effort on OpenRouter Chat", routerWith(config.APIFormatOpenAI, openRouter), effort, true},
		{"effort on a top-level transport", routerWith(config.APIFormatOpenAI, topLevel), effort, true},
		{"budget on a top-level transport", routerWith(config.APIFormatOpenAI, topLevel), budgeted, false},
		{"effort on a chat-template transport", routerWith(config.APIFormatOpenAI, chatTemplate), effort, false},
		{"budget on Messages", routerWith(config.APIFormatAnthropic, anthropic), budgeted, true},
		// A worker that accepts Messages and Chat is sent either, per request,
		// so the action must be carriable in both.
		{"budget on Messages and a top-level Chat transport", acceptingBoth(routerWith(config.APIFormatAnthropic, topLevel)), budgeted, false},
		{"effort on Messages and a top-level Chat transport", acceptingBoth(routerWith(config.APIFormatAnthropic, topLevel)), effort, true},
		// Responses carries an effort, but has no reasoning budget.
		{"effort on Responses", routerWith(config.APIFormatResponses, topLevel), effort, true},
		{"budget on Responses", routerWith(config.APIFormatResponses, topLevel), budgeted, false},
		// Thinking-off needs an off signal the worker's family can say.
		{"thinking off on Responses with no reasoning family", routerWith(config.APIFormatResponses, topLevel), off, false},
		{"thinking off on Responses with an off signal", withOffFamily(routerWith(config.APIFormatResponses, topLevel)), off, true},
	}
	for _, test := range cases {
		if got := raylineARCPolicyActionsCarriable(test.cfg, policyDecisionWithActions(test.action)); got != test.carries {
			t.Fatalf("%s: carriable = %v, want %v", test.name, got, test.carries)
		}
	}
}

// Messages counts thinking inside max_tokens and refuses a budget that leaves
// no room below it, so a budget action keeps the caller's allowance on top.
func TestPolicyBudgetActionKeepsRoomBelowMaxTokens(t *testing.T) {
	budget := int64(4096)
	action := policyAction("think", "none", "think-trained", nil, &budget, "")
	ctx := policyDispatchContext(policyDecisionWithActions(action), action)
	for name, test := range map[string]struct {
		clientMax *int64
		want      int64
	}{
		"an allowance at the budget":    {llmprotocol.Int64(4096), 8192},
		"an allowance below the budget": {llmprotocol.Int64(1024), 5120},
		"no stated allowance":           {nil, 8192},
		"an allowance above the budget": {llmprotocol.Int64(32000), 32000},
	} {
		request := &llmprotocol.Request{}
		request.Sampling.MaxOutputTokens = test.clientMax
		if _, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.AnthropicMessagesV1, ctx); err != nil {
			t.Fatal(err)
		}
		if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != test.want {
			t.Fatalf("%s: max_tokens = %v, want %d", name, request.Sampling.MaxOutputTokens, test.want)
		}
	}
}

// Under dispatch_effort: provider_default a withheld effort on Messages sends
// adaptive thinking without it, whatever mode the router derived: a derived
// enabled mode with no budget would be refused by the encoder.
func TestPolicyActionWithheldEffortOnMessagesIsAdaptive(t *testing.T) {
	action := policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, "")
	decision := policyDecisionWithActions(action)
	decision.Algorithm.RaylineARC.PolicyService.DispatchEffort = config.RaylineARCPolicyDispatchEffortProviderDefault
	for name, derived := range map[string]llmprotocol.ReasoningMode{
		"derived enabled":  llmprotocol.ReasoningModeEnabled,
		"derived adaptive": llmprotocol.ReasoningModeAdaptive,
	} {
		t.Run(name, func(t *testing.T) {
			request := &llmprotocol.Request{ReasoningMode: derived, ReasoningEffort: "high"}
			if _, err := applyRaylineARCPolicyActionReasoning(request, llmprotocol.AnthropicMessagesV1, policyDispatchContext(decision, action)); err != nil {
				t.Fatal(err)
			}
			if request.ReasoningMode != llmprotocol.ReasoningModeAdaptive || request.ReasoningEffort != "" || request.ReasoningBudgetTokens != nil {
				t.Fatalf("mode %q effort %q budget %v, want adaptive with no effort or budget",
					request.ReasoningMode, request.ReasoningEffort, request.ReasoningBudgetTokens)
			}
		})
	}
}
