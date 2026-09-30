package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const adaptiveClaudeCodeRequest = `{
 "model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],
 "thinking":{"type":"adaptive","display":"omitted"},
 "output_config":{"effort":"medium"},
 "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}
}`

func TestAnthropicAdaptiveRequestDispatchesToChatAndResponses(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			router := routingTestRouter("chat")
			if target == llmprotocol.OpenAIResponsesV1 {
				model := router.Config.ModelConfig["chat"]
				model.APIFormat = config.APIFormatResponses
				model.ReasoningFamily = "reasoning"
				router.Config.ModelConfig["chat"] = model
				router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
					"reasoning": {Type: config.ReasoningFamilyTypeReasoningEffort, Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}},
				}
			}
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(adaptiveClaudeCodeRequest))
			if err != nil {
				t.Fatal(err)
			}
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
			ctx.ProtocolEnvelope = envelope
			dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
			if err != nil {
				t.Fatalf("dispatch rejected adaptive request: %v", err)
			}
			if dispatch.targetFormat != target {
				t.Fatalf("target = %q, want %q", dispatch.targetFormat, target)
			}
			if err = router.candidateCapabilityMismatch(config.ModelRef{Model: "chat"}, &request, nil, nil, nil); err != nil {
				t.Fatalf("selection disagrees with dispatch: %v", err)
			}
			body, err := router.encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			if _, found := wire["thinking"]; found || len(wire["context_management"]) > 0 {
				t.Fatalf("Anthropic-only controls reached backend: %s", body)
			}
			// Fork policy (CP9m): a Chat turn the client asked to reason carries
			// a reasoning bound derived from its output allowance in place of
			// the effort, since OpenRouter refuses the two together.
			if target == llmprotocol.OpenAIChatV1 && !bytes.Contains(wire["reasoning"], []byte(`"max_tokens"`)) &&
				string(wire["reasoning_effort"]) != `"medium"` {
				t.Fatalf("Chat effort was lost: %s", body)
			}
			if target == llmprotocol.OpenAIResponsesV1 && !bytes.Contains(wire["reasoning"], []byte(`"medium"`)) {
				t.Fatalf("Responses effort was lost: %s", body)
			}
			if request.ReasoningMode != llmprotocol.ReasoningModeAdaptive || request.ReasoningDisplay != "omitted" || len(request.ContextManagement) == 0 {
				t.Fatalf("public request was mutated: %+v", request)
			}
		})
	}
}

func TestAnthropicAdaptiveStrictCandidatesUseProjectedDemand(t *testing.T) {
	router := routingTestRouter("chat")
	router.Config.CandidateRequirements = &config.CandidateRequirements{
		Capabilities: config.CandidateCapabilitiesDeclared,
		Context:      config.CandidateContextKnownLimits,
	}
	model := router.Config.ModelConfig["chat"]
	model.Capabilities = []string{"chat", "reasoning"}
	model.ContextWindowSize = 8192
	model.MaxOutputTokens = 1024
	router.Config.ModelConfig["chat"] = model
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(adaptiveClaudeCodeRequest))
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{Name: "default", ModelRefs: []config.ModelRef{{Model: "chat"}}}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.VSRSelectedDecision = decision
	refs, err := router.decisionEligibleModelRefs(decision, ctx)
	if err != nil || len(refs) != 1 || refs[0].Model != "chat" {
		t.Fatalf("strict prefilter rejected projected request: refs=%+v err=%v", refs, err)
	}
	decision.Action = &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "chat"}
	modelName, found, err := router.decisionRouteActionDestination(decision, ctx)
	if err != nil || !found || modelName != "chat" {
		t.Fatalf("strict route action rejected projected request: model=%q found=%t err=%v", modelName, found, err)
	}
}

func TestAnthropicAdaptiveResponsesDropsCacheDirectiveWithWarning(t *testing.T) {
	router := routingTestRouter("chat")
	model := router.Config.ModelConfig["chat"]
	model.APIFormat = config.APIFormatResponses
	router.Config.ModelConfig["chat"] = model
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
		"model":"chat","max_tokens":64,
		"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}],
		"thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"medium"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = llmprotocol.Envelope{}
	if _, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx); err != nil {
		t.Fatalf("Responses rejected an Anthropic cache hint: %v", err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte("cache_control")) || !bytes.Contains(body, []byte("hello")) {
		t.Fatalf("Responses backend received the cache hint or lost prompt content: %s", body)
	}
	if len(ctx.ProtocolDiagnostics) != 1 ||
		ctx.ProtocolDiagnostics[0].Field != "cache_control" ||
		ctx.ProtocolDiagnostics[0].Action != llmprotocol.DiagnosticDropped ||
		ctx.ProtocolDiagnostics[0].Target != llmprotocol.OpenAIResponsesV1 {
		t.Fatalf("Responses cache drop was not reported once: %+v", ctx.ProtocolDiagnostics)
	}
	if !llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityCacheDirectives) {
		t.Fatal("dispatch modified the public Anthropic request")
	}
}

// Accept by default (decision 36, US-003g/h): meaningful edits and a display
// the backend cannot produce are dropped and counted, not refused.
func TestAnthropicCrossFormatProjectionDropsMeaningfulEditsAndDisplay(t *testing.T) {
	router := routingTestRouter("chat")
	engine := protocolcodec.NewBuiltinEngine()
	for _, test := range []struct {
		name, body, field string
	}{
		{"clear_last", `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"last"}]}}`, "context_management"},
		{"unknown_edit_option", `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all","new_option":true}]}}`, "context_management"},
		{"summarized", `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","display":"summarized"}}`, "reasoning_display"},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			projected, diagnostics, err := router.projectAnthropicRequestForBackendWithDiagnostics(request, "chat", llmprotocol.OpenAIChatV1)
			if err != nil {
				t.Fatalf("projection refused the turn: %v", err)
			}
			if len(projected.ContextManagement) > 0 || projected.ReasoningDisplay != "" {
				t.Fatalf("an unhonoured control survived projection: %+v", projected)
			}
			found := false
			for _, diagnostic := range diagnostics {
				found = found || (diagnostic.Field == test.field && diagnostic.Action == llmprotocol.DiagnosticDropped)
			}
			if !found {
				t.Fatalf("no %s drop diagnostic: %+v", test.field, diagnostics)
			}
		})
	}
}

// US-003e: a model with no configured family still gets the portable
// off-signal instead of a refusal; a configured family gets its own control.
func TestAnthropicDisabledUsesConfiguredOffControl(t *testing.T) {
	router := routingTestRouter("chat")
	engine := protocolcodec.NewBuiltinEngine()
	body := []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	unconfigured, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatalf("unconfigured model refused disabled thinking: %v", err)
	}
	unconfiguredBody, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if unconfiguredBody, err = router.adaptProviderRequest(unconfiguredBody, unconfigured, ctx); err != nil {
		t.Fatalf("unconfigured model refused disabled thinking at the adapter: %v", err)
	}
	if !bytes.Contains(unconfiguredBody, []byte(`"reasoning_effort":"none"`)) {
		t.Fatalf("unconfigured model received no off-signal: %s", unconfiguredBody)
	}
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"qwen3": {
			Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
			Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
		},
	}
	model := router.Config.ModelConfig["chat"]
	model.ReasoningFamily = "qwen3"
	router.Config.ModelConfig["chat"] = model
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = "vllm"
	router.Config.ProviderProfiles["provider"] = profile
	dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatalf("configured reasoning-off model rejected: %v", err)
	}
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	adapted, err := router.adaptProviderRequest(encoded, dispatch, ctx)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		ChatTemplateKwargs map[string]bool `json:"chat_template_kwargs"`
	}
	if err := json.Unmarshal(adapted, &wire); err != nil {
		t.Fatal(err)
	}
	if enabled, found := wire.ChatTemplateKwargs["enable_thinking"]; !found || enabled {
		t.Fatalf("backend did not receive an explicit off control: %s", adapted)
	}
}

// US-003e: a family without a working off control is approximated, not
// refused: the request keeps its disabled mode for the codec's off-signal.
func TestAnthropicDisabledApproximatesFamiliesWithoutOffControl(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		family config.ReasoningFamilyConfig
	}{
		{"always_on", config.ReasoningFamilyConfig{Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking", Modes: []string{config.ReasoningModeEnabled}}},
		{"effort_without_off_value", config.ReasoningFamilyConfig{Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort", Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := routingTestRouter("chat")
			router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"family": test.family}
			model := router.Config.ModelConfig["chat"]
			model.ReasoningFamily = "family"
			router.Config.ModelConfig["chat"] = model
			projected, diagnostics, projectionErr := router.projectAnthropicRequestForBackendWithDiagnostics(request, "chat", llmprotocol.OpenAIChatV1)
			if projectionErr != nil {
				t.Fatalf("unsupported off control refused the turn: %v", projectionErr)
			}
			if projected.ReasoningMode != llmprotocol.ReasoningModeDisabled {
				t.Fatalf("the off request was lost: %q", projected.ReasoningMode)
			}
			if len(diagnostics) == 0 || diagnostics[0].Field != "thinking.type" || diagnostics[0].Action != llmprotocol.DiagnosticApproximated {
				t.Fatalf("no approximation diagnostic: %+v", diagnostics)
			}
		})
	}
}

// US-003e: a provider whose transport cannot express the family's off
// control still receives an explicit off-signal instead of a refusal.
func TestAnthropicDisabledStatesOffOnProviderWithoutEffectiveOffControl(t *testing.T) {
	router := routingTestRouter("chat")
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"family": {
			Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
			ActivationParameter: "enable_thinking", Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
		},
	}
	model := router.Config.ModelConfig["chat"]
	model.ReasoningFamily = "family"
	router.Config.ModelConfig["chat"] = model
	profile := router.Config.ProviderProfiles["provider"]
	profile.ReasoningTransport = modelcatalog.ReasoningTransportTopLevelEffort
	router.Config.ProviderProfiles["provider"] = profile
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatal(err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	adapted, err := router.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		t.Fatalf("provider refused disabled thinking: %v", err)
	}
	if !bytes.Contains(adapted, []byte(`"reasoning_effort":"none"`)) {
		t.Fatalf("provider received no off-signal: %s", adapted)
	}
}

func TestAnthropicOmittedDisplayFiltersBufferedChatReasoning(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	body := []byte(`{"id":"response_1","model":"chat","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"private thinking","content":"public answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`)
	response, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatal(err)
	}
	public, err := router.encodeClientResponse(*response, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte(`"thinking"`)) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("omitted display leaked reasoning or lost answer: %s", public)
	}
}

func TestAnthropicOmittedDisplayFiltersBufferedResponsesReasoning(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	body := []byte(`{
		"id":"resp_1","model":"chat","status":"completed",
		"output":[
			{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"private summary"}],"content":[{"type":"reasoning_text","text":"private thinking"}]},
			{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"public answer"}]}
		],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
	}`)
	response, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatal(err)
	}
	public, err := router.encodeClientResponse(*response, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte("private summary")) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("omitted display leaked reasoning or lost answer: %s", public)
	}
}

func TestAnthropicOmittedDisplayFiltersStreamingChatReasoning(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	engine := protocolcodec.NewBuiltinEngine()
	stream, err := engine.NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: t.Context()}, clientStreamMutation(ctx, llmprotocol.OpenAIChatV1))
	if err != nil {
		t.Fatal(err)
	}
	chunks := []string{
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"reasoning_content":"private thinking"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"content":"public answer"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	var public []byte
	for _, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		for _, frame := range frames {
			public = append(public, frame...)
		}
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		public = append(public, frame...)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte("thinking_delta")) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("stream leaked reasoning or lost answer: %s", public)
	}
}
