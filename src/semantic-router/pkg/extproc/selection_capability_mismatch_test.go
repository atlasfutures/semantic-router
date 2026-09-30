package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Explicitly disabling reasoning needs a configured backend off control;
// unannotated Chat candidates cannot promise that semantic behavior.
func unsupportedThinkingRequest(model string) *llmprotocol.Request {
	request := testNeutralRequest(model, "Create hello.txt containing hi.")
	request.ReasoningMode = llmprotocol.ReasoningModeDisabled
	request.Trusted.SourceFormat = llmprotocol.AnthropicMessagesV1
	return request
}

// Accept by default (decision 36, US-003e): an explicit thinking-off is
// honoured with the portable off-signal, so an unannotated Chat candidate is
// eligible whether it is auto-routed or named.
func TestAutoRoutingAcceptsDisabledThinkingLikeNamedModel(t *testing.T) {
	router, chatModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := &config.Decision{
		Name: "default-route", Priority: 1, ModelRefs: []config.ModelRef{{Model: chatModel}},
	}
	router.Config.Decisions = []config.Decision{*decision}

	auto := unsupportedThinkingRequest("auto")
	if err := router.candidateCapabilityMismatch(config.ModelRef{Model: chatModel}, auto, decision, nil, nil); err != nil {
		t.Fatalf("auto-routed candidate excluded for disabled thinking: %v", err)
	}
	named := unsupportedThinkingRequest(chatModel)
	if _, err := router.prepareProviderDispatch(named, chatModel, "", false, routingTestContext(llmprotocol.AnthropicMessagesV1, named)); err != nil {
		t.Fatalf("named model refused disabled thinking: %v", err)
	}
}

func TestAutoRoutingKeepsMixedContextAndWireExclusionsUnavailable(t *testing.T) {
	router, chatModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	chatParams := router.Config.ModelConfig[chatModel]
	chatParams.ContextWindowSize = 32_768
	messagesParams := chatParams
	// The Chat candidate's wire exclusion is a declared text-only card facing
	// an image turn. (Disabled thinking used to be the exclusion; it is now
	// honoured with an off-signal under accept-by-default, US-003e.)
	chatParams.Capabilities = []string{"chat"}
	router.Config.ModelConfig[chatModel] = chatParams
	messagesModel := "messages-small-context"
	messagesParams.APIFormat = config.APIFormatAnthropic
	messagesParams.ContextWindowSize = 1
	router.Config.ModelConfig[messagesModel] = messagesParams
	router.Config.Decisions = []config.Decision{{
		Name: "default-route", Priority: 1,
		ModelRefs: []config.ModelRef{{Model: messagesModel}, {Model: chatModel}},
	}}
	classifier, err := classification.NewClassifier(router.Config, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier

	request := unsupportedThinkingRequest("auto")
	request.Messages[0].Content = append(request.Messages[0].Content,
		llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.RequestModel = "auto"
	_, response := router.runRequestPreRoutingStages("auto", extractSemanticRequestSignals(request), ctx)
	if ctx.VSRContextTokenCount <= messagesParams.ContextWindowSize {
		t.Fatalf("context estimate %d did not exercise the context-window exclusion", ctx.VSRContextTokenCount)
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != 503 {
		t.Fatalf("mixed context and wire exclusions returned %d, body %s; want fail-closed 503", got, response.GetImmediateResponse().GetBody())
	}
	if ctx.ImmediateProtocolError != nil {
		t.Fatalf("mixed exclusions were attributed solely to the caller: %v", ctx.ImmediateProtocolError)
	}
}
