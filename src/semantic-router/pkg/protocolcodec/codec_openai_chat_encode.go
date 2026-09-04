package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (OpenAIChatCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	if request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens < 0 {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_chat_max_output_tokens",
			"Chat Completions cannot represent a negative output token limit",
			nil,
		)
	}
	if len(request.Sampling.Stop) > 4 {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_chat_stop_sequence_limit",
			"Chat Completions cannot represent more than four stop sequences",
			nil,
		)
	}
	diagnostics, validationErr := chatRequestDiagnostics(request, policy)
	if validationErr != nil {
		return nil, diagnostics, validationErr
	}
	wire := encodeChatBaseRequest(request)
	encodeChatReasoningControls(&wire, request, &diagnostics, policy)
	if encodeErr := appendChatMessages(&wire, request); encodeErr != nil {
		return nil, diagnostics, encodeErr
	}
	if len(wire.Messages) == 0 {
		return nil, diagnostics, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"chat_messages_required",
			"Chat Completions requires at least one conversation message",
			nil,
		)
	}
	appendChatTools(&wire, request.Tools)
	if encodeErr := encodeChatRequestOptions(&wire, request); encodeErr != nil {
		return nil, diagnostics, encodeErr
	}
	body, encodeErr := marshalWire(wire)
	if encodeErr != nil {
		return nil, diagnostics, encodeErr
	}
	body, encodeErr = mergeUnmodeledFields(body, request, llmprotocol.OpenAIChatV1, &diagnostics, policy)
	return body, diagnostics, encodeErr
}

func chatRequestDiagnostics(request llmprotocol.Request, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	for _, message := range request.Messages {
		appendCarriedBlockDrops(&diagnostics, message.Content, llmprotocol.OpenAIChatV1, policy)
	}
	if request.PreviousResponseID == "" && request.ConversationID == "" && request.Truncation == "" {
		return diagnostics, nil
	}
	err := appendLossy(
		&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIChatV1,
		"conversation_state", "Chat Completions has no stateful response reference",
	)
	return diagnostics, err
}

func encodeChatBaseRequest(request llmprotocol.Request) chatRequestWire {
	wire := chatRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		Store: request.Store, User: request.EndUserID,
		ParallelToolCalls: request.ParallelToolCalls, CandidateCount: request.CandidateCount,
		Temperature: request.Sampling.Temperature, TopP: request.Sampling.TopP,
		MaxCompletionTokens: request.Sampling.MaxOutputTokens, Seed: request.Sampling.Seed,
		FrequencyPenalty: request.Sampling.FrequencyPenalty, PresencePenalty: request.Sampling.PresencePenalty,
		ReasoningEffort: request.ReasoningEffort, ReasoningBudget: request.ReasoningBudgetTokens,
	}
	if request.Stream && (request.StreamOptions.IncludeUsage != nil || request.StreamOptions.IncludeObfuscation != nil) {
		wire.StreamOptions = &chatStreamOptionsWire{
			IncludeUsage:       request.StreamOptions.IncludeUsage,
			IncludeObfuscation: request.StreamOptions.IncludeObfuscation,
		}
	}
	return wire
}

// encodeChatReasoningControls carries the thinking controls the Chat wire can
// hold and records the one it cannot.
//
//   - adaptive: the model decides whether and how much to think, which is what
//     Chat Completions does when the request names no reasoning control. It is
//     represented by carrying nothing, so the effort level the client sent
//     still reaches the provider.
//   - disabled: reasoning_effort "none" is the explicit off-signal. It replaces
//     any effort the client sent, because the two collapse into one Chat knob
//     and thinking-off is the stronger statement.
//   - display: Chat Completions has no control over whether reasoning comes
//     back or how it is summarized. The turn still runs and the drop is
//     recorded, because losing a presentation preference is the routable
//     outcome and refusing the conversation is not.
func encodeChatReasoningControls(
	wire *chatRequestWire,
	request llmprotocol.Request,
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
) {
	if request.ReasoningMode == llmprotocol.ReasoningModeDisabled {
		wire.ReasoningEffort = "none"
	}
	budget, derived := chatReasoningBound(request)
	if budget != nil {
		wire.Reasoning = &chatReasoningWire{MaxTokens: budget}
	}
	if derived {
		// OpenRouter's reasoning parameter takes an effort level or a token
		// bound, "One of the following (not both)", and documents no
		// precedence for a request that sends both. A derived bound is the
		// Router's own number, put there to end a turn that would otherwise
		// run to the platform deadline, and the measurement says an effort
		// level beside it makes it inert. So the derived bound travels alone.
		// Nothing is lost by it: "For models that only support
		// reasoning.effort ... the max_tokens value will be used to determine
		// the effort level." Read 2026-09-04:
		//   https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
		//
		// A budget the client stated is a different case. Both fields are then
		// the client's own and the Router is only carrying them.
		wire.ReasoningEffort = ""
	}
	if request.ReasoningDisplay != "" {
		appendPresentationDrop(
			diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIChatV1,
			"reasoning_display", "Chat Completions cannot control how reasoning is returned",
		)
	}
}

// minimumReasoningBudget is the smallest reasoning budget any documented
// reasoning API accepts. Anthropic states it for thinking.budget_tokens, and a
// bound below it would be refused rather than obeyed.
const minimumReasoningBudget = 1024

// chatReasoningBound returns how many tokens the turn may spend on reasoning,
// or nil to send no bound at all, and whether the Router derived that number
// rather than reading it off the request.
//
// max_completion_tokens does not bound reasoning: the models behind the
// thinking arms do not count reasoning against it, so a turn that asks to
// reason and states no budget runs until it finishes or until the platform
// cuts the connection. OpenRouter's reasoning.max_tokens is the bound that
// does apply.
//
// The rule:
//
//   - A stated thinking budget is the client's own number and travels
//     verbatim, whatever the output allowance is.
//   - A turn that asks to reason without stating a budget -- adaptive
//     thinking, or an effort level on its own -- is bounded by what the client
//     allowed for output, floored at the smallest budget a provider accepts.
//     Reasoning plus output then cannot exceed twice the output allowance, and
//     the output allowance itself is never reduced: the client gets everything
//     it asked to be shown.
//   - A turn that asks for no reasoning gets no bound. Sending one to a
//     thinking-off arm would ask it to start reasoning, and thinking-off says
//     so even when an effort level travels beside it: Claude Code sends
//     thinking disabled with output_config.effort high, and the off-signal is
//     the stronger statement of the two.
//   - A request with no output allowance gets no bound either. There is
//     nothing to derive one from, and capping a client that asked for no cap
//     is not the Router's to do.
func chatReasoningBound(request llmprotocol.Request) (bound *int64, derived bool) {
	if request.ReasoningMode == llmprotocol.ReasoningModeDisabled {
		return nil, false
	}
	if request.ReasoningBudgetTokens != nil {
		return request.ReasoningBudgetTokens, false
	}
	reasoningAsked := request.ReasoningMode == llmprotocol.ReasoningModeAdaptive ||
		request.ReasoningMode == llmprotocol.ReasoningModeEnabled ||
		request.ReasoningEffort != "" && request.ReasoningEffort != "none"
	if !reasoningAsked || request.Sampling.MaxOutputTokens == nil {
		return nil, false
	}
	allowance := *request.Sampling.MaxOutputTokens
	if allowance < minimumReasoningBudget {
		allowance = minimumReasoningBudget
	}
	return &allowance, true
}

func appendChatMessages(wire *chatRequestWire, request llmprotocol.Request) error {
	for _, instruction := range request.Instructions {
		if messageDropsWhole(instruction.Content, llmprotocol.OpenAIChatV1) {
			continue
		}
		encoded, err := encodeChatMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content})
		if err != nil {
			return err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	for _, message := range request.Messages {
		if messageDropsWhole(message.Content, llmprotocol.OpenAIChatV1) {
			continue
		}
		encoded, err := encodeChatMessage(message)
		if err != nil {
			return err
		}
		wire.Messages = append(wire.Messages, encoded)
	}
	return nil
}

func appendChatTools(wire *chatRequestWire, tools []llmprotocol.Tool) {
	for _, tool := range tools {
		wire.Tools = append(wire.Tools, chatToolWire{Type: "function", Function: chatFunctionDefinitionWire{
			Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict,
		}, CacheControl: encodeAnthropicCacheControl(tool.Cache)})
	}
}

func encodeChatRequestOptions(wire *chatRequestWire, request llmprotocol.Request) error {
	wire.ToolChoice = encodeChatToolChoice(request.ToolChoice)
	if len(request.Sampling.Stop) == 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop[0])
	} else if len(request.Sampling.Stop) > 1 {
		wire.Stop, _ = json.Marshal(request.Sampling.Stop)
	}
	output, err := encodeChatOutputFormat(request.OutputFormat)
	if err != nil {
		return err
	}
	wire.ResponseFormat = output
	return nil
}

func encodeChatMessage(message llmprotocol.Message) (chatMessageWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return chatMessageWire{}, err
	}
	wire := chatMessageWire{ID: message.ID, Role: role}
	state := chatMessageEncodingState{wire: &wire, parts: make([]chatContentWire, 0, len(message.Content))}
	for _, content := range message.Content {
		if err := state.appendContent(content); err != nil {
			return chatMessageWire{}, err
		}
	}
	if state.citationTextBlocks > 1 {
		return chatMessageWire{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "citation_text_ambiguous", "Chat Completions citations require one text block", nil)
	}
	if len(state.parts) == 1 && state.parts[0].Type == "text" && state.parts[0].CacheControl == nil {
		wire.Content, _ = json.Marshal(state.parts[0].Text)
	} else if len(state.parts) > 0 {
		wire.Content, _ = json.Marshal(state.parts)
	}
	return wire, nil
}

type chatMessageEncodingState struct {
	wire               *chatMessageWire
	parts              []chatContentWire
	citationTextBlocks int
}

func (state *chatMessageEncodingState) appendContent(content llmprotocol.Content) error {
	switch content.Kind {
	case llmprotocol.ContentText:
		state.appendText(content)
	case llmprotocol.ContentRefusal:
		return appendChatTextField(&state.wire.Refusal, content)
	case llmprotocol.ContentReasoning:
		return appendChatTextField(&state.wire.Reasoning, content)
	case llmprotocol.ContentImage:
		return state.appendImage(content)
	case llmprotocol.ContentAudio:
		state.parts = append(state.parts, chatContentWire{Type: "input_audio", InputAudio: &chatInputAudioWire{Data: content.Data, Format: content.MediaType}, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	case llmprotocol.ContentFile:
		return state.appendFile(content)
	case llmprotocol.ContentToolCall:
		return state.appendCachelessToolCall(content)
	case llmprotocol.ContentToolResult:
		return state.appendCachelessToolResult(content)
	case llmprotocol.ContentUnmodeled:
		// A carried block belongs to the contract it came from. Chat
		// Completions never names one, so the block is dropped here and the
		// drop is recorded in chatRequestDiagnostics.
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as chat", nil)
	}
	return nil
}

func appendChatTextField(target **string, content llmprotocol.Content) error {
	if content.Cache != nil {
		return unsupportedChatCacheDirective(string(content.Kind))
	}
	if *target == nil {
		*target = new(string)
	}
	**target += content.Text
	return nil
}

func (state *chatMessageEncodingState) appendCachelessToolCall(content llmprotocol.Content) error {
	if content.Cache != nil {
		return unsupportedChatCacheDirective(string(content.Kind))
	}
	return state.appendToolCall(content.ToolCall)
}

func (state *chatMessageEncodingState) appendCachelessToolResult(content llmprotocol.Content) error {
	if content.Cache != nil {
		return unsupportedChatCacheDirective(string(content.Kind))
	}
	return state.appendToolResult(content.ToolResult)
}

func (state *chatMessageEncodingState) appendText(content llmprotocol.Content) {
	state.parts = append(state.parts, chatContentWire{Type: "text", Text: content.Text, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	if len(content.Citations) > 0 {
		state.citationTextBlocks++
		state.wire.Annotations = append(state.wire.Annotations, encodeChatAnnotations(content.Citations)...)
	}
}

func (state *chatMessageEncodingState) appendImage(content llmprotocol.Content) error {
	if content.FileID != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "image_file_id", "Chat Completions cannot encode image file IDs", nil)
	}
	imageURL := content.URL
	if content.Data != "" {
		if !strings.HasPrefix(content.MediaType, "image/") {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_media_type", "inline Chat images require an image media type", nil)
		}
		imageURL = "data:" + content.MediaType + ";base64," + content.Data
	}
	if imageURL == "" {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_source", "Chat images require a URL or inline data", nil)
	}
	state.parts = append(state.parts, chatContentWire{Type: "image_url", ImageURL: &chatImageURLWire{URL: imageURL, Detail: content.Detail}, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	return nil
}

func (state *chatMessageEncodingState) appendFile(content llmprotocol.Content) error {
	if content.Detail != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_detail", "Chat Completions cannot encode file detail", nil)
	}
	if content.URL != "" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_url", "Chat Completions cannot encode file URLs", nil)
	}
	file := &chatFileWire{Filename: content.Filename, FileID: content.FileID}
	if content.Data != "" {
		if content.MediaType == "" {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "file_media_type", "inline Chat files require a media type", nil)
		}
		file.FileData = "data:" + content.MediaType + ";base64," + content.Data
	}
	state.parts = append(state.parts, chatContentWire{Type: "file", File: file, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	return nil
}

func (state *chatMessageEncodingState) appendToolCall(call *llmprotocol.ToolCall) error {
	if call == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call content is invalid", nil)
	}
	state.wire.ToolCalls = append(state.wire.ToolCalls, chatToolCallWire{
		ID: call.ID, Type: "function",
		Function: chatFunctionCallWire{Name: call.Name, Arguments: call.Arguments},
	})
	return nil
}

func (state *chatMessageEncodingState) appendToolResult(result *llmprotocol.ToolResult) error {
	if result == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result content is invalid", nil)
	}
	state.wire.ToolCallID = result.CallID
	for _, resultContent := range result.Content {
		if resultContent.Kind == llmprotocol.ContentUnmodeled {
			// The block belongs to the contract it came from. Dropping it keeps
			// the tool result routable; refusing it would lose the whole turn.
			continue
		}
		if resultContent.Kind != llmprotocol.ContentText {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "tool_result_media", "chat tool results support text only", nil)
		}
		state.parts = append(state.parts, chatContentWire{
			Type: "text", Text: resultContent.Text,
			CacheControl: encodeAnthropicCacheControl(resultContent.Cache),
		})
	}
	return nil
}

func encodeChatToolChoice(choice llmprotocol.ToolChoice) json.RawMessage {
	switch choice.Mode {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
		body, _ := json.Marshal(choice.Mode)
		return body
	case llmprotocol.ToolChoiceNamed:
		body, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]string{"name": choice.Name}})
		return body
	default:
		return nil
	}
}

func encodeChatOutputFormat(format llmprotocol.OutputFormat) (*chatOutputWire, error) {
	switch format.Kind {
	case "", llmprotocol.OutputText:
		return nil, nil
	case llmprotocol.OutputJSONObject:
		return &chatOutputWire{Type: "json_object"}, nil
	case llmprotocol.OutputJSONSchema:
		schema := map[string]any{"name": format.Name, "schema": format.Schema}
		if format.Description != "" {
			schema["description"] = format.Description
		}
		if format.Strict != nil {
			schema["strict"] = format.Strict
		}
		body, err := json.Marshal(schema)
		return &chatOutputWire{Type: "json_schema", JSONObject: body}, err
	default:
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_format", "output format cannot be encoded as chat", nil)
	}
}
