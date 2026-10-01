package protocolcodec

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (OpenAIResponsesCodec) EncodeRequest(request llmprotocol.Request, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if !holdsWhatEveryTargetDrops(request) && envelope.CanReplay(llmprotocol.OpenAIResponsesV1, request.Generation, policy, false) {
		return append([]byte(nil), envelope.Request...), nil, nil
	}
	// The engine projects an Anthropic client's breakpoints away before this
	// point (ProjectAnthropicCacheDirectives), so this guards only a direct
	// codec call.
	if llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityCacheDirectives) {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_cache_directive",
			"Responses cannot encode per-block cache directives without an explicit projection",
			nil,
		)
	}
	if err := validateResponsesEncodableRequest(request); err != nil {
		return nil, nil, err
	}
	var diagnostics llmprotocol.Diagnostics
	for _, message := range request.Messages {
		if message.ReasoningEffort != "" {
			appendProviderFieldOmission(&diagnostics, policy, request.Trusted.SourceFormat,
				"messages[].output_config.effort", "Responses cannot apply Anthropic per-message effort")
		}
	}
	if len(request.ContextManagement) > 0 {
		if err := appendLossy(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIResponsesV1,
			"context_management", "Responses cannot apply Anthropic context edits"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire, err := encodeResponsesRequestWire(request)
	if err != nil {
		return nil, diagnostics, err
	}
	// A summary alone would put back a reasoning object the router took away
	// for this arm, turning reasoning on at the provider's default effort.
	// See clientStatedReasoningEffort.
	if wire.Reasoning != nil && wire.Reasoning.Effort == "" && clientStatedReasoningEffort(envelope) {
		wire.Reasoning = nil
		appendPresentationDrop(&diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIResponsesV1,
			"reasoning.summary", "the request sends no reasoning object to carry it in")
	}
	body, err := marshalWire(wire)
	if err != nil {
		return nil, diagnostics, err
	}
	// See chatRequestDiagnostics: the table states what Responses cannot
	// express, and the two carriers hold what no contract names.
	appendRequestDispositions(&diagnostics, request, llmprotocol.OpenAIResponsesV1, policy)
	appendUnchoosableToolChoiceDrop(&diagnostics, policy, request, llmprotocol.OpenAIResponsesV1, len(wire.Tools))
	appendToolExtensionDrops(&diagnostics, request.Tools, llmprotocol.OpenAIResponsesV1, policy)
	appendCarriedToolDrops(&diagnostics, request, llmprotocol.OpenAIResponsesV1, policy)
	for _, message := range request.Messages {
		appendContentExtensionDrops(&diagnostics, message.Content, llmprotocol.OpenAIResponsesV1, policy)
		appendCarriedBlockDrops(&diagnostics, message.Content, llmprotocol.OpenAIResponsesV1, policy)
		if dropsEncryptedReasoning(request, message) {
			appendUnmodeledDrop(&diagnostics, policy, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, "content.reasoning")
		}
	}
	if wire.Reasoning == nil && clientStatedReasoningEffort(envelope) {
		request.Unmodeled = withoutCarriedReasoning(request.Unmodeled, &diagnostics, policy)
	}
	body, err = mergeUnmodeledFields(body, request, llmprotocol.OpenAIResponsesV1, &diagnostics, policy)
	if err != nil {
		return body, diagnostics, err
	}
	return replayEquivalentResponsesSource(body, request, envelope, policy), diagnostics, nil
}

// holdsWhatEveryTargetDrops reports whether the request carries something no
// target is sent -- a carried tool, or a resent encrypted reasoning item this
// dispatch does not forward -- so the client bytes, which still hold it, are
// never replayed.
func holdsWhatEveryTargetDrops(request llmprotocol.Request) bool {
	if len(request.CarriedTools) > 0 {
		return true
	}
	for _, message := range request.Messages {
		if dropsEncryptedReasoning(request, message) {
			return true
		}
	}
	return false
}

// clientStatedReasoningEffort reports whether the client's own reasoning
// object named an effort. When it did and the encoded request sends none, the
// router took reasoning away for this arm, and a carried summary must not put
// it back. When it did not -- reasoning: {summary: "auto"} alone -- the client
// asked for the provider's default effort with a summary, and the carried
// summary rebuilds exactly that. Without the client bytes, the answer is yes,
// which drops the summary rather than risk turning reasoning on.
func clientStatedReasoningEffort(envelope llmprotocol.Envelope) bool {
	if len(envelope.Request) == 0 {
		return true
	}
	var client struct {
		Reasoning *struct {
			Effort json.RawMessage `json:"effort"`
		} `json:"reasoning"`
	}
	if json.Unmarshal(envelope.Request, &client) != nil || client.Reasoning == nil {
		return true
	}
	return hasJSONValue(client.Reasoning.Effort)
}

// withoutCarriedReasoning removes a carried reasoning member, such as
// reasoning.summary, when the encoded request sends no reasoning object and
// the router took the client's reasoning away. The merge would otherwise
// create one, and a reasoning object on an arm dispatched without reasoning
// turns it back on at the provider's default effort.
func withoutCarriedReasoning(
	carrier *llmprotocol.UnmodeledFields,
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
) *llmprotocol.UnmodeledFields {
	if carrier == nil || carrier.Children["reasoning"].Empty() {
		return carrier
	}
	for _, path := range unnamedMemberPaths(carrier.Children["reasoning"], "reasoning.") {
		appendPresentationDrop(
			diagnostics, policy, carrier.Format, llmprotocol.OpenAIResponsesV1, path,
			"the request sends no reasoning object to carry it in",
		)
	}
	trimmed := *carrier
	trimmed.Children = make(map[string]*llmprotocol.UnmodeledFields, len(carrier.Children))
	for name, child := range carrier.Children {
		if name != "reasoning" {
			trimmed.Children[name] = child
		}
	}
	return &trimmed
}

func validateResponsesEncodableRequest(request llmprotocol.Request) error {
	if err := rejectChatOnlyControls(request); err != nil {
		return err
	}
	if request.ReasoningDisplay != "" {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_reasoning_display",
			"Responses cannot represent reasoning display controls",
			nil,
		)
	}
	if request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens < 16 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_responses_max_output_tokens",
			"Responses requires an output token limit of at least 16",
			nil,
		)
	}
	return nil
}

func encodeResponsesRequestWire(request llmprotocol.Request) (responsesRequestWire, error) {
	wire := responsesRequestWire{
		Model: request.Model, Stream: request.Stream, Metadata: request.Metadata,
		Store: request.Store, AutoStore: request.AutoStore, PreviousResponseID: request.PreviousResponseID,
		Truncation: request.Truncation, User: request.EndUserID, PromptCacheKey: request.PromptCacheKey,
		ParallelToolCalls: request.ParallelToolCalls, Temperature: request.Sampling.Temperature,
		TopP: request.Sampling.TopP, MaxOutputTokens: request.Sampling.MaxOutputTokens,
	}
	if request.Stream && request.StreamOptions.IncludeObfuscation != nil {
		wire.StreamOptions = &responsesStreamOptionsWire{IncludeObfuscation: request.StreamOptions.IncludeObfuscation}
	}
	if request.ConversationID != "" {
		wire.Conversation, _ = json.Marshal(request.ConversationID)
	}
	if request.ReasoningEffort != "" || request.ReasoningSummary != "" {
		wire.Reasoning = &responsesReasoningWire{Effort: request.ReasoningEffort}
		if request.ReasoningSummary != "" {
			wire.Reasoning.Summary, _ = json.Marshal(request.ReasoningSummary)
		}
	}
	items, err := encodeResponsesRequestItems(request)
	if err != nil {
		return responsesRequestWire{}, err
	}
	wire.Input, _ = json.Marshal(items)
	wire.Tools = encodeResponsesTools(request.Tools, request.ImageGeneration, forwardedCarriedTools(request, llmprotocol.OpenAIResponsesV1))
	// Gated on the tools this target encoded; see encodeChatRequestOptions for
	// why a server-tool-only turn otherwise states a choice with no tools.
	if len(wire.Tools) > 0 {
		wire.ToolChoice = encodeResponsesToolChoice(request.ToolChoice)
	}
	wire.Text = encodeResponsesOutputFormat(request.OutputFormat, request.TextVerbosity)
	return wire, nil
}

func encodeResponsesRequestItems(request llmprotocol.Request) ([]json.RawMessage, error) {
	items := make([]json.RawMessage, 0, len(request.Messages))
	callKinds := make(map[string]llmprotocol.ToolKind)
	appendMessage := func(message llmprotocol.Message) error {
		if dropsEncryptedReasoning(request, message) {
			return nil
		}
		if carried, isCarried := carriedItemBytes(message, llmprotocol.OpenAIResponsesV1); isCarried {
			items = append(items, carried)
			return nil
		}
		if messageDropsWhole(message.Content, llmprotocol.OpenAIResponsesV1) {
			return nil
		}
		encoded, err := encodeResponsesMessage(message, "input")
		if err != nil {
			return err
		}
		for _, item := range encoded {
			body, marshalErr := json.Marshal(item)
			if marshalErr != nil {
				return llmprotocol.NewError(llmprotocol.ErrorInternal, "encode_wire", "wire request could not be encoded", marshalErr)
			}
			items = append(items, body)
		}
		return nil
	}
	for _, instruction := range request.Instructions {
		content := instructionContentFor(instruction.Content, llmprotocol.OpenAIResponsesV1)
		if err := appendMessage(llmprotocol.Message{Role: instruction.Role, Content: content}); err != nil {
			return nil, err
		}
	}
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				callKinds[content.ToolCall.ID] = content.ToolCall.Kind
			}
		}
		if len(message.Content) == 0 && message.ReasoningEffort != "" {
			// Omitted and counted in EncodeRequest.
			continue
		}
		// Chat tool messages do not carry a result kind. A preceding custom
		// call identifies the correct Responses output variant.
		message.Content = append([]llmprotocol.Content(nil), message.Content...)
		for index := range message.Content {
			content := &message.Content[index]
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil && content.ToolResult.Kind == "" {
				if kind := callKinds[content.ToolResult.CallID]; kind == llmprotocol.ToolKindCustom {
					copy := *content.ToolResult
					copy.Kind = kind
					content.ToolResult = &copy
				}
			}
		}
		if err := appendMessage(message); err != nil {
			return nil, err
		}
	}
	return items, nil
}

func encodeResponsesTools(
	input []llmprotocol.Tool,
	imageGeneration *llmprotocol.ImageGenerationOptions,
	carried []llmprotocol.UnmodeledBlock,
) json.RawMessage {
	if len(input) == 0 && imageGeneration == nil && len(carried) == 0 {
		return nil
	}
	tools := make([]responsesToolWire, 0, len(input)+1)
	namespaces := map[string]int{}
	for _, tool := range input {
		if tool.Kind == llmprotocol.ToolKindCustom {
			encoded := responsesToolWire{Type: "custom", Name: tool.Name, Description: tool.Description}
			if tool.CustomFormat != nil {
				encoded.Format = &responsesCustomToolFormat{Type: "grammar", Syntax: tool.CustomFormat.Syntax, Definition: tool.CustomFormat.Definition}
			}
			tools = append(tools, encoded)
			continue
		}
		// An Anthropic-defined tool comes back callable, counted as a transform
		// in the request dispositions; a server tool comes back as it was and
		// is dropped, counted there too.
		tool = tool.Materialized()
		if tool.ServerTool() {
			continue
		}
		function := responsesToolWire{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema, Strict: tool.Strict}
		if tool.Namespace == "" {
			tools = append(tools, function)
			continue
		}
		// A namespace's functions go back into one namespace tool, at the
		// place of its first function.
		member, _ := json.Marshal(function)
		if index, grouped := namespaces[tool.Namespace]; grouped {
			tools[index].Tools = append(tools[index].Tools, member)
			continue
		}
		namespaces[tool.Namespace] = len(tools)
		tools = append(tools, responsesToolWire{
			Type: "namespace", Name: tool.Namespace, Description: tool.NamespaceDescription,
			Tools: []json.RawMessage{member},
		})
	}
	if imageGeneration != nil {
		tool := responsesToolWire{
			Type: "image_generation", Model: imageGeneration.Model,
			Quality: imageGeneration.Quality, Size: imageGeneration.Size,
			OutputFormat:      imageGeneration.OutputFormat,
			OutputCompression: imageGeneration.OutputCompression,
			Moderation:        imageGeneration.Moderation, Background: imageGeneration.Background,
			InputFidelity: imageGeneration.InputFidelity, PartialImages: imageGeneration.PartialImages,
			Action: imageGeneration.Action,
		}
		if imageGeneration.InputImageMask != nil {
			tool.InputImageMask = &responsesImageGenMaskWire{
				ImageURL: imageGeneration.InputImageMask.EncodedImage,
				FileID:   imageGeneration.InputImageMask.FileID,
			}
		}
		tools = append(tools, tool)
	}
	encoded := make([]json.RawMessage, 0, len(tools)+len(carried))
	for _, tool := range tools {
		body, _ := json.Marshal(tool)
		encoded = append(encoded, body)
	}
	for _, tool := range carried {
		encoded = append(encoded, append(json.RawMessage(nil), tool.Raw...))
	}
	if len(encoded) == 0 {
		// Every declared tool was one this target cannot express. An empty
		// list says nothing the absent member does not, and leaving it out is
		// what lets the tool choice be gated on it.
		return nil
	}
	body, _ := json.Marshal(encoded)
	return body
}

func encodeResponsesOutputFormat(output llmprotocol.OutputFormat, verbosity string) *responsesTextWire {
	if (output.Kind == "" || output.Kind == llmprotocol.OutputText) && verbosity == "" {
		return nil
	}
	wire := &responsesTextWire{}
	if verbosity != "" {
		wire.Verbosity, _ = json.Marshal(verbosity)
	}
	if output.Kind != "" && output.Kind != llmprotocol.OutputText {
		wire.Format = &responsesFormatWire{
			Type: string(output.Kind), Name: output.Name,
			Description: output.Description, Strict: output.Strict, Schema: output.Schema,
		}
	}
	return wire
}

func encodeResponsesMessage(message llmprotocol.Message, textDirection string) ([]responsesItemWire, error) {
	role, err := wireRole(message.Role)
	if err != nil {
		return nil, err
	}
	state := responsesMessageEncodingState{messageID: message.ID, role: role, textDirection: textDirection}
	for _, content := range message.Content {
		if err := state.appendContent(content); err != nil {
			return nil, err
		}
	}
	if err := state.flushOrdinary(); err != nil {
		return nil, err
	}
	if err := state.flushReasoning(); err != nil {
		return nil, err
	}
	return state.items, nil
}

type responsesMessageEncodingState struct {
	messageID     string
	role          string
	textDirection string
	ordinary      []llmprotocol.Content
	reasoning     []llmprotocol.Content
	items         []responsesItemWire
}

func (state *responsesMessageEncodingState) appendContent(content llmprotocol.Content) error {
	switch content.Kind {
	case llmprotocol.ContentToolCall:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendToolCall(content.ToolCall)
	case llmprotocol.ContentToolResult:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendToolResult(content.ToolResult)
	case llmprotocol.ContentReasoning:
		if err := state.flushOrdinary(); err != nil {
			return err
		}
		state.reasoning = append(state.reasoning, content)
	case llmprotocol.ContentGeneratedImage:
		if err := state.flushPending(); err != nil {
			return err
		}
		return state.appendGeneratedImage(content.GeneratedImage)
	case llmprotocol.ContentUnmodeled:
		if call, webSearch := carriedWebSearchCall(content); webSearch {
			if err := state.flushPending(); err != nil {
				return err
			}
			state.items = append(state.items, call)
			return nil
		}
		return state.appendCarriedBlock(content)
	default:
		if err := state.flushReasoning(); err != nil {
			return err
		}
		state.ordinary = append(state.ordinary, content)
	}
	return nil
}

// appendCarriedBlock handles a block belonging to the contract it came from.
// Responses names no Anthropic block, so it is dropped here and the drop is
// recorded beside the encoded request. The table's transform row is the one
// exception: a document whose source is text becomes a text part, because the
// block holds the text the turn is about.
func (state *responsesMessageEncodingState) appendCarriedBlock(content llmprotocol.Content) error {
	text, transformed := carriedDocumentText(content)
	if !transformed {
		return nil
	}
	if err := state.flushReasoning(); err != nil {
		return err
	}
	state.ordinary = append(state.ordinary, llmprotocol.Content{
		Kind: llmprotocol.ContentText, Text: text,
	})
	return nil
}

func (state *responsesMessageEncodingState) flushPending() error {
	if err := state.flushOrdinary(); err != nil {
		return err
	}
	return state.flushReasoning()
}

func (state *responsesMessageEncodingState) appendGeneratedImage(image *llmprotocol.GeneratedImage) error {
	if image == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_generated_image", "generated image is invalid", nil)
	}
	item := responsesItemWire{
		Type:   "image_generation_call",
		ID:     state.itemID("image_generation_call"),
		Status: string(image.Status),
	}
	if image.Result != nil {
		result := *image.Result
		item.Result = &result
	}
	state.items = append(state.items, item)
	return nil
}

func (state *responsesMessageEncodingState) flushOrdinary() error {
	if len(state.ordinary) == 0 {
		return nil
	}
	content, err := encodeResponsesContent(state.ordinary, state.textDirection)
	if err != nil {
		return err
	}
	item := responsesItemWire{
		Type: "message", ID: state.itemID("message"),
		Role: state.role, Content: content,
	}
	if state.textDirection == "output" {
		item.Status = "completed"
	}
	state.items = append(state.items, item)
	state.ordinary = nil
	return nil
}

func (state *responsesMessageEncodingState) appendToolCall(call *llmprotocol.ToolCall) error {
	if call == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_call", "tool call is invalid", nil)
	}
	item := responsesItemWire{
		Type: "function_call", ID: state.itemID("function_call"),
		CallID: call.ID, Name: call.Name, Arguments: call.Arguments, Namespace: call.Namespace,
	}
	if call.Kind == llmprotocol.ToolKindCustom {
		item.Type, item.Input = "custom_tool_call", call.Arguments
		item.Arguments = ""
	}
	state.items = append(state.items, item)
	return nil
}

func (state *responsesMessageEncodingState) appendToolResult(result *llmprotocol.ToolResult) error {
	if result == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool_result", "tool result is invalid", nil)
	}
	output, err := encodeResponsesContent(result.Content, "input")
	if err != nil {
		return err
	}
	item := responsesItemWire{
		Type: "function_call_output", ID: state.itemID("function_call_output"),
		CallID: result.CallID, Output: output,
	}
	if result.Kind == llmprotocol.ToolKindCustom {
		item.Type = "custom_tool_call_output"
	}
	state.items = append(state.items, item)
	return nil
}

func (state *responsesMessageEncodingState) flushReasoning() error {
	if len(state.reasoning) == 0 {
		return nil
	}
	item := responsesItemWire{
		Type: "reasoning", ID: state.itemID("reasoning"),
	}
	summaries := make([]map[string]string, 0, len(state.reasoning))
	texts := make([]map[string]string, 0, len(state.reasoning))
	for _, content := range state.reasoning {
		if encrypted, carrierOnly := encryptedReasoningOf(content); encrypted != nil {
			item.EncryptedContent = encrypted
			if carrierOnly {
				continue
			}
		}
		if content.Reasoning == llmprotocol.ReasoningScopeSummary {
			summaries = append(summaries, map[string]string{"type": "summary_text", "text": content.Text})
		} else {
			texts = append(texts, map[string]string{"type": "reasoning_text", "text": content.Text})
		}
	}
	var err error
	item.Summary, err = json.Marshal(summaries)
	if err != nil {
		return err
	}
	if len(texts) > 0 {
		item.Content, err = json.Marshal(texts)
		if err != nil {
			return err
		}
	}
	state.items = append(state.items, item)
	state.reasoning = nil
	return nil
}

func responsesItemID(messageID string, index int, kind string) string {
	if index == 0 && messageID != "" {
		return messageID
	}
	return llmprotocol.StableID("responses-item", messageID, fmt.Sprint(index), kind)
}

func (state *responsesMessageEncodingState) itemID(kind string) string {
	if state.textDirection == "input" {
		if len(state.items) == 0 {
			return state.messageID
		}
		return ""
	}
	return responsesItemID(state.messageID, len(state.items), kind)
}

func decodeResponsesReasoning(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	providerOutput bool,
) ([]llmprotocol.Content, error) {
	var summaries []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	var err error
	if providerOutput {
		err = decodeProviderValue(raw, &summaries, policy)
	} else {
		err = decodeWireValue(raw, &summaries, policy)
	}
	if err != nil {
		return nil, err
	}
	contents := make([]llmprotocol.Content, 0, len(summaries))
	for _, summary := range summaries {
		if summary.Type != "summary_text" {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_reasoning_summary", "Responses reasoning summary is unsupported", nil)
		}
		contents = append(contents, llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: summary.Text, Reasoning: llmprotocol.ReasoningScopeSummary,
		})
	}
	return contents, nil
}

func encodeResponsesContent(contents []llmprotocol.Content, direction string) (json.RawMessage, error) {
	parts := make([]responsesContentWire, 0, len(contents))
	for _, content := range withoutCarriedContent(contents) {
		switch content.Kind {
		case llmprotocol.ContentText:
			part := responsesContentWire{Type: direction + "_text", Text: content.Text}
			if direction == "output" {
				part.Annotations = responsesAnnotations(content.Citations)
			}
			parts = append(parts, part)
		case llmprotocol.ContentRefusal:
			parts = append(parts, responsesContentWire{Type: "refusal", Refusal: content.Text})
		case llmprotocol.ContentReasoning:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "reasoning_content_position", "Responses reasoning must be encoded as an ordered reasoning item", nil)
		case llmprotocol.ContentImage:
			if direction != "input" {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "image_content_position", "Responses output messages cannot contain input images", nil)
			}
			imageURL := content.URL
			if content.Data != "" {
				imageURL = "data:" + content.MediaType + ";base64," + content.Data
			}
			parts = append(parts, responsesContentWire{Type: "input_image", ImageURL: imageURL, FileID: content.FileID, Detail: content.Detail})
		case llmprotocol.ContentFile:
			if direction != "input" {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "file_content_position", "Responses output messages cannot contain input files", nil)
			}
			parts = append(parts, responsesContentWire{Type: "input_file", FileURL: content.URL, FileID: content.FileID, FileData: content.Data, Filename: content.Filename, Detail: content.Detail})
		default:
			return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "content cannot be encoded as Responses", nil)
		}
	}
	return json.Marshal(parts)
}

func encodeResponsesToolChoice(choice llmprotocol.ToolChoice) json.RawMessage {
	switch choice.Mode {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
		body, _ := json.Marshal(choice.Mode)
		return body
	case llmprotocol.ToolChoiceNamed:
		kind := "function"
		if choice.Kind == llmprotocol.ToolKindCustom {
			kind = "custom"
		}
		body, _ := json.Marshal(map[string]string{"type": kind, "name": choice.Name})
		return body
	case llmprotocol.ToolChoiceImageGeneration:
		body, _ := json.Marshal(map[string]string{"type": "image_generation"})
		return body
	default:
		return nil
	}
}
