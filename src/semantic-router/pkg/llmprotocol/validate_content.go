package llmprotocol

import (
	"encoding/json"
	"fmt"
	"strings"
)

// A content block is where a refusal is hardest to find. A Claude Code turn
// carries hundreds of them, the refused body is the user's conversation and is
// never stored, and the block that failed is identified by its position alone.
// So every refusal raised here names the block it came from and the member of
// that block it refused, and nothing else about it.
//
// This file holds the block contract. What a whole request must state is in
// validate.go, and what a tool declaration must state is in validate_tools.go.

// contentLocation names a block by where it sits and what it claims to be.
// The parent is the message, the instruction, or the tool result the block
// belongs to, so a nested block reads as the path that reaches it.
func contentLocation(parent string, index int, kind ContentKind) string {
	location := fmt.Sprintf("content block %d of type %q", index, kind)
	if parent == "" {
		return location
	}
	return parent + " " + location
}

func validateMessageContent(role Role, content Content, location string, limits Limits, blocks *int) error {
	if err := validateRoleContent(role, content, location); err != nil {
		return err
	}
	return validateContent(content, location, blocks, limits, 0)
}

func validateRoleContent(role Role, content Content, location string) error {
	invalid := func(message string) error {
		return NewFieldError(ErrorInvalidRequest, "invalid_role_content", message, location, "messages.content")
	}
	switch role {
	case RoleSystem, RoleDeveloper:
		if contentKindIsOneOf(content.Kind, ContentToolCall, ContentToolResult, ContentGeneratedImage) {
			return invalid("system and developer messages cannot contain tool control blocks")
		}
	case RoleAssistant:
		if content.Kind == ContentToolResult {
			return invalid("assistant messages cannot contain tool results")
		}
	case RoleTool:
		if content.Kind != ContentToolResult {
			return invalid("tool messages may contain only tool results")
		}
	case RoleUser:
		if contentKindIsOneOf(content.Kind, ContentToolCall, ContentToolResult, ContentRefusal, ContentGeneratedImage) {
			return invalid("user messages contain assistant-only content")
		}
	}
	return nil
}

func contentKindIsOneOf(kind ContentKind, candidates ...ContentKind) bool {
	for _, candidate := range candidates {
		if kind == candidate {
			return true
		}
	}
	return false
}

func validateContent(content Content, location string, blocks *int, limits Limits, depth int) error {
	if limits.ToolResultDepth <= 0 || depth > limits.ToolResultDepth {
		return NewFieldError(ErrorInvalidRequest, "tool_result_depth",
			"nested tool result depth exceeded", location, "content.content").
			WithCount("", depth, limits.ToolResultDepth)
	}
	if err := validateCacheDirective(content.Cache, location, "content.cache_control"); err != nil {
		return err
	}
	switch content.Kind {
	case ContentText, ContentRefusal, ContentReasoning:
		return validateTextContent(content, location, limits)
	case ContentImage, ContentAudio, ContentVideo, ContentFile:
		return validateMediaContent(content, location, limits)
	case ContentToolCall:
		return validateToolCallContent(content, location, limits)
	case ContentToolResult:
		return validateToolResultContent(content, location, blocks, limits, depth)
	case ContentGeneratedImage:
		return locateRefusal(
			validateGeneratedImageContent(content, limits), location, "content.generated_image",
		)
	case ContentUnmodeled:
		return validateUnmodeledContent(content, location)
	default:
		return NewFieldError(ErrorInvalidRequest, "unknown_content_kind",
			"content kind is unsupported", location, "content.type")
	}
}

// validateUnmodeledContent keeps the carrier inert. It holds source bytes and
// nothing else, so any semantic field beside it would be a field the Router
// could read by accident.
func validateUnmodeledContent(content Content, location string) error {
	invalid := func(message, field string) error {
		return NewFieldError(ErrorInvalidRequest, "invalid_content", message, location, field)
	}
	if content.Unmodeled == nil || content.Unmodeled.Format == "" || len(content.Unmodeled.Raw) == 0 {
		return invalid("carried content requires its source format and bytes", "content.type")
	}
	if !json.Valid(content.Unmodeled.Raw) {
		return invalid("carried content must be the JSON the client sent", "content.type")
	}
	if content.Text != "" {
		return invalid("carried content cannot hold fields from another content kind", "content.text")
	}
	if content.Cache != nil {
		return invalid("carried content cannot hold fields from another content kind", "content.cache_control")
	}
	if len(content.Citations) != 0 {
		return invalid("carried content cannot hold fields from another content kind", "content.citations")
	}
	if foreign := foreignTextField(content); foreign != "" {
		return invalid("carried content cannot hold fields from another content kind", foreign)
	}
	return nil
}

// validateCacheDirective is shared by a content block and a tool declaration,
// which spell the member at different paths, so the caller states the base.
func validateCacheDirective(cache *CacheDirective, location, field string) error {
	if cache == nil {
		return nil
	}
	if cache.Type != "ephemeral" {
		return NewFieldError(ErrorInvalidRequest, "invalid_cache_directive",
			"cache directive type must be ephemeral", location, field+".type")
	}
	switch cache.TTL {
	case "", "5m", "1h":
		return nil
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_cache_directive",
			"cache directive TTL must be 5m or 1h", location, field+".ttl")
	}
}

func validateTextContent(content Content, location string, limits Limits) error {
	if foreign := foreignTextField(content); foreign != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"text content contains fields from another content kind", location, foreign)
	}
	if content.Kind == ContentReasoning {
		if err := validateReasoningContent(content, location); err != nil {
			return err
		}
	}
	if content.Kind != ContentText && len(content.Citations) != 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"only text content may contain citations", location, "content.citations")
	}
	if limits.TextBytes > 0 && len(content.Text) > limits.TextBytes {
		return NewFieldError(ErrorInvalidRequest, "text_limit",
			"content text exceeds the configured limit", location, "content.text").
			WithCount("bytes", len(content.Text), limits.TextBytes)
	}
	return locateRefusal(
		ValidateTextCitations(content.Text, content.Citations, limits), location, "content.citations",
	)
}

func validateReasoningContent(content Content, location string) error {
	switch content.Reasoning {
	case "", ReasoningScopeText, ReasoningScopeSummary:
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_scope",
			"reasoning content scope is unsupported", location, "content.thinking")
	}
	if content.Reasoning == ReasoningScopeSummary && content.Signature != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_signature",
			"reasoning summaries cannot carry a private reasoning signature", location, "content.signature")
	}
	return nil
}

// presentMember is one row of a foreign-member table: the path to state if
// the member is set on a block whose kind does not name it.
type presentMember struct {
	field   string
	present bool
}

func firstPresent(members []presentMember) string {
	for _, member := range members {
		if member.present {
			return member.field
		}
	}
	return ""
}

// foreignTextField names the member a text, refusal or reasoning block holds
// that belongs to another content kind, so the refusal states which member to
// remove rather than that one exists.
func foreignTextField(content Content) string {
	return firstPresent([]presentMember{
		{"content.tool_use", content.ToolCall != nil},
		{"content.tool_use_id", content.ToolResult != nil},
		{"content.generated_image", content.GeneratedImage != nil},
		{"content.url", content.URL != ""},
		{"content.data", content.Data != ""},
		{"content.file_id", content.FileID != ""},
		{"content.filename", content.Filename != ""},
		{"content.media_type", content.MediaType != ""},
		{"content.detail", content.Detail != ""},
		{"content.signature", content.Kind != ContentReasoning && content.Signature != ""},
		{"content.thinking", content.Kind != ContentReasoning && content.Reasoning != ""},
	})
}

func validateMediaContent(content Content, location string, limits Limits) error {
	if err := validateMediaBounds(content, location, limits); err != nil {
		return err
	}
	if foreign := foreignMediaField(content); foreign != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"media content contains fields from another content kind", location, foreign)
	}
	if mediaSourceCount(content) != 1 {
		return NewFieldError(ErrorInvalidRequest, "media_reference_required",
			"media content requires exactly one data source or reference", location, "content.source")
	}
	if err := validateMediaSource(content, location); err != nil {
		return err
	}
	return validateMediaKindFields(content, location)
}

func foreignMediaField(content Content) string {
	return firstPresent([]presentMember{
		{"content.tool_use", content.ToolCall != nil},
		{"content.tool_use_id", content.ToolResult != nil},
		{"content.generated_image", content.GeneratedImage != nil},
		{"content.text", content.Text != ""},
		{"content.signature", content.Signature != ""},
		{"content.citations", len(content.Citations) != 0},
	})
}

func validateMediaKindFields(content Content, location string) error {
	if content.Kind != ContentImage && content.Kind != ContentFile && content.Detail != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"only image or file content may specify detail", location, "content.detail")
	}
	if content.Kind != ContentFile && content.Filename != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"only file content may specify a filename", location, "content.filename")
	}
	return nil
}

func validateMediaBounds(content Content, location string, limits Limits) error {
	if limits.MediaDataBytes > 0 && len(content.Data) > limits.MediaDataBytes {
		return NewFieldError(ErrorInvalidRequest, "media_data_limit",
			"inline media exceeds the configured limit", location, "content.data").
			WithCount("bytes", len(content.Data), limits.MediaDataBytes)
	}
	for _, reference := range []struct {
		field string
		value string
	}{
		{"content.url", content.URL},
		{"content.file_id", content.FileID},
		{"content.filename", content.Filename},
		{"content.media_type", content.MediaType},
		{"content.detail", content.Detail},
	} {
		if limits.MediaReferenceBytes > 0 && len(reference.value) > limits.MediaReferenceBytes {
			return NewFieldError(ErrorInvalidRequest, "media_reference_limit",
				"media reference exceeds the configured limit", location, reference.field).
				WithCount("bytes", len(reference.value), limits.MediaReferenceBytes)
		}
	}
	return nil
}

func mediaSourceCount(content Content) int {
	sources := 0
	for _, source := range []string{content.URL, content.Data, content.FileID} {
		if source != "" {
			sources++
		}
	}
	return sources
}

func validateMediaSource(content Content, location string) error {
	if content.URL != "" {
		return locateRefusal(validateMediaURL(content.URL), location, "content.url")
	}
	if content.Data == "" {
		return nil
	}
	if strings.TrimSpace(content.MediaType) == "" {
		return NewFieldError(ErrorInvalidRequest, "media_type_required",
			"inline media requires a media type", location, "content.media_type")
	}
	// The payload bytes are not read here. The Router routes the block, it
	// never opens it, and the provider validates the encoding it is given. So
	// decoding it at ingress buys nothing and costs one pass over every inline
	// image, while turning a body the provider would have judged into a 400.
	return nil
}

func validateToolCallContent(content Content, location string, limits Limits) error {
	call := content.ToolCall
	if call == nil || strings.TrimSpace(call.ID) == "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_call",
			"tool call requires an ID, name, and JSON arguments", location, "content.id")
	}
	if strings.TrimSpace(call.Name) == "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_call",
			"tool call requires an ID, name, and JSON arguments", location, "content.name")
	}
	if err := ValidateJSONObject([]byte(call.Arguments), limits.JSONDepth); err != nil {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_call",
			"tool call arguments must be one strict JSON object", location, "content.input")
	}
	if err := validateToolCallBounds(call, location, limits); err != nil {
		return err
	}
	if content.ToolResult != nil {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"tool call contains fields from another content kind", location, "content.tool_use_id")
	}
	if foreign := foreignToolControlField(content); foreign != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"tool call contains fields from another content kind", location, foreign)
	}
	return nil
}

func validateToolCallBounds(call *ToolCall, location string, limits Limits) error {
	if exceeds(call.ID, limits.IdentifierBytes) {
		return NewFieldError(ErrorInvalidRequest, "tool_call_limit",
			"tool call ID or name exceeds the configured limit", location, "content.id").
			WithCount("bytes", len(call.ID), limits.IdentifierBytes)
	}
	if exceeds(call.Name, limits.ToolNameBytes) {
		return NewFieldError(ErrorInvalidRequest, "tool_call_limit",
			"tool call ID or name exceeds the configured limit", location, "content.name").
			WithCount("bytes", len(call.Name), limits.ToolNameBytes)
	}
	if limits.ToolArgumentsBytes > 0 && len(call.Arguments) > limits.ToolArgumentsBytes {
		return NewFieldError(ErrorInvalidRequest, "tool_arguments_limit",
			"tool arguments exceed the configured limit", location, "content.input").
			WithCount("bytes", len(call.Arguments), limits.ToolArgumentsBytes)
	}
	return nil
}

func validateToolResultContent(content Content, location string, blocks *int, limits Limits, depth int) error {
	result := content.ToolResult
	if result == nil || strings.TrimSpace(result.CallID) == "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_result",
			"tool result requires a call ID", location, "content.tool_use_id")
	}
	if exceeds(result.CallID, limits.IdentifierBytes) {
		return NewFieldError(ErrorInvalidRequest, "tool_result_id_limit",
			"tool result call ID exceeds the configured limit", location, "content.tool_use_id").
			WithCount("bytes", len(result.CallID), limits.IdentifierBytes)
	}
	if content.ToolCall != nil {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"tool result contains fields from another content kind", location, "content.tool_use")
	}
	if foreign := foreignToolControlField(content); foreign != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_content",
			"tool result contains fields from another content kind", location, foreign)
	}
	return validateNestedToolResult(result.Content, location, blocks, limits, depth)
}

func foreignToolControlField(content Content) string {
	return firstPresent([]presentMember{
		{"content.text", content.Text != ""},
		{"content.url", content.URL != ""},
		{"content.data", content.Data != ""},
		{"content.citations", len(content.Citations) != 0},
		{"content.file_id", content.FileID != ""},
		{"content.media_type", content.MediaType != ""},
		{"content.filename", content.Filename != ""},
		{"content.detail", content.Detail != ""},
		{"content.signature", content.Signature != ""},
		{"content.generated_image", content.GeneratedImage != nil},
	})
}

func validateNestedToolResult(contents []Content, location string, blocks *int, limits Limits, depth int) error {
	*blocks += len(contents)
	if limits.ContentBlocks > 0 && *blocks > limits.ContentBlocks {
		return NewFieldError(ErrorInvalidRequest, "content_limit",
			"content block limit exceeded", location, "content.content").
			WithCount("", *blocks, limits.ContentBlocks)
	}
	for index, nested := range contents {
		if nested.Kind == ContentToolResult || nested.Kind == ContentToolCall {
			return NewFieldError(ErrorInvalidRequest, "nested_tool_control",
				"tool results cannot contain tool control blocks", location, "content.content")
		}
		if err := validateContent(nested, contentLocation(location, index, nested.Kind), blocks, limits, depth+1); err != nil {
			return err
		}
	}
	return nil
}

func recordToolLink(content Content, location string, calls, results map[string]struct{}, retainedHistory bool) error {
	switch {
	case content.Kind == ContentToolCall && content.ToolCall != nil:
		return recordToolCall(content.ToolCall, location, calls, results)
	case content.Kind == ContentToolResult && content.ToolResult != nil:
		return recordToolResult(content.ToolResult, location, calls, results, retainedHistory)
	default:
		return nil
	}
}

func recordToolCall(call *ToolCall, location string, calls, results map[string]struct{}) error {
	if _, resultAlreadySeen := results[call.ID]; resultAlreadySeen {
		return NewFieldError(ErrorInvalidRequest, "tool_result_order",
			"tool result must follow its request tool call", location, "content.id")
	}
	if _, duplicate := calls[call.ID]; duplicate {
		return NewFieldError(ErrorInvalidRequest, "duplicate_tool_call",
			"tool call IDs must be unique", location, "content.id")
	}
	calls[call.ID] = struct{}{}
	return nil
}

func recordToolResult(
	result *ToolResult,
	location string,
	calls map[string]struct{},
	results map[string]struct{},
	retainedHistory bool,
) error {
	if _, duplicate := results[result.CallID]; duplicate {
		return NewFieldError(ErrorInvalidRequest, "duplicate_tool_result",
			"tool results must be unique per call", location, "content.tool_use_id")
	}
	_, found := calls[result.CallID]
	if found && result.DeferredLink {
		return NewFieldError(ErrorInvalidRequest, "invalid_deferred_tool_result",
			"a locally linked tool result cannot be deferred", location, "content.tool_use_id")
	}
	if !found && (!retainedHistory || !result.DeferredLink) {
		return NewFieldError(ErrorInvalidRequest, "orphan_tool_result",
			"tool result must follow its request tool call or carry a retained-history link",
			location, "content.tool_use_id")
	}
	results[result.CallID] = struct{}{}
	return nil
}

// MarkDeferredToolLinks reconciles tool-result links against the messages
// currently materialized in the request. A missing call is deferred only while
// previous_response_id still identifies retained history; after that history is
// expanded locally, ordinary lifecycle validation applies to every result.
func MarkDeferredToolLinks(request *Request) {
	if request == nil {
		return
	}
	calls := make(map[string]struct{})
	for messageIndex := range request.Messages {
		for contentIndex := range request.Messages[messageIndex].Content {
			content := &request.Messages[messageIndex].Content[contentIndex]
			if content.Kind == ContentToolCall && content.ToolCall != nil {
				calls[content.ToolCall.ID] = struct{}{}
				continue
			}
			if content.Kind != ContentToolResult || content.ToolResult == nil {
				continue
			}
			_, local := calls[content.ToolResult.CallID]
			content.ToolResult.DeferredLink = request.PreviousResponseID != "" && !local
		}
	}
}
