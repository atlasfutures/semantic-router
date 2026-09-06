package llmprotocol

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Accept-by-default leaves the shape rules and the hard limits standing, and
// the gateway does not replay a body-level 400, so each of these refusals is a
// user-visible failed turn. Every one names the member it refused and, where
// the member sits in a list, the element it came from. The client repairs its
// own request from the message; the operator finds the block from the cause.
//
// This file holds what a whole request must state. What one content block must
// state is in validate_content.go, and what a tool declaration must state is
// in validate_tools.go.

func ValidateRequest(request Request, limits Limits) error {
	blocks, err := validateRequestEnvelope(request, limits)
	if err != nil {
		return err
	}
	if messageErr := validateRequestMessages(request, limits, &blocks); messageErr != nil {
		return messageErr
	}
	if instructionErr := validateRequestInstructions(request.Instructions, limits, &blocks); instructionErr != nil {
		return instructionErr
	}
	if limits.ContentBlocks > 0 && blocks > limits.ContentBlocks {
		return NewFieldError(ErrorInvalidRequest, "content_limit",
			"content block limit exceeded", "", "messages.content").
			WithCount("", blocks, limits.ContentBlocks)
	}
	namedTools, schemaBytes, err := validateRequestTools(request.Tools, limits)
	if err != nil {
		return err
	}
	if err := validateToolChoice(request.ToolChoice, namedTools, len(request.Tools), request.ImageGeneration != nil); err != nil {
		return err
	}
	if err := locateRefusal(
		validateImageGenerationOptions(request.ImageGeneration, limits), "", "image_generation",
	); err != nil {
		return err
	}
	if err := validateOutputFormat(request.OutputFormat, schemaBytes, limits); err != nil {
		return err
	}
	if err := validateSampling(request.Sampling, limits); err != nil {
		return err
	}
	return validateReasoning(request, limits)
}

// messageLocation and instructionLocation name which element of a list a
// refusal came from. A resumed Claude Code turn carries hundreds of messages
// and the refused body is never stored, so the index is the only thing that
// makes a refusal findable.
func messageLocation(index int) string {
	return fmt.Sprintf("message %d", index)
}

func instructionLocation(index int) string {
	return fmt.Sprintf("instruction %d", index)
}

func validateRequestEnvelope(request Request, limits Limits) (int, error) {
	if err := validateRequestIdentity(request, limits); err != nil {
		return 0, err
	}
	if err := validateRequestCardinality(request, limits); err != nil {
		return 0, err
	}
	if err := validateCandidateCount(request.CandidateCount, limits.Candidates); err != nil {
		return 0, err
	}
	if !request.Stream && (request.StreamOptions.IncludeUsage != nil || request.StreamOptions.IncludeObfuscation != nil) {
		return 0, NewFieldError(ErrorInvalidRequest, "stream_options_without_stream",
			"stream options require streaming", "", "stream_options")
	}
	blocks := 0
	for _, instruction := range request.Instructions {
		blocks += len(instruction.Content)
	}
	if err := validateMetadata(request.Metadata, limits); err != nil {
		return 0, err
	}
	return blocks, nil
}

func validateRequestIdentity(request Request, limits Limits) error {
	if strings.TrimSpace(request.Model) == "" {
		return NewFieldError(ErrorInvalidRequest, "model_required", "model is required", "", "model")
	}
	if exceeds(request.Model, limits.ModelBytes) {
		return NewFieldError(ErrorInvalidRequest, "model_limit",
			"model exceeds the configured limit", "", "model").
			WithCount("bytes", len(request.Model), limits.ModelBytes)
	}
	if request.Generation == 0 {
		return NewFieldError(ErrorInternal, "generation_required",
			"semantic generation is required", "", "generation")
	}
	if exceeds(request.EndUserID, limits.IdentifierBytes) {
		return NewFieldError(ErrorInvalidRequest, "end_user_id_limit",
			"end-user ID exceeds the configured limit", "", "end_user_id").
			WithCount("bytes", len(request.EndUserID), limits.IdentifierBytes)
	}
	if request.PreviousResponseID != "" && request.ConversationID != "" {
		return NewFieldError(ErrorInvalidRequest, "conflicting_conversation_state",
			"previous response and conversation references cannot be used together",
			"", "previous_response_id")
	}
	if request.Truncation != "" && request.Truncation != "disabled" && request.Truncation != "auto" {
		return NewFieldError(ErrorInvalidRequest, "invalid_truncation",
			"truncation mode is invalid", "", "truncation")
	}
	return nil
}

func validateRequestCardinality(request Request, limits Limits) error {
	if limits.Messages > 0 && len(request.Messages) > limits.Messages {
		return NewFieldError(ErrorInvalidRequest, "messages_limit",
			"message limit exceeded", "", "messages").
			WithCount("", len(request.Messages), limits.Messages)
	}
	if limits.Instructions > 0 && len(request.Instructions) > limits.Instructions {
		return NewFieldError(ErrorInvalidRequest, "instructions_limit",
			"instruction limit exceeded", "", "instructions").
			WithCount("", len(request.Instructions), limits.Instructions)
	}
	if limits.Tools > 0 && len(request.Tools) > limits.Tools {
		return NewFieldError(ErrorInvalidRequest, "tools_limit",
			"tool limit exceeded", "", "tools").
			WithCount("", len(request.Tools), limits.Tools)
	}
	return nil
}

func validateCandidateCount(candidateCount *int64, limit int) error {
	if candidateCount == nil {
		return nil
	}
	if *candidateCount <= 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_candidate_count",
			"candidate count must be positive", "", "candidate_count")
	}
	if limit > 0 && *candidateCount > int64(limit) {
		return NewFieldError(ErrorInvalidRequest, "candidate_count_limit",
			"candidate count exceeds the configured limit", "", "candidate_count").
			WithCount("", int(*candidateCount), limit)
	}
	return nil
}

func validateMetadata(metadata map[string]string, limits Limits) error {
	metadataBytes := 0
	if limits.MetadataEntries > 0 && len(metadata) > limits.MetadataEntries {
		return NewFieldError(ErrorInvalidRequest, "metadata_entries_limit",
			"metadata entry limit exceeded", "", "metadata").
			WithCount("", len(metadata), limits.MetadataEntries)
	}
	for key, value := range metadata {
		if limits.MetadataKeyBytes > 0 && len(key) > limits.MetadataKeyBytes ||
			limits.MetadataValueBytes > 0 && len(value) > limits.MetadataValueBytes {
			return NewFieldError(ErrorInvalidRequest, "metadata_field_limit",
				"metadata key or value limit exceeded", "", "metadata")
		}
		metadataBytes += len(key) + len(value)
	}
	if limits.MetadataBytes > 0 && metadataBytes > limits.MetadataBytes {
		return NewFieldError(ErrorInvalidRequest, "metadata_limit",
			"metadata limit exceeded", "", "metadata").
			WithCount("bytes", metadataBytes, limits.MetadataBytes)
	}
	return nil
}

func validateRequestMessages(request Request, limits Limits, blocks *int) error {
	toolCalls := make(map[string]struct{})
	toolResults := make(map[string]struct{})
	for index, message := range request.Messages {
		if err := validateRequestMessage(
			message, messageLocation(index), limits, blocks, toolCalls, toolResults,
			request.PreviousResponseID != "",
		); err != nil {
			return err
		}
	}
	return nil
}

func validateRequestMessage(
	message Message,
	location string,
	limits Limits,
	blocks *int,
	toolCalls map[string]struct{},
	toolResults map[string]struct{},
	retainedHistory bool,
) error {
	if err := validateMessageEnvelope(message, location, limits); err != nil {
		return err
	}
	*blocks += len(message.Content)
	for index, content := range message.Content {
		blockLocation := contentLocation(location, index, content.Kind)
		if err := validateMessageContent(message.Role, content, blockLocation, limits, blocks); err != nil {
			return err
		}
		if err := recordToolLink(content, blockLocation, toolCalls, toolResults, retainedHistory); err != nil {
			return err
		}
	}
	return nil
}

func validateMessageEnvelope(message Message, location string, limits Limits) error {
	if exceeds(message.ID, limits.IdentifierBytes) {
		return NewFieldError(ErrorInvalidRequest, "message_id_limit",
			"message ID exceeds the configured limit", location, "messages.id").
			WithCount("bytes", len(message.ID), limits.IdentifierBytes)
	}
	if !validRequestRole(message.Role) {
		return NewFieldError(ErrorInvalidRequest, "invalid_role",
			"message role is invalid", location, "messages.role")
	}
	if len(message.Content) == 0 {
		return NewFieldError(ErrorInvalidRequest, "empty_message",
			"messages must contain at least one content block", location, "messages.content")
	}
	if message.Role == RoleTool && len(message.Content) != 1 {
		return NewFieldError(ErrorInvalidRequest, "tool_message_cardinality",
			"tool messages contain exactly one tool result", location, "messages.content")
	}
	return nil
}

func validRequestRole(role Role) bool {
	return role == RoleSystem || role == RoleDeveloper || role == RoleUser ||
		role == RoleAssistant || role == RoleTool
}

func validateRequestInstructions(instructions []InstructionBlock, limits Limits, blocks *int) error {
	for index, instruction := range instructions {
		location := instructionLocation(index)
		if instruction.Role != RoleSystem && instruction.Role != RoleDeveloper {
			return NewFieldError(ErrorInvalidRequest, "invalid_instruction_role",
				"instruction role must be system or developer", location, "instructions.role")
		}
		if len(instruction.Content) == 0 {
			return NewFieldError(ErrorInvalidRequest, "empty_instruction",
				"instructions must contain at least one content block", location, "instructions.content")
		}
		for contentIndex, content := range instruction.Content {
			blockLocation := contentLocation(location, contentIndex, content.Kind)
			if content.Kind == ContentToolCall || content.Kind == ContentToolResult {
				return NewFieldError(ErrorInvalidRequest, "invalid_instruction_content",
					"instructions cannot contain tool control blocks", location, "instructions.content")
			}
			if err := validateContent(content, blockLocation, blocks, limits, 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateOutputFormat(format OutputFormat, schemaBytes int, limits Limits) error {
	switch format.Kind {
	case "", OutputText, OutputJSONObject, OutputJSONSchema:
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_output_format",
			"output format is invalid", "", "output_format")
	}
	if format.Kind != OutputJSONSchema {
		if !outputFormatHasSchemaFields(format) {
			return nil
		}
		return NewFieldError(ErrorInvalidRequest, "invalid_output_format",
			"output format contains fields from another format kind", "", "output_format")
	}
	return validateJSONSchemaOutputFormat(format, schemaBytes, limits)
}

func outputFormatHasSchemaFields(format OutputFormat) bool {
	return len(format.Schema) > 0 || format.Name != "" || format.Description != "" || format.Strict != nil
}

func validateJSONSchemaOutputFormat(format OutputFormat, schemaBytes int, limits Limits) error {
	if len(format.Schema) == 0 || !json.Valid(format.Schema) {
		return NewFieldError(ErrorInvalidRequest, "invalid_output_schema",
			"output JSON Schema is invalid", "", "output_format.schema")
	}
	if limits.SchemaBytes > 0 && len(format.Schema) > limits.SchemaBytes {
		return NewFieldError(ErrorInvalidRequest, "schema_limit",
			"output schema limit exceeded", "", "output_format.schema").
			WithCount("bytes", len(format.Schema), limits.SchemaBytes)
	}
	if limits.SchemaBytes > 0 && schemaBytes+len(format.Schema) > limits.SchemaBytes {
		return NewFieldError(ErrorInvalidRequest, "schema_limit",
			"total schema limit exceeded", "", "output_format.schema").
			WithCount("bytes", schemaBytes+len(format.Schema), limits.SchemaBytes)
	}
	if err := validateSchemaObject(format.Schema, "output schema", "", "output_format.schema", limits); err != nil {
		return err
	}
	if strings.TrimSpace(format.Name) == "" {
		return NewFieldError(ErrorInvalidRequest, "output_schema_name_required",
			"output JSON Schema requires a name", "", "output_format.name")
	}
	return nil
}

func validateSchemaObject(schema json.RawMessage, label, location, field string, limits Limits) error {
	if err := ValidateJSONObject(schema, limits.JSONDepth); err != nil {
		return NewFieldError(ErrorInvalidRequest, "invalid_schema",
			label+" must be a JSON object", location, field)
	}
	return nil
}

func validateSampling(sampling Sampling, limits Limits) error {
	if err := validateSamplingScalars(sampling); err != nil {
		return err
	}
	return validateStopSequences(sampling.Stop, limits)
}

func validateSamplingScalars(sampling Sampling) error {
	if err := validateSamplingProbability(sampling); err != nil {
		return err
	}
	if err := validateSamplingCounts(sampling); err != nil {
		return err
	}
	for _, penalty := range []struct {
		field string
		value *float64
	}{
		{"frequency_penalty", sampling.FrequencyPenalty},
		{"presence_penalty", sampling.PresencePenalty},
	} {
		if penalty.value != nil && (!finiteFloat(*penalty.value) || *penalty.value < -2 || *penalty.value > 2) {
			return NewFieldError(ErrorInvalidRequest, "invalid_penalty",
				"sampling penalty must be between -2 and 2", "", penalty.field)
		}
	}
	return nil
}

func validateSamplingProbability(sampling Sampling) error {
	if sampling.Temperature != nil && (!finiteFloat(*sampling.Temperature) || *sampling.Temperature < 0 || *sampling.Temperature > 2) {
		return NewFieldError(ErrorInvalidRequest, "invalid_temperature",
			"temperature must be between 0 and 2", "", "temperature")
	}
	if sampling.TopP != nil && (!finiteFloat(*sampling.TopP) || *sampling.TopP < 0 || *sampling.TopP > 1) {
		return NewFieldError(ErrorInvalidRequest, "invalid_top_p",
			"top_p must be between 0 and 1", "", "top_p")
	}
	return nil
}

func finiteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateSamplingCounts(sampling Sampling) error {
	if sampling.TopK != nil && *sampling.TopK < 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_top_k",
			"top_k cannot be negative", "", "top_k")
	}
	if sampling.MaxOutputTokens != nil && *sampling.MaxOutputTokens < 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_max_output_tokens",
			"max output tokens cannot be negative", "", "max_output_tokens")
	}
	return nil
}

func validateStopSequences(stops []string, limits Limits) error {
	if limits.StopSequences > 0 && len(stops) > limits.StopSequences {
		return NewFieldError(ErrorInvalidRequest, "stop_limit",
			"too many stop sequences", "", "stop_sequences").
			WithCount("", len(stops), limits.StopSequences)
	}
	stopBytes := 0
	for _, stop := range stops {
		if stop == "" {
			return NewFieldError(ErrorInvalidRequest, "invalid_stop",
				"stop sequences must be non-empty and bounded", "", "stop_sequences")
		}
		stopBytes += len(stop)
		if limits.StopBytes > 0 && stopBytes > limits.StopBytes {
			return NewFieldError(ErrorInvalidRequest, "stop_bytes_limit",
				"stop sequences exceed the configured limit", "", "stop_sequences").
				WithCount("bytes", stopBytes, limits.StopBytes)
		}
	}
	return nil
}

func validateReasoning(request Request, limits Limits) error {
	if err := validateReasoningEffort(request, limits); err != nil {
		return err
	}
	if err := validateReasoningDisplay(request); err != nil {
		return err
	}
	return validateReasoningMode(request)
}

func validateReasoningEffort(request Request, limits Limits) error {
	if exceeds(request.ReasoningEffort, limits.ReasoningEffortBytes) {
		return NewFieldError(ErrorInvalidRequest, "reasoning_effort_limit",
			"reasoning effort exceeds the configured limit", "", "reasoning_effort").
			WithCount("bytes", len(request.ReasoningEffort), limits.ReasoningEffortBytes)
	}
	switch request.ReasoningEffort {
	case "", "none", "minimal", "low", "medium", "high", "xhigh", "max":
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_effort",
			"reasoning effort is invalid", "", "reasoning_effort")
	}
	if request.ReasoningBudgetTokens != nil && *request.ReasoningBudgetTokens <= 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_budget",
			"reasoning budget must be positive", "", "reasoning_budget_tokens")
	}
	return nil
}

func validateReasoningDisplay(request Request) error {
	switch request.ReasoningDisplay {
	case "", "summarized", "omitted":
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_display",
			"reasoning display mode is invalid", "", "reasoning_display")
	}
	if request.ReasoningDisplay != "" &&
		request.ReasoningMode != ReasoningModeEnabled &&
		request.ReasoningMode != ReasoningModeAdaptive {
		return NewFieldError(ErrorInvalidRequest, "conflicting_reasoning_display",
			"reasoning display requires enabled or adaptive reasoning", "", "reasoning_display")
	}
	return nil
}

func validateReasoningMode(request Request) error {
	switch request.ReasoningMode {
	case "", ReasoningModeEnabled:
	case ReasoningModeDisabled, ReasoningModeAdaptive:
		// Only a token budget contradicts these modes: neither variant of the
		// thinking object carries one. An effort level is a separate control
		// that Anthropic documents as the way to steer thinking depth once a
		// budget is no longer accepted, and Claude Code sends the pair on
		// every turn, so the Router routes it.
		if request.ReasoningBudgetTokens != nil {
			return NewFieldError(ErrorInvalidRequest, "conflicting_reasoning_control",
				string(request.ReasoningMode)+" reasoning cannot include a token budget",
				"", "reasoning_budget_tokens")
		}
	default:
		return NewFieldError(ErrorInvalidRequest, "invalid_reasoning_mode",
			"reasoning mode is invalid", "", "reasoning_mode")
	}
	if request.ReasoningMode == ReasoningModeEnabled && request.ReasoningBudgetTokens == nil {
		return NewFieldError(ErrorInvalidRequest, "reasoning_budget_required",
			"enabled reasoning requires a token budget", "", "reasoning_budget_tokens")
	}
	return nil
}
