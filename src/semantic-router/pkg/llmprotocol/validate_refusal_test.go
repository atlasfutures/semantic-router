package llmprotocol

import (
	"encoding/json"
	"strings"
	"testing"
)

// Rule 6, for the request contract itself. Accept-by-default leaves the shape
// rules and the hard limits standing, and every one of those is a user-visible
// failed turn: the gateway does not replay a body-level 400.
//
// So each has to say where it happened, in both places anyone looks. The cause
// reaches the ingress_request_refused line, which is all an operator gets --
// the refused body is the user's conversation and is never stored. The message
// reaches the client, which is all a client gets, and a client repairs its own
// request only from what the body names.
//
// The tool validator and the content decoder already do this. These are the
// rest: the envelope, the message and instruction walk, the sampling and
// reasoning controls, and every limit.

type refusalCase struct {
	name string
	// request is validated as given. limits, when set, replaces the default.
	request  Request
	limits   *Limits
	code     string
	field    string
	location string
	// message names extra text the client body must carry, such as the two
	// byte counts a limit refusal states.
	message []string
}

func refusalBaseRequest() Request {
	return Request{
		Generation: 1,
		Model:      "m",
		Messages: []Message{{
			Role:    RoleUser,
			Content: []Content{{Kind: ContentText, Text: "hi"}},
		}},
	}
}

func withRequest(mutate func(*Request)) Request {
	request := refusalBaseRequest()
	mutate(&request)
	return request
}

func tightLimits(mutate func(*Limits)) *Limits {
	limits := DefaultPolicy().Limits
	mutate(&limits)
	return &limits
}

func TestRequestRefusalsNameTheirField(t *testing.T) {
	var cases []refusalCase
	for _, group := range [][]refusalCase{
		envelopeRefusalCases(),
		cardinalityRefusalCases(),
		messageRefusalCases(),
		instructionRefusalCases(),
		outputFormatRefusalCases(),
		samplingRefusalCases(),
		reasoningRefusalCases(),
		contentRefusalCases(),
	} {
		cases = append(cases, group...)
	}
	seen := make(map[string]struct{}, len(cases))
	for _, test := range cases {
		if _, duplicate := seen[test.name]; duplicate {
			t.Fatalf("duplicate case name %q", test.name)
		}
		seen[test.name] = struct{}{}
		t.Run(test.name, func(t *testing.T) {
			limits := DefaultPolicy().Limits
			if test.limits != nil {
				limits = *test.limits
			}
			assertRefusalNamesItsField(t, ValidateRequest(test.request, limits), test)
		})
	}
}

func assertRefusalNamesItsField(t *testing.T, err error, test refusalCase) {
	t.Helper()
	protocolError, ok := err.(*ProtocolError)
	if !ok {
		t.Fatalf("ValidateRequest() error = %v, want a protocol error", err)
	}
	if protocolError.Code != test.code {
		t.Fatalf("code = %q, want %q", protocolError.Code, test.code)
	}
	// The client half. A body carries the message and the parameter, and
	// nothing else, so both have to name the member.
	if protocolError.Parameter != test.field {
		t.Fatalf("parameter = %q, want %q", protocolError.Parameter, test.field)
	}
	for _, want := range append([]string{`"` + test.field + `"`, test.location}, test.message...) {
		if want == "" {
			continue
		}
		if !strings.Contains(protocolError.Message, want) {
			t.Fatalf("client message %q does not carry %s", protocolError.Message, want)
		}
	}
	// The log half. The cause is what reaches ingress_request_refused.
	if protocolError.Cause == nil {
		t.Fatal("refusal carries no cause, so the log line names nothing")
	}
	if !strings.Contains(protocolError.Cause.Error(), test.field) {
		t.Fatalf("cause %q does not name %q", protocolError.Cause, test.field)
	}
	if test.location != "" && !strings.Contains(protocolError.Cause.Error(), test.location) {
		t.Fatalf("cause %q does not carry %s", protocolError.Cause, test.location)
	}
}

func envelopeRefusalCases() []refusalCase {
	longIdentifier := strings.Repeat("u", 32)
	return []refusalCase{
		{
			name:    "model missing",
			request: withRequest(func(request *Request) { request.Model = "" }),
			code:    "model_required", field: "model",
		},
		{
			name:    "model over the limit",
			request: withRequest(func(request *Request) { request.Model = strings.Repeat("m", 32) }),
			limits:  tightLimits(func(limits *Limits) { limits.ModelBytes = 8 }),
			code:    "model_limit", field: "model",
			message: []string{"32 bytes, limit 8"},
		},
		{
			name:    "end-user ID over the limit",
			request: withRequest(func(request *Request) { request.EndUserID = longIdentifier }),
			limits:  tightLimits(func(limits *Limits) { limits.IdentifierBytes = 8 }),
			code:    "end_user_id_limit", field: "end_user_id",
			message: []string{"32 bytes, limit 8"},
		},
		{
			name: "conversation state stated twice",
			request: withRequest(func(request *Request) {
				request.PreviousResponseID, request.ConversationID = "resp_1", "conv_1"
			}),
			code: "conflicting_conversation_state", field: "previous_response_id",
		},
		{
			name:    "truncation mode the contract does not name",
			request: withRequest(func(request *Request) { request.Truncation = "sometimes" }),
			code:    "invalid_truncation", field: "truncation",
		},
		{
			name: "stream options without a stream",
			request: withRequest(func(request *Request) {
				include := true
				request.StreamOptions.IncludeUsage = &include
			}),
			code: "stream_options_without_stream", field: "stream_options",
		},
		{
			name: "candidate count of zero",
			request: withRequest(func(request *Request) {
				count := int64(0)
				request.CandidateCount = &count
			}),
			code: "invalid_candidate_count", field: "candidate_count",
		},
		{
			name: "candidate count over the limit",
			request: withRequest(func(request *Request) {
				count := int64(4)
				request.CandidateCount = &count
			}),
			limits: tightLimits(func(limits *Limits) { limits.Candidates = 1 }),
			code:   "candidate_count_limit", field: "candidate_count",
			message: []string{"4, limit 1"},
		},
	}
}

func cardinalityRefusalCases() []refusalCase {
	textBlock := []Content{{Kind: ContentText, Text: "hi"}}
	schema := json.RawMessage(`{"type":"object"}`)
	return []refusalCase{
		{
			name: "more messages than the limit",
			request: withRequest(func(request *Request) {
				request.Messages = append(request.Messages, Message{Role: RoleUser, Content: textBlock})
			}),
			limits: tightLimits(func(limits *Limits) { limits.Messages = 1 }),
			code:   "messages_limit", field: "messages",
			message: []string{"2, limit 1"},
		},
		{
			name: "more instructions than the limit",
			request: withRequest(func(request *Request) {
				request.Instructions = []InstructionBlock{
					{Role: RoleSystem, Content: textBlock},
					{Role: RoleSystem, Content: textBlock},
				}
			}),
			limits: tightLimits(func(limits *Limits) { limits.Instructions = 1 }),
			code:   "instructions_limit", field: "instructions",
			message: []string{"2, limit 1"},
		},
		{
			name: "more tools than the limit",
			request: withRequest(func(request *Request) {
				request.Tools = []Tool{
					{Name: "one", InputSchema: schema},
					{Name: "two", InputSchema: schema},
				}
			}),
			limits: tightLimits(func(limits *Limits) { limits.Tools = 1 }),
			code:   "tools_limit", field: "tools",
			message: []string{"2, limit 1"},
		},
		{
			name: "more content blocks than the limit",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{
					{Kind: ContentText, Text: "one"},
					{Kind: ContentText, Text: "two"},
				}
			}),
			limits: tightLimits(func(limits *Limits) { limits.ContentBlocks = 1 }),
			code:   "content_limit", field: "messages.content",
			message: []string{"2, limit 1"},
		},
		{
			name: "more metadata entries than the limit",
			request: withRequest(func(request *Request) {
				request.Metadata = map[string]string{"a": "1", "b": "2"}
			}),
			limits: tightLimits(func(limits *Limits) { limits.MetadataEntries = 1 }),
			code:   "metadata_entries_limit", field: "metadata",
			message: []string{"2, limit 1"},
		},
		{
			name: "a metadata key over the limit",
			request: withRequest(func(request *Request) {
				request.Metadata = map[string]string{strings.Repeat("k", 8): "1"}
			}),
			limits: tightLimits(func(limits *Limits) { limits.MetadataKeyBytes = 2 }),
			code:   "metadata_field_limit", field: "metadata",
		},
		{
			name: "metadata over the total limit",
			request: withRequest(func(request *Request) {
				request.Metadata = map[string]string{"key": "value"}
			}),
			limits: tightLimits(func(limits *Limits) { limits.MetadataBytes = 2 }),
			code:   "metadata_limit", field: "metadata",
			message: []string{"8 bytes, limit 2"},
		},
	}
}

func messageRefusalCases() []refusalCase {
	toolResult := Content{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1"}}
	return []refusalCase{
		{
			name: "message ID over the limit",
			request: withRequest(func(request *Request) {
				request.Messages[0].ID = strings.Repeat("i", 32)
			}),
			limits: tightLimits(func(limits *Limits) { limits.IdentifierBytes = 8 }),
			code:   "message_id_limit", field: "messages.id", location: "message 0",
			message: []string{"32 bytes, limit 8"},
		},
		{
			name:    "role the contract does not name",
			request: withRequest(func(request *Request) { request.Messages[0].Role = "wizard" }),
			code:    "invalid_role", field: "messages.role", location: "message 0",
		},
		{
			name:    "message with no content",
			request: withRequest(func(request *Request) { request.Messages[0].Content = nil }),
			code:    "empty_message", field: "messages.content", location: "message 0",
		},
		{
			name: "tool message holding two results",
			request: withRequest(func(request *Request) {
				request.Messages[0] = Message{Role: RoleTool, Content: []Content{toolResult, toolResult}}
			}),
			code: "tool_message_cardinality", field: "messages.content", location: "message 0",
		},
		{
			name: "assistant-only content in a user message",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{toolResult}
			}),
			code: "invalid_role_content", field: "messages.content",
			location: `message 0 content block 0 of type "tool_result"`,
		},
	}
}

func instructionRefusalCases() []refusalCase {
	textBlock := []Content{{Kind: ContentText, Text: "be brief"}}
	return []refusalCase{
		{
			name: "instruction role that is not system or developer",
			request: withRequest(func(request *Request) {
				request.Instructions = []InstructionBlock{{Role: RoleUser, Content: textBlock}}
			}),
			code: "invalid_instruction_role", field: "instructions.role", location: "instruction 0",
		},
		{
			name: "instruction with no content",
			request: withRequest(func(request *Request) {
				request.Instructions = []InstructionBlock{{Role: RoleSystem}}
			}),
			code: "empty_instruction", field: "instructions.content", location: "instruction 0",
		},
		{
			name: "tool control inside an instruction",
			request: withRequest(func(request *Request) {
				request.Instructions = []InstructionBlock{{Role: RoleSystem, Content: []Content{
					{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "lookup", Arguments: "{}"}},
				}}}
			}),
			code: "invalid_instruction_content", field: "instructions.content", location: "instruction 0",
		},
	}
}

func outputFormatRefusalCases() []refusalCase {
	schema := json.RawMessage(`{"type":"object"}`)
	return []refusalCase{
		{
			name: "output format kind the contract does not name",
			request: withRequest(func(request *Request) {
				request.OutputFormat = OutputFormat{Kind: "yaml"}
			}),
			code: "invalid_output_format", field: "output_format",
		},
		{
			name: "schema members on a text output format",
			request: withRequest(func(request *Request) {
				request.OutputFormat = OutputFormat{Kind: OutputText, Name: "answer"}
			}),
			code: "invalid_output_format", field: "output_format",
		},
		{
			name: "output schema that is not JSON",
			request: withRequest(func(request *Request) {
				request.OutputFormat = OutputFormat{Kind: OutputJSONSchema, Name: "answer", Schema: json.RawMessage(`{`)}
			}),
			code: "invalid_output_schema", field: "output_format.schema",
		},
		{
			name: "output schema over the limit",
			request: withRequest(func(request *Request) {
				request.OutputFormat = OutputFormat{Kind: OutputJSONSchema, Name: "answer", Schema: schema}
			}),
			limits: tightLimits(func(limits *Limits) { limits.SchemaBytes = 4 }),
			code:   "schema_limit", field: "output_format.schema",
			message: []string{"17 bytes, limit 4"},
		},
		{
			name: "output schema with no name",
			request: withRequest(func(request *Request) {
				request.OutputFormat = OutputFormat{Kind: OutputJSONSchema, Schema: schema}
			}),
			code: "output_schema_name_required", field: "output_format.name",
		},
	}
}

func samplingRefusalCases() []refusalCase {
	outOfRange := 5.0
	negative := int64(-1)
	return []refusalCase{
		{
			name:    "temperature outside its range",
			request: withRequest(func(request *Request) { request.Sampling.Temperature = &outOfRange }),
			code:    "invalid_temperature", field: "temperature",
		},
		{
			name:    "top_p outside its range",
			request: withRequest(func(request *Request) { request.Sampling.TopP = &outOfRange }),
			code:    "invalid_top_p", field: "top_p",
		},
		{
			name:    "negative top_k",
			request: withRequest(func(request *Request) { request.Sampling.TopK = &negative }),
			code:    "invalid_top_k", field: "top_k",
		},
		{
			name:    "negative output allowance",
			request: withRequest(func(request *Request) { request.Sampling.MaxOutputTokens = &negative }),
			code:    "invalid_max_output_tokens", field: "max_output_tokens",
		},
		{
			name:    "frequency penalty outside its range",
			request: withRequest(func(request *Request) { request.Sampling.FrequencyPenalty = &outOfRange }),
			code:    "invalid_penalty", field: "frequency_penalty",
		},
		{
			name:    "presence penalty outside its range",
			request: withRequest(func(request *Request) { request.Sampling.PresencePenalty = &outOfRange }),
			code:    "invalid_penalty", field: "presence_penalty",
		},
		{
			name:    "more stop sequences than the limit",
			request: withRequest(func(request *Request) { request.Sampling.Stop = []string{"a", "b"} }),
			limits:  tightLimits(func(limits *Limits) { limits.StopSequences = 1 }),
			code:    "stop_limit", field: "stop_sequences",
			message: []string{"2, limit 1"},
		},
		{
			name:    "an empty stop sequence",
			request: withRequest(func(request *Request) { request.Sampling.Stop = []string{""} }),
			code:    "invalid_stop", field: "stop_sequences",
		},
		{
			name:    "stop sequences over the byte limit",
			request: withRequest(func(request *Request) { request.Sampling.Stop = []string{"abcd"} }),
			limits:  tightLimits(func(limits *Limits) { limits.StopBytes = 2 }),
			code:    "stop_bytes_limit", field: "stop_sequences",
			message: []string{"4 bytes, limit 2"},
		},
	}
}

func reasoningRefusalCases() []refusalCase {
	budget := int64(1024)
	zero := int64(0)
	return []refusalCase{
		{
			name:    "reasoning effort over the limit",
			request: withRequest(func(request *Request) { request.ReasoningEffort = strings.Repeat("h", 8) }),
			limits:  tightLimits(func(limits *Limits) { limits.ReasoningEffortBytes = 2 }),
			code:    "reasoning_effort_limit", field: "reasoning_effort",
			message: []string{"8 bytes, limit 2"},
		},
		{
			name:    "reasoning effort the contract does not name",
			request: withRequest(func(request *Request) { request.ReasoningEffort = "medium-rare" }),
			code:    "invalid_reasoning_effort", field: "reasoning_effort",
		},
		{
			name: "reasoning budget of zero",
			request: withRequest(func(request *Request) {
				request.ReasoningMode, request.ReasoningBudgetTokens = ReasoningModeEnabled, &zero
			}),
			code: "invalid_reasoning_budget", field: "reasoning_budget_tokens",
		},
		{
			name: "enabled reasoning with no budget",
			request: withRequest(func(request *Request) {
				request.ReasoningMode = ReasoningModeEnabled
			}),
			code: "reasoning_budget_required", field: "reasoning_budget_tokens",
		},
		{
			name: "a budget beside disabled reasoning",
			request: withRequest(func(request *Request) {
				request.ReasoningMode, request.ReasoningBudgetTokens = ReasoningModeDisabled, &budget
			}),
			code: "conflicting_reasoning_control", field: "reasoning_budget_tokens",
		},
		{
			name:    "reasoning mode the contract does not name",
			request: withRequest(func(request *Request) { request.ReasoningMode = "pondering" }),
			code:    "invalid_reasoning_mode", field: "reasoning_mode",
		},
		{
			name: "reasoning display the contract does not name",
			request: withRequest(func(request *Request) {
				request.ReasoningMode, request.ReasoningBudgetTokens = ReasoningModeEnabled, &budget
				request.ReasoningDisplay = "verbose"
			}),
			code: "invalid_reasoning_display", field: "reasoning_display",
		},
		{
			name: "reasoning display without reasoning",
			request: withRequest(func(request *Request) { request.ReasoningDisplay = "summarized" }),
			code:  "conflicting_reasoning_display", field: "reasoning_display",
		},
	}
}

func contentRefusalCases() []refusalCase {
	return []refusalCase{
		{
			name: "text block carrying a media member",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{{Kind: ContentText, Text: "hi", URL: "https://example.test/a"}}
			}),
			code: "invalid_content", field: "content.url",
			location: `message 0 content block 0 of type "text"`,
		},
		{
			name: "text over the limit",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{{Kind: ContentText, Text: strings.Repeat("t", 32)}}
			}),
			limits: tightLimits(func(limits *Limits) { limits.TextBytes = 8 }),
			code:   "text_limit", field: "content.text",
			location: `message 0 content block 0 of type "text"`,
			message:  []string{"32 bytes, limit 8"},
		},
		{
			name: "image with no source",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{{Kind: ContentImage}}
			}),
			code: "media_reference_required", field: "content.source",
			location: `message 0 content block 0 of type "image"`,
		},
		{
			name: "inline image with no media type",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{{Kind: ContentImage, Data: "aW1n"}}
			}),
			code: "media_type_required", field: "content.media_type",
			location: `message 0 content block 0 of type "image"`,
		},
		{
			name: "inline image over the data limit",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{
					{Kind: ContentImage, MediaType: "image/png", Data: strings.Repeat("a", 32)},
				}
			}),
			limits: tightLimits(func(limits *Limits) { limits.MediaDataBytes = 8 }),
			code:   "media_data_limit", field: "content.data",
			location: `message 0 content block 0 of type "image"`,
			message:  []string{"32 bytes, limit 8"},
		},
		{
			name: "cache directive the contract does not name",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content[0].Cache = &CacheDirective{Type: "permanent"}
			}),
			code: "invalid_cache_directive", field: "content.cache_control.type",
			location: `message 0 content block 0 of type "text"`,
		},
		{
			name: "cache TTL the contract does not name",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content[0].Cache = &CacheDirective{Type: "ephemeral", TTL: "7d"}
			}),
			code: "invalid_cache_directive", field: "content.cache_control.ttl",
			location: `message 0 content block 0 of type "text"`,
		},
		{
			name: "tool result with no call ID",
			request: withRequest(func(request *Request) {
				request.Messages[0] = Message{Role: RoleTool, Content: []Content{
					{Kind: ContentToolResult, ToolResult: &ToolResult{}},
				}}
			}),
			code: "invalid_tool_result", field: "content.tool_use_id",
			location: `message 0 content block 0 of type "tool_result"`,
		},
		{
			name: "tool control nested inside a tool result",
			request: withRequest(func(request *Request) {
				request.Messages[0] = Message{Role: RoleAssistant, Content: []Content{
					{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "lookup", Arguments: "{}"}},
				}}
				request.Messages = append(request.Messages, Message{Role: RoleTool, Content: []Content{
					{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1", Content: []Content{
						{Kind: ContentToolCall, ToolCall: &ToolCall{ID: "call_2", Name: "lookup", Arguments: "{}"}},
					}}},
				}})
			}),
			code: "nested_tool_control", field: "content.content",
			location: `message 1 content block 0 of type "tool_result"`,
		},
		{
			name: "content kind the contract does not name",
			request: withRequest(func(request *Request) {
				request.Messages[0].Content = []Content{{Kind: "hologram"}}
			}),
			code: "unknown_content_kind", field: "content.type",
			location: `message 0 content block 0 of type "hologram"`,
		},
	}
}

// Every refusal the request contract can still raise names a member. This is
// the sweep that keeps a new validator branch from arriving unnamed: a branch
// added without a field path fails here, not on a user's turn.
func TestNoRequestRefusalIsUnnamed(t *testing.T) {
	for _, group := range [][]refusalCase{
		envelopeRefusalCases(), cardinalityRefusalCases(), messageRefusalCases(),
		instructionRefusalCases(), outputFormatRefusalCases(), samplingRefusalCases(),
		reasoningRefusalCases(), contentRefusalCases(),
	} {
		for _, test := range group {
			limits := DefaultPolicy().Limits
			if test.limits != nil {
				limits = *test.limits
			}
			err := ValidateRequest(test.request, limits)
			protocolError, ok := err.(*ProtocolError)
			if !ok {
				t.Fatalf("%s: ValidateRequest() error = %v, want a protocol error", test.name, err)
			}
			if protocolError.Parameter == "" {
				t.Fatalf("%s: refusal %q names no member", test.name, protocolError.Code)
			}
		}
	}
}

// A refusal names the contract and never the conversation. A field path, a
// block index and a byte count are contract text; what the user wrote is not,
// and it must not leave the cell inside an error.
func TestRequestRefusalsCarryNoRequestValues(t *testing.T) {
	secret := "the-user-wrote-this"
	limits := DefaultPolicy().Limits
	limits.TextBytes = 4
	limits.MetadataValueBytes = 4
	for _, request := range []Request{
		withRequest(func(request *Request) { request.Messages[0].Content[0].Text = secret }),
		withRequest(func(request *Request) { request.Metadata = map[string]string{"note": secret} }),
		withRequest(func(request *Request) { request.Model = secret; request.Truncation = "sometimes" }),
	} {
		err := ValidateRequest(request, limits)
		if err == nil {
			t.Fatal("the request was accepted")
		}
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("refusal %q carries a request value", err)
		}
	}
}
