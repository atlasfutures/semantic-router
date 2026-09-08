package llmprotocol

import (
	"strings"
	"testing"
)

// validationRefusals are three refusals ValidateRequest raises, each about a
// different member of the request. Every one is built by NewError
// (validate.go:283, :572, :599), which leaves ProtocolError.Parameter unset.
var validationRefusals = []struct {
	code string
	// member is the request member a client would have to repair.
	member  string
	request func() Request
}{
	{"unknown_tool_choice", "tool_choice", func() Request {
		request := validSemanticRequest()
		request.ToolChoice = ToolChoice{Mode: ToolChoiceNamed, Name: "absent"}
		return request
	}},
	{"unknown_content_kind", "messages", func() Request {
		request := validSemanticRequest()
		request.Messages[0].Content[0] = Content{Kind: ContentKind("chart"), Text: "hello"}
		return request
	}},
	{"invalid_reasoning_scope", "messages", func() Request {
		request := validSemanticRequest()
		request.Messages[0].Content[0] = Content{Kind: ContentReasoning, Text: "hello", Reasoning: ReasoningScope("private")}
		return request
	}},
}

// TestValidationRefusalNamesTheRejectedParameter is the ask.
//
// ProtocolError.Parameter (errors.go:24) is serialised as the client-facing
// "param" key (protocolcodec/openai_transport_error.go:68), and a
// provider-originated error fills it in
// (protocolcodec/codec_openai_chat_response.go:81). A refusal the router raises
// itself should name the member too. validate.go has 93 NewError sites raising
// 74 distinct codes, and none of them sets Parameter, so a client told only the
// code cannot tell which member of the request to fix.
func TestValidationRefusalNamesTheRejectedParameter(t *testing.T) {
	for _, refusal := range validationRefusals {
		t.Run(refusal.code, func(t *testing.T) {
			err := ValidateRequest(refusal.request(), DefaultPolicy().Limits)
			requireLLMProtocolErrorCode(t, err, refusal.code)
			parameter := err.(*ProtocolError).Parameter
			if parameter == "" {
				t.Fatalf("%s refused with an empty parameter, want a path to %q", refusal.code, refusal.member)
			}
			if !strings.Contains(parameter, refusal.member) {
				t.Errorf("%s named parameter %q, want a path to %q", refusal.code, parameter, refusal.member)
			}
		})
	}
}

// TestValidationRefusalCarriesNoParameterToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
func TestValidationRefusalCarriesNoParameterToday(t *testing.T) {
	for _, refusal := range validationRefusals {
		t.Run(refusal.code, func(t *testing.T) {
			err := ValidateRequest(refusal.request(), DefaultPolicy().Limits)
			requireLLMProtocolErrorCode(t, err, refusal.code)
			if parameter := err.(*ProtocolError).Parameter; parameter != "" {
				t.Errorf("%s named parameter %q: the recorded behaviour has changed", refusal.code, parameter)
			}
		})
	}
}
