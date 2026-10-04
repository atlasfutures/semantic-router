package llmprotocol

import "testing"

func incompleteToolCallResponse(stop StopReason) Response {
	return Response{
		Generation: 1, ID: "response", StopReason: stop,
		Output: []OutputItem{
			{ID: "text", Role: RoleAssistant, Content: []Content{{Kind: ContentText, Text: "running it"}}},
			{ID: "call", Role: RoleAssistant, Content: []Content{{
				Kind: ContentToolCall,
				ToolCall: &ToolCall{
					ID: "toolu_1", Name: "bash", Arguments: `{"command": "cat > /tmp/p.js`, Incomplete: true,
				},
			}}},
		},
		Usage: Usage{State: UsageUnavailable},
	}
}

func TestAnIncompleteToolCallEndsAMaxTokensResponse(t *testing.T) {
	if err := ValidateResponse(incompleteToolCallResponse(StopMaxTokens), DefaultPolicy().Limits); err != nil {
		t.Fatalf("a max_tokens response ending in a cut tool call was rejected: %v", err)
	}
}

func TestAnIncompleteToolCallUnderAnotherStopIsMalformed(t *testing.T) {
	for _, stop := range []StopReason{StopEndTurn, StopToolCall, StopUnknown} {
		err := ValidateResponse(incompleteToolCallResponse(stop), DefaultPolicy().Limits)
		requireLLMProtocolErrorCode(t, err, "invalid_incomplete_tool_call")
	}
}

func TestAnIncompleteToolCallMustBeTheLastContent(t *testing.T) {
	response := incompleteToolCallResponse(StopMaxTokens)
	response.Output = append(response.Output, OutputItem{
		ID: "after", Role: RoleAssistant, Content: []Content{{Kind: ContentText, Text: "more"}},
	})
	requireLLMProtocolErrorCode(t, ValidateResponse(response, DefaultPolicy().Limits), "invalid_incomplete_tool_call")
}

func TestPartialArgumentsWithoutTheIncompleteMarkStayInvalid(t *testing.T) {
	response := incompleteToolCallResponse(StopMaxTokens)
	response.Output[1].Content[0].ToolCall.Incomplete = false
	requireLLMProtocolErrorCode(t, ValidateResponse(response, DefaultPolicy().Limits), "invalid_tool_call")
}

// An incomplete call's arguments are a whole object or the start of one. A
// complete value that is not an object, or text that never opened an object,
// is a malformed reply, not a cut, even when the call is marked incomplete.
func TestAnIncompleteToolCallStillNeedsAnObjectOrItsStart(t *testing.T) {
	for _, arguments := range []string{`{}`, `{"command": "ls`, ` {"a":`} {
		response := incompleteToolCallResponse(StopMaxTokens)
		response.Output[1].Content[0].ToolCall.Arguments = arguments
		if err := ValidateResponse(response, DefaultPolicy().Limits); err != nil {
			t.Fatalf("arguments %q were refused: %v", arguments, err)
		}
	}
	for _, arguments := range []string{`[]`, `true`, `"text"`, `run ls`, `[1, 2`, ``} {
		response := incompleteToolCallResponse(StopMaxTokens)
		response.Output[1].Content[0].ToolCall.Arguments = arguments
		requireLLMProtocolErrorCode(t, ValidateResponse(response, DefaultPolicy().Limits), "invalid_tool_call")
	}
}
