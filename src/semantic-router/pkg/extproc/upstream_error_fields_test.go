package extproc

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// The rejected member is named by its contract path, and nothing the client
// wrote passes the pattern: a quoted value, an identifier, a word that is not
// a request member.
func TestUpstreamErrorFieldsNamesContractPathsOnly(t *testing.T) {
	cases := []struct {
		message string
		want    []string
	}{
		{"Invalid value for 'messages[3].content[0].type': expected one of text, image_url.", []string{"messages[3].content[0].type"}},
		{"Invalid 'input[12].name': string too long. Expected a string with maximum length 64.", []string{"input[12].name"}},
		{"Missing required parameter: 'tools[2].function.parameters'.", []string{"tools[2].function.parameters"}},
		{"messages[4] and messages[4] and messages[5].tool_calls", []string{"messages[4]", "messages[5].tool_calls"}},
		{"No tool output found for function call call_AbC123xyz.", nil},
		{"Your request said 'hello world' and user.name was invalid", nil},
		{"Request contains an invalid argument.", nil},
		{"Unsupported parameter: 'max_tokens' is not supported with this model.", []string{"max_tokens"}},
	}
	for _, c := range cases {
		if got := upstreamErrorFields(c.message); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("upstreamErrorFields(%q) = %v, want %v", c.message, got, c.want)
		}
	}
	many := ""
	for i := 0; i < 10; i++ {
		many += fmt.Sprintf(" messages[%d]", i)
	}
	if got := upstreamErrorFields(many); len(got) != maxUpstreamErrorFields {
		t.Fatalf("fields = %v, want at most %d", got, maxUpstreamErrorFields)
	}
}

// The redaction line carries the named paths beside the withheld message's
// descriptor, and no field member when the message names none.
func TestRedactedUpstreamErrorLogNamesTheRejectedField(t *testing.T) {
	logs := captureLogs(t)
	message := "Invalid value for 'messages[7].content[1].type': value is not supported. See https://example.com/docs."
	_ = publicUpstreamError(&RequestContext{RequestID: "request_fields"},
		&llmprotocol.ProtocolError{Category: llmprotocol.ErrorInvalidRequest, Code: "invalid_request_error", Message: message}, 0, "stream")
	fields := findLogEvent(t, logs, "upstream_error_redacted")
	if got := fmt.Sprint(fields["upstream_error_fields"]); got != "[messages[7].content[1].type]" {
		t.Fatalf("upstream_error_fields = %v", fields["upstream_error_fields"])
	}
	logs.TakeAll()
	_ = publicUpstreamError(&RequestContext{RequestID: "request_nofields"},
		&llmprotocol.ProtocolError{Category: llmprotocol.ErrorInvalidRequest, Code: "invalid_request_error", Message: "Bad request. See https://example.com/docs."}, 0, "stream")
	fields = findLogEvent(t, logs, "upstream_error_redacted")
	if _, present := fields["upstream_error_fields"]; present {
		t.Fatalf("upstream_error_fields present for a message naming no field: %v", fields)
	}
}
