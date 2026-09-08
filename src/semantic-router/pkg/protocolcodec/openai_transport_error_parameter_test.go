package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestValidationRefusalShipsANullParamToday pins what a client actually reads
// when the router refuses its request. openai_transport_error.go:68 puts
// ProtocolError.Parameter in the "param" key, and the field has no omitempty
// (openai_transport_error.go:16), so a refusal that leaves Parameter unset --
// which is every refusal ValidateRequest raises -- ships "param":null rather
// than omitting the key.
func TestValidationRefusalShipsANullParamToday(t *testing.T) {
	refusal := llmprotocol.NewError(
		llmprotocol.ErrorInvalidRequest,
		"unknown_tool_choice",
		"named tool choice does not reference a declared tool",
		nil,
	)
	body := string(OpenAIChatCodec{}.EncodeTransportError(llmprotocol.TransportError{Error: refusal}))
	if !strings.Contains(body, `"param":null`) {
		t.Errorf("error body %s: the recorded behaviour has changed", body)
	}
}
