package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenAI Chat Completions and Responses share the same non-2xx API error
// envelope. This is intentionally not a Responses resource whose status is
// "failed"; that object describes model-generation failure after acceptance.
type openAITransportErrorWire struct {
	Error *openAITransportErrorDetailWire `json:"error"`
}

type openAITransportErrorDetailWire struct {
	Type    string               `json:"type"`
	Code    *openAIErrorCodeWire `json:"code"`
	Message string               `json:"message"`
	Param   *string              `json:"param"`
}

// openAIErrorCodeWire accepts the code as the string the OpenAI contract
// documents and as the HTTP status integer several providers send in its
// place. Both name the same thing, and rejecting one shape loses the message
// the envelope exists to carry.
type openAIErrorCodeWire struct {
	Value string
}

func (code *openAIErrorCodeWire) UnmarshalJSON(raw []byte) error {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		code.Value = text
		return nil
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err == nil {
		if _, convErr := strconv.ParseInt(number.String(), 10, 64); convErr == nil {
			code.Value = number.String()
			return nil
		}
	}
	return fmt.Errorf("upstream error code must be a string or an integer")
}

func decodeOpenAITransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire openAITransportErrorWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	if wire.Error == nil {
		return llmprotocol.TransportError{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_error_required",
			"upstream transport error body is missing error details",
			nil,
		)
	}
	// The OpenAI contract documents a type and several providers omit it. The
	// message is the part a client acts on; the type only steers category
	// selection, which the code and the fallback already cover. Refusing the
	// envelope for a missing type would replace an actionable upstream answer
	// with a generic one.
	if err := validateTransportErrorMessage(wire.Error.Message); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	code, parameter := "", ""
	if wire.Error.Code != nil {
		code = wire.Error.Code.Value
	}
	if wire.Error.Param != nil {
		parameter = *wire.Error.Param
	}
	return llmprotocol.TransportError{Error: &llmprotocol.ProtocolError{
		Category:  decodeProviderErrorCategory(wire.Error.Type, code),
		Code:      code,
		Message:   wire.Error.Message,
		Parameter: parameter,
	}}, nil, nil
}

func encodeOpenAITransportError(transportError llmprotocol.TransportError) []byte {
	protocolError := transportError.Error
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	wire := openAITransportErrorEnvelope(protocolError)
	body, _ := marshalWire(wire)
	return body
}

func openAITransportErrorEnvelope(protocolError *llmprotocol.ProtocolError) openAITransportErrorWire {
	return openAITransportErrorWire{Error: &openAITransportErrorDetailWire{
		Type:    canonicalOpenAIErrorType(protocolError.Category),
		Code:    optionalErrorCode(protocolError.Code),
		Message: protocolError.Message,
		Param:   optionalString(protocolError.Parameter),
	}}
}

// The encoder re-emits the code as the string the OpenAI contract documents,
// whatever shape it arrived in. A client reading the envelope sees one type.
func optionalErrorCode(value string) *openAIErrorCodeWire {
	if value == "" {
		return nil
	}
	return &openAIErrorCodeWire{Value: value}
}

func (code openAIErrorCodeWire) MarshalJSON() ([]byte, error) {
	return json.Marshal(code.Value)
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func canonicalOpenAIErrorType(category llmprotocol.ErrorCategory) string {
	switch category {
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_request_error"
	case llmprotocol.ErrorAuthentication:
		return "authentication_error"
	case llmprotocol.ErrorPermission:
		return "permission_error"
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_error"
	default:
		return "server_error"
	}
}
