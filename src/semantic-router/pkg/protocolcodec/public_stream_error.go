package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// PublicStreamFilter is the public form of a same-format stream: the upstream
// frames the client receives, with what it may not see rewritten. Both Chat
// and Anthropic same-format streams pass through one (semantic-router #222).
type PublicStreamFilter interface {
	Push(chunk []byte) ([]byte, error)
	Finalize() ([]byte, error)
}

// publicChatStreamError replaces the error member of a Chat stream object
// with its public form (llmprotocol.PublicUpstreamError). Only the members
// the Chat error envelope names survive: OpenRouter's metadata member carries
// the provider's raw body. It reports whether the object had an error.
func publicChatStreamError(object map[string]json.RawMessage) (bool, error) {
	raw, exists := object["error"]
	if !exists || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return false, nil
	}
	var wire openAITransportErrorDetailWire
	if err := json.Unmarshal(raw, &wire); err != nil {
		return true, invalidProviderResponse("invalid_chat_stream_error", "Chat stream error is invalid")
	}
	code := ""
	if wire.Code != nil {
		code = wire.Code.Value
	}
	parameter := ""
	if wire.Param != nil {
		parameter = *wire.Param
	}
	public := llmprotocol.PublicUpstreamError(&llmprotocol.ProtocolError{
		Category: decodeProviderErrorCategory(wire.Type, code),
		Code:     code, Message: wire.Message, Parameter: parameter,
	}, 0)
	encoded, err := json.Marshal(openAITransportErrorEnvelope(public).Error)
	if err != nil {
		return true, err
	}
	object["error"] = encoded
	return true, nil
}

// AnthropicPublicStreamFilter passes a same-format Anthropic stream through
// frame by frame, as the upstream wrote it, except an error event, which is
// restated in its public form.
type AnthropicPublicStreamFilter struct {
	framer    sseFramer
	failure   error
	finalized bool
}

func NewAnthropicPublicStreamFilter(limit int) *AnthropicPublicStreamFilter {
	return &AnthropicPublicStreamFilter{framer: newSSEFramer(limit)}
}

func (filter *AnthropicPublicStreamFilter) Push(chunk []byte) ([]byte, error) {
	if filter.failure != nil {
		return nil, filter.failure
	}
	if filter.finalized {
		return nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already finalized", nil)
	}
	frames, err := filter.framer.Push(chunk)
	if err != nil {
		filter.failure = err
		return nil, err
	}
	return filter.filterFrames(frames)
}

func (filter *AnthropicPublicStreamFilter) Finalize() ([]byte, error) {
	if filter.finalized {
		return nil, nil
	}
	filter.finalized = true
	if filter.failure != nil {
		return nil, filter.failure
	}
	frames, err := filter.framer.Finalize()
	if err != nil {
		filter.failure = err
		return nil, err
	}
	return filter.filterFrames(frames)
}

func (filter *AnthropicPublicStreamFilter) filterFrames(frames [][]byte) ([]byte, error) {
	var output bytes.Buffer
	for _, frame := range frames {
		public, err := publicAnthropicStreamFrame(frame, filter.framer.limit)
		if err != nil {
			filter.failure = err
			return nil, err
		}
		output.Write(public)
	}
	return output.Bytes(), nil
}

func publicAnthropicStreamFrame(frame []byte, limit int) ([]byte, error) {
	parsed, err := parseSSEFrame(frame, limit)
	if err != nil {
		return nil, err
	}
	if !parsed.HasData || !bytes.Contains(parsed.Data, []byte(`"error"`)) {
		return frame, nil
	}
	var wire anthropicTransportErrorWire
	if json.Unmarshal(parsed.Data, &wire) != nil || wire.Type != "error" {
		// Not an error event: a content frame that mentions the word.
		return frame, nil
	}
	upstream := &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable}
	if wire.Error != nil {
		upstream.Category = decodeProviderErrorCategory(wire.Error.Type)
		upstream.Code, upstream.Message = wire.Error.Type, wire.Error.Message
	}
	public := llmprotocol.PublicUpstreamError(upstream, 0)
	return encodeSSE(parsed.Event, anthropicTransportErrorWire{
		Type: "error", RequestID: wire.RequestID,
		Error: &anthropicErrorWire{Type: canonicalAnthropicErrorType(public), Message: public.Message},
	})
}
