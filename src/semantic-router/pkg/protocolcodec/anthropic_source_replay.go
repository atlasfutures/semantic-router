package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"

	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// replayEquivalentAnthropicSource returns the client's own Messages body, with
// only its model member rewritten, when that body is the canonical encoding
// spelled differently. A routed turn that changed nothing but the model then
// reaches the provider as the client wrote it -- string content stays a
// string, members keep their order -- so the provider's prompt cache sees the
// bytes the client's own history built.
//
// The two bodies are compared as JSON values, which already ignores member
// order, spacing and escaping. On top of that only the rewrites the encoder
// makes without changing meaning are allowed (anthropicReplayForm). Every
// other difference sends the canonical body: whatever the router changed,
// whatever the encoder dropped, and whatever the decoder repaired or
// normalised on the way in (a role spelling, an explicit null, a synthesized
// input_schema), since only the canonical spelling is known to be valid.
func replayEquivalentAnthropicSource(
	canonical []byte,
	request llmprotocol.Request,
	envelope llmprotocol.Envelope,
	policy llmprotocol.Policy,
) []byte {
	if policy.SourcePreservation != llmprotocol.SourceBoundedSameFormat ||
		envelope.Format != llmprotocol.AnthropicMessagesV1 || len(envelope.Request) == 0 || envelope.ReplayDisabled {
		return canonical
	}
	source, err := sjson.SetBytes(append([]byte(nil), envelope.Request...), "model", request.Model)
	if err != nil {
		return canonical
	}
	sent, sentErr := decodeJSONValue(canonical)
	client, clientErr := decodeJSONValue(source)
	if sentErr != nil || clientErr != nil {
		return canonical
	}
	sentObject, sentOK := sent.(map[string]any)
	clientObject, clientOK := client.(map[string]any)
	if !sentOK || !clientOK {
		return canonical
	}
	// The engine makes an unstated tool choice explicit (auto), and the
	// encoder writes it out; the client's omission means the same.
	if _, stated := clientObject["tool_choice"]; !stated &&
		reflect.DeepEqual(sentObject["tool_choice"], map[string]any{"type": "auto"}) {
		delete(sentObject, "tool_choice")
	}
	if !reflect.DeepEqual(anthropicReplayForm(sentObject), anthropicReplayForm(clientObject)) {
		return canonical
	}
	return source
}

// anthropicReplayForm rewrites the one shorthand Messages allows and the
// encoder never writes: string content, in a message, the system prompt or a
// tool result, is one text block. Nothing else is rewritten.
func anthropicReplayForm(body map[string]any) map[string]any {
	if system, ok := body["system"].(string); ok {
		body["system"] = textBlocks(system)
	}
	messages, _ := body["messages"].([]any)
	for _, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if content, ok := message["content"].(string); ok {
			message["content"] = textBlocks(content)
		}
		blocks, _ := message["content"].([]any)
		for _, rawBlock := range blocks {
			block, ok := rawBlock.(map[string]any)
			if !ok || block["type"] != "tool_result" {
				continue
			}
			if content, ok := block["content"].(string); ok {
				block["content"] = textBlocks(content)
			}
		}
	}
	return body
}

func textBlocks(text string) []any {
	return []any{map[string]any{"type": "text", "text": text}}
}

// decodeJSONValue reads one JSON document keeping numbers as written, so a
// number the encoder respells counts as a difference.
func decodeJSONValue(body []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("trailing data")
	}
	return value, nil
}
