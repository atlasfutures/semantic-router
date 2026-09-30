package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// replayEquivalentAnthropicSource returns the client's own Messages body, with
// only its model member rewritten, when that body says exactly what the
// canonical encoding says: both decode to the same request. A routed turn
// that changed nothing but the model then reaches the provider as the client
// wrote it -- string content stays a string, members keep their order -- so
// the provider's prompt cache sees the bytes the client's own history built.
//
// The comparison is of the two decodings, not of the request the router
// holds, so whatever the encoder drops or rewrites (a hosted tool no model
// admitted, a member carried for another format) makes them differ and the
// canonical body is sent instead. So does any other change the router made.
func replayEquivalentAnthropicSource(
	canonical []byte,
	request llmprotocol.Request,
	envelope llmprotocol.Envelope,
	policy llmprotocol.Policy,
) []byte {
	if policy.SourcePreservation != llmprotocol.SourceBoundedSameFormat ||
		envelope.Format != llmprotocol.AnthropicMessagesV1 || len(envelope.Request) == 0 {
		return canonical
	}
	codec := AnthropicMessagesCodec{}
	sent, _, _, err := codec.DecodeRequest(canonical, policy)
	if err != nil {
		return canonical
	}
	source, err := sjson.SetBytes(append([]byte(nil), envelope.Request...), "model", request.Model)
	if err != nil {
		return canonical
	}
	client, _, _, err := codec.DecodeRequest(source, policy)
	if err != nil {
		return canonical
	}
	// The engine makes documented defaults explicit after every decode (an
	// unstated tool choice is auto), and the encoder writes them out; the
	// client's omission means the same.
	applyRequestSemanticDefaults(&sent)
	applyRequestSemanticDefaults(&client)
	// The decodings are compared as JSON: the request holds no unexported
	// state, and marshalling normalises how raw members (tool schemas) are
	// escaped, which the encoder and the client may spell differently.
	sentJSON, sentErr := json.Marshal(sent)
	clientJSON, clientErr := json.Marshal(client)
	if sentErr != nil || clientErr != nil || !bytes.Equal(sentJSON, clientJSON) {
		return canonical
	}
	// Roles are the one wire value the decoder normalises rather than reads:
	// it folds case and space, and maps tool to user and developer to system,
	// because Messages names only user, assistant and system. Two bodies that
	// decode alike can still spell a role differently, and only the canonical
	// spelling is valid on the wire.
	if !sameAnthropicWireRoles(canonical, source) || holdsJSONNull(source) {
		return canonical
	}
	return source
}

// holdsJSONNull reports whether a body states any member as null. The decoder
// reads an explicit null as the member's zero value and the encoder omits it,
// so a null the provider may refuse (stop_sequences, metadata) decodes like an
// absent member; such a body is encoded, never replayed.
func holdsJSONNull(body []byte) bool {
	decoder := json.NewDecoder(bytes.NewReader(body))
	for {
		token, err := decoder.Token()
		if err != nil {
			// io.EOF ends a body with no null; anything else is not replayed.
			return !errors.Is(err, io.EOF)
		}
		if token == nil {
			return true
		}
	}
}

func sameAnthropicWireRoles(canonical, source []byte) bool {
	var sent, client struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if json.Unmarshal(canonical, &sent) != nil || json.Unmarshal(source, &client) != nil ||
		len(sent.Messages) != len(client.Messages) {
		return false
	}
	for index := range sent.Messages {
		if sent.Messages[index].Role != client.Messages[index].Role {
			return false
		}
	}
	return true
}
