package protocolcodec

import (
	"reflect"

	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// replayEquivalentResponsesSource is the Responses counterpart of
// replayEquivalentAnthropicSource: it returns the client's own Responses body,
// with only its model member rewritten, when that body is the canonical
// encoding spelled differently. A routed Codex turn that changed nothing but
// the model then reaches the provider as Codex wrote it, so the provider's
// prompt cache sees the bytes the client's own history built.
//
// The bodies are compared as JSON values, and only the respellings the
// encoder makes without changing meaning are allowed (responsesReplayForm).
// Every other difference sends the canonical body: whatever the router
// changed, whatever the encoder dropped (an encrypted reasoning item bound for
// another issuer, a web_search tool this target is not sent), and whatever the
// decoder repaired or normalised, since only the canonical spelling is known
// to be valid.
func replayEquivalentResponsesSource(
	canonical []byte,
	request llmprotocol.Request,
	envelope llmprotocol.Envelope,
	policy llmprotocol.Policy,
) []byte {
	if policy.SourcePreservation != llmprotocol.SourceBoundedSameFormat ||
		envelope.Format != llmprotocol.OpenAIResponsesV1 || len(envelope.Request) == 0 || envelope.ReplayDisabled {
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
	// encoder writes it out; the client's omission means the same, auto being
	// Responses' default.
	if _, stated := clientObject["tool_choice"]; !stated && sentObject["tool_choice"] == "auto" {
		delete(sentObject, "tool_choice")
	}
	sentForm, clientForm := responsesReplayForm(sentObject), responsesReplayForm(clientObject)
	if sentForm == nil || clientForm == nil {
		return canonical
	}
	withoutGeneratedItemIDs(sentForm, clientForm, len(request.Instructions))
	if !reflect.DeepEqual(sentForm, clientForm) {
		return canonical
	}
	return source
}

// responsesReplayForm rewrites the shorthands Responses allows and the encoder
// never writes, each into the spelling the encoder does write:
//
//   - a string instructions member is a leading developer message holding it,
//     which is where the decoder puts it;
//   - a string input is one user message;
//   - string message content, and a string function_call_output output, is
//     one input_text part;
//   - an assistant message's output_text part with no annotations is the
//     input_text part the encoder writes for resent assistant text.
//
// Nothing else is rewritten. A body whose instructions, input or items have
// any other shape returns nil, which sends the canonical body.
func responsesReplayForm(body map[string]any) map[string]any {
	var items []any
	switch input := body["input"].(type) {
	case nil:
	case string:
		items = []any{map[string]any{"type": "message", "role": "user", "content": input}}
	case []any:
		items = input
	default:
		return nil
	}
	switch instructions := body["instructions"].(type) {
	case nil:
	case string:
		developer := map[string]any{"type": "message", "role": "developer", "content": instructions}
		items = append([]any{developer}, items...)
		delete(body, "instructions")
	default:
		return nil
	}
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			return nil
		}
		switch item["type"] {
		case "message":
			if content, ok := item["content"].(string); ok {
				item["content"] = inputTextParts(content)
			}
			if item["role"] == "assistant" {
				resentAssistantText(item)
			}
		case "function_call_output":
			if output, ok := item["output"].(string); ok {
				item["output"] = inputTextParts(output)
			}
		}
	}
	if items != nil {
		body["input"] = items
	}
	return body
}

// resentAssistantText respells an assistant message's annotation-free
// output_text parts as the input_text parts the encoder writes. A part that
// carries annotations is left as it is, so its difference sends the
// canonical body.
func resentAssistantText(item map[string]any) {
	parts, _ := item["content"].([]any)
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok || part["type"] != "output_text" {
			continue
		}
		if annotations, stated := part["annotations"]; stated {
			if list, ok := annotations.([]any); !ok || len(list) > 0 {
				continue
			}
		}
		delete(part, "annotations")
		part["type"] = "input_text"
	}
}

// withoutGeneratedItemIDs takes the client's side of each item id the encoder
// generated rather than respelled:
//
//   - an item the client sent without an id, which the encoder gives one;
//   - the leading instruction items: the decoder keeps a system or developer
//     message, and the instructions member, as an instruction, which carries
//     no id, so the encoder writes a generated one there whatever the client
//     sent.
//
// Any other item the client gave an id must keep it.
func withoutGeneratedItemIDs(sent, client map[string]any, instructions int) {
	sentItems, _ := sent["input"].([]any)
	clientItems, _ := client["input"].([]any)
	if len(sentItems) != len(clientItems) {
		return
	}
	for index, raw := range clientItems {
		clientItem, clientOK := raw.(map[string]any)
		sentItem, sentOK := sentItems[index].(map[string]any)
		if !clientOK || !sentOK {
			continue
		}
		id, stated := clientItem["id"]
		switch {
		case !stated:
			delete(sentItem, "id")
		case index < instructions:
			// Only an id the provider can take: a null or empty one is
			// normalised away by the decoder, so its difference stays.
			if text, ok := id.(string); ok && text != "" {
				sentItem["id"] = id
			}
		}
	}
}

func inputTextParts(text string) []any {
	return []any{map[string]any{"type": "input_text", "text": text}}
}
