package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Messages provider reply whose text block cites -- an empty list, as
// OpenRouter Messages replies send, or a document location -- reaches every
// client through the router's response path. These failed billed turns with
// unsupported_citations (kimi-k3 on OpenRouter Messages, 2026-10-01).
func TestMessagesReplyCitationsReachEveryClient(t *testing.T) {
	router := &OpenAIRouter{}
	for name, citations := range map[string]string{
		"empty":    `[]`,
		"document": `[{"type":"char_location","cited_text":"ten","document_index":0,"document_title":"Handbook","start_char_index":0,"end_char_index":10}]`,
	} {
		reply := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
			`"content":[{"type":"text","text":"The handbook says ten.","citations":` + citations + `}],` +
			`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":5}}`)
		for _, client := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			t.Run(name+"_to_"+string(client), func(t *testing.T) {
				ctx := &RequestContext{SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1}
				semantic, err := router.decodeClientResponse(reply, ctx)
				if err != nil {
					t.Fatalf("decodeClientResponse(): %v", err)
				}
				body, err := router.encodeClientResponse(*semantic, ctx)
				if err != nil {
					t.Fatalf("encodeClientResponse(): %v", err)
				}
				if !strings.Contains(string(body), "The handbook says ten.") {
					t.Fatalf("the text did not reach the client: %s", body)
				}
				if client == llmprotocol.AnthropicMessagesV1 && name == "document" && !strings.Contains(string(body), `"char_location"`) {
					t.Fatalf("the Anthropic client lost its citation: %s", body)
				}
			})
		}
	}
}
