package protocolcodec

import (
	"bytes"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// citedHistoryRequest is an ordinary later turn of a conversation whose earlier
// answer came from a web search or an attached document. The client echoes that
// answer back in history exactly as it received it, citations included.
//
// validateAnthropicContentExtensions refuses any block carrying citations
// before it looks at the direction of travel or at the target
// (codec_anthropic_content.go:209-212, unsupported_citations), although the
// Anthropic wire struct already models the member as opaque bytes
// (codec_anthropic_messages.go:72).
const citedHistoryRequest = `{"model":"router-auto","max_tokens":32,"messages":[` +
	`{"role":"user","content":"how many are there?"},` +
	`{"role":"assistant","content":[{"type":"text","text":"The handbook says ten.",` +
	`"citations":[{"type":"char_location","cited_text":"ten","document_index":0,` +
	`"document_title":"Handbook","start_char_index":0,"end_char_index":10}]}]},` +
	`{"role":"user","content":"and how many were there last year?"}]}`

// TestRequestCitationsTravel is the ask.
//
// A request-side citations member records where an earlier answer came from. It
// does not change what the model is asked, so an Anthropic target should return
// it untouched, and a target that cannot express it should keep the cited text
// and drop the member -- appendLossy (codec_openai_chat_encode.go:67) is the
// seam that already does this for a member Chat Completions cannot carry.
func TestRequestCitationsTravel(t *testing.T) {
	engine := NewBuiltinEngine()
	t.Run("anthropic_messages_returns_them_unchanged", func(t *testing.T) {
		result, err := engine.TranslateRequest(
			llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
			[]byte(citedHistoryRequest), nil,
		)
		if err != nil {
			t.Fatalf("a turn whose history carries citations was refused: %v", err)
		}
		if !bytes.Contains(result.Body, []byte(`"cited_text":"ten"`)) {
			t.Fatalf("the re-encoded request lost its citations: %s", result.Body)
		}
	})
	t.Run("chat_keeps_the_cited_text", func(t *testing.T) {
		result, err := engine.TranslateRequest(
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
			[]byte(citedHistoryRequest), nil,
		)
		if err != nil {
			t.Fatalf("a turn whose history carries citations was refused: %v", err)
		}
		if !bytes.Contains(result.Body, []byte("The handbook says ten.")) {
			t.Fatalf("the translated turn lost the cited answer: %s", result.Body)
		}
	})
}

// TestRequestCitationsAreRefusedToday pins the present behaviour so a change to
// it shows up in the diff rather than only in the test above. The refusal is
// raised while decoding the request, so it does not depend on the target.
func TestRequestCitationsAreRefusedToday(t *testing.T) {
	engine := NewBuiltinEngine()
	targets := []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1}
	for _, target := range targets {
		t.Run(string(target), func(t *testing.T) {
			_, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, target, []byte(citedHistoryRequest), nil,
			)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "unsupported_citations" {
				t.Fatalf("TranslateRequest() error = %v, want code unsupported_citations", err)
			}
		})
	}
}
