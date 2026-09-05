package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code echoes the citations of a web-search or document answer back in
// assistant history on every later turn, and marks a document for the
// Citations API with {"enabled": true}. Refusing either failed real turns at
// ingress on the dev cell on 2026-09-05 (routes rt_ae5e3e62-4d9 and
// rt_ffae7c99-26e), so a request block carries its citations instead.
const (
	charLocationCitations = `[{"type":"char_location","cited_text":"ten","document_index":0,` +
		`"document_title":"Handbook","start_char_index":0,"end_char_index":10}]`
	webSearchCitations = `[{"type":"web_search_result_location","url":"https://example.com/release",` +
		`"title":"Release notes","cited_text":"The release is out.","encrypted_index":"ZW5jcnlwdGVk"}]`
)

func anthropicCitedTextRequest(citations string) []byte {
	return []byte(`{"model":"client-model","max_tokens":32,"messages":[` +
		`{"role":"user","content":"question"},` +
		`{"role":"assistant","content":[{"type":"text","text":"The handbook says ten.",` +
		`"citations":` + citations + `}]}]}`)
}

func anthropicCitedDocumentRequest() []byte {
	return []byte(`{"model":"client-model","max_tokens":32,"messages":[` +
		`{"role":"user","content":[{"type":"document","source":{"type":"base64",` +
		`"media_type":"application/pdf","data":"ZG9jdW1lbnQ="},"citations":{"enabled":true}}]}]}`)
}

func routeToModel(request *llmprotocol.Request) error {
	request.Model = "routed-model"
	return nil
}

func TestAnthropicRequestCitationsAreAcceptedAtIngress(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name string
		body []byte
	}{
		{name: "char_location", body: anthropicCitedTextRequest(charLocationCitations)},
		{name: "web_search_result_location", body: anthropicCitedTextRequest(webSearchCitations)},
		{name: "empty_array", body: anthropicCitedTextRequest(`[]`)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, test.body)
			if err != nil {
				t.Fatalf("DecodeRequest() refused a cited assistant turn: %v", err)
			}
			content := request.Messages[1].Content[0]
			if content.Kind != llmprotocol.ContentText || content.Text != "The handbook says ten." {
				t.Fatalf("cited text was not kept as text: %+v", content)
			}
		})
	}
}

func TestAnthropicDocumentCitationsAreAcceptedAtIngress(t *testing.T) {
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, anthropicCitedDocumentRequest())
	if err != nil {
		t.Fatalf("DecodeRequest() refused a citable document: %v", err)
	}
	content := request.Messages[0].Content[0]
	if content.Kind != llmprotocol.ContentFile || content.Data != "ZG9jdW1lbnQ=" {
		t.Fatalf("document block was not kept: %+v", content)
	}
}

func TestAnthropicRequestCitationsSurviveMessagesEncode(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name     string
		body     []byte
		expected string
	}{
		{name: "char_location", body: anthropicCitedTextRequest(charLocationCitations), expected: charLocationCitations},
		{name: "web_search_result_location", body: anthropicCitedTextRequest(webSearchCitations), expected: webSearchCitations},
		{name: "empty_array", body: anthropicCitedTextRequest(`[]`), expected: `[]`},
		{name: "document_enabled", body: anthropicCitedDocumentRequest(), expected: `{"enabled":true}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, test.body, routeToModel,
			)
			if err != nil {
				t.Fatalf("TranslateRequest() to Messages error = %v", err)
			}
			assertSameJSON(t, encodedBlockCitations(t, result.Body), []byte(test.expected))
			if droppedCitationCount(result.Diagnostics) != 0 {
				t.Fatalf("carried citations were counted as dropped: %+v", result.Diagnostics)
			}
		})
	}
}

func TestAnthropicRequestCitationsAreDroppedAndCountedForChat(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name    string
		body    []byte
		keeping string
	}{
		{name: "char_location", body: anthropicCitedTextRequest(charLocationCitations), keeping: "The handbook says ten."},
		{name: "web_search_result_location", body: anthropicCitedTextRequest(webSearchCitations), keeping: "The handbook says ten."},
		{name: "empty_array", body: anthropicCitedTextRequest(`[]`), keeping: "The handbook says ten."},
		{name: "document_enabled", body: anthropicCitedDocumentRequest(), keeping: "ZG9jdW1lbnQ="},
	}
	for _, test := range tests {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			t.Run(test.name+"_to_"+string(target), func(t *testing.T) {
				result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, test.body, routeToModel)
				if err != nil {
					t.Fatalf("TranslateRequest() to %s error = %v", target, err)
				}
				if bytes.Contains(result.Body, []byte(`"citations"`)) {
					t.Fatalf("%s body still carries citations: %s", target, result.Body)
				}
				if !bytes.Contains(result.Body, []byte(test.keeping)) {
					t.Fatalf("%s body lost the cited content: %s", target, result.Body)
				}
				if droppedCitationCount(result.Diagnostics) != 1 {
					t.Fatalf("%s dropped citations without one count: %+v", target, result.Diagnostics)
				}
			})
		}
	}
}

// The canonical-field policy is unchanged: only citations is carried, and a
// sibling the Messages contract does not name still fails the turn.
func TestAnthropicRequestCitationsKeepCanonicalFieldPolicy(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"client-model","max_tokens":32,"messages":[` +
		`{"role":"assistant","content":[{"type":"text","text":"hi","citations":[],` +
		`"citation_style":"footnote"}]}]}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
}

// Response-side citations still fail closed. Carrying them would mean
// generating spans the Router cannot derive, so only the request side accepts.
func TestAnthropicResponseCitationsStayRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
		`"content":[{"type":"text","text":"The handbook says ten.","citations":` + charLocationCitations + `}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	_, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
}

func droppedCitationCount(diagnostics llmprotocol.Diagnostics) int {
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "content.citations" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			count++
		}
	}
	return count
}

func encodedBlockCitations(t *testing.T, body []byte) json.RawMessage {
	t.Helper()
	var wire struct {
		Messages []struct {
			Content []struct {
				Citations json.RawMessage `json:"citations"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("encoded body is not a Messages request: %v", err)
	}
	last := wire.Messages[len(wire.Messages)-1]
	return last.Content[0].Citations
}

func assertSameJSON(t *testing.T, actual, expected []byte) {
	t.Helper()
	var actualValue, expectedValue any
	if err := json.Unmarshal(actual, &actualValue); err != nil {
		t.Fatalf("citations are not JSON: %q (%v)", actual, err)
	}
	if err := json.Unmarshal(expected, &expectedValue); err != nil {
		t.Fatal(err)
	}
	actualText, _ := json.Marshal(actualValue)
	expectedText, _ := json.Marshal(expectedValue)
	if !bytes.Equal(actualText, expectedText) {
		t.Fatalf("citations = %s, want %s", actualText, expectedText)
	}
}
