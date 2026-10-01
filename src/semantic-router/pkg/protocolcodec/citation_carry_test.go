package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
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

// A sibling the Messages contract does not name is carried now rather than
// refused: accept-by-default made the block carrier reach every depth. The
// spelling rule survives it. A member that differs from one the contract does
// name only in case is not a new member but a second spelling of an old one,
// and Go would decode it into that member without saying so.
func TestAnthropicRequestCarriesUnknownSiblingAndRefusesCaseFolded(t *testing.T) {
	engine := NewBuiltinEngine()
	carried := []byte(`{"model":"client-model","max_tokens":32,"messages":[` +
		`{"role":"assistant","content":[{"type":"text","text":"hi","citations":[],` +
		`"citation_style":"footnote"}]}]}`)
	if _, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, carried); err != nil {
		t.Fatalf("an unknown block sibling was refused: %v", err)
	}
	folded := []byte(`{"model":"client-model","max_tokens":32,"messages":[` +
		`{"role":"assistant","content":[{"type":"text","text":"hi",` +
		`"CACHE_CONTROL":{"type":"ephemeral"}}]}]}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, folded)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
}

// A provider text block's citations of any kind are carried, never refused:
// refusing them failed billed turns (kimi-k3 on OpenRouter Messages). An
// Anthropic client gets them back as sent; a client of another format gets the
// text, with a URL citation for each citation that names a URL.
func TestAnthropicResponseCitationsOfAnyKindAreCarried(t *testing.T) {
	mixed := `[` + strings.TrimSuffix(strings.TrimPrefix(charLocationCitations, "["), "]") + `,` +
		strings.TrimSuffix(strings.TrimPrefix(webSearchCitations, "["), "]") + `]`
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
		`"content":[{"type":"text","text":"first"},` +
		`{"type":"text","text":"The handbook says ten.","citations":` + mixed + `}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

	anthropic := string(translateAnthropicResponse(t, body, llmprotocol.AnthropicMessagesV1))
	for _, want := range []string{`"char_location"`, `"document_title":"Handbook"`, `"web_search_result_location"`} {
		if !strings.Contains(anthropic, want) {
			t.Fatalf("the Anthropic client lost %s: %s", want, anthropic)
		}
	}
	chat := string(translateAnthropicResponse(t, body, llmprotocol.OpenAIChatV1))
	if !strings.Contains(chat, "The handbook says ten.") || !strings.Contains(chat, "https://example.com/release") {
		t.Fatalf("the Chat client lost the text or the URL citation: %s", chat)
	}
	if strings.Contains(chat, "char_location") || strings.Contains(chat, `"url":""`) {
		t.Fatalf("a citation without a URL leaked to Chat: %s", chat)
	}
	documentOnly := strings.Replace(body, mixed, charLocationCitations, 1)
	responses := string(translateAnthropicResponse(t, documentOnly, llmprotocol.OpenAIResponsesV1))
	if !strings.Contains(responses, "The handbook says ten.") || strings.Contains(responses, "url_citation") {
		t.Fatalf("a document citation: %s", responses)
	}
}

// Streamed, a citation of any kind is held to its block's end and carried.
func TestStreamedDocumentCitationIsCarried(t *testing.T) {
	stream := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The handbook says ten."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + strings.TrimSuffix(strings.TrimPrefix(charLocationCitations, "["), "]") + `}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`
	if anthropic := string(runAnthropicStream(t, stream, llmprotocol.AnthropicMessagesV1)); !strings.Contains(anthropic, `"char_location"`) {
		t.Fatalf("the Anthropic client lost the citation: %s", anthropic)
	}
	if chat := string(runAnthropicStream(t, stream, llmprotocol.OpenAIChatV1)); !strings.Contains(chat, "The handbook says ten.") {
		t.Fatalf("the Chat client lost the text: %s", chat)
	}
}

// Every refusal raised for a content block names the block: the unsupported
// members and the union rules alike. The refused body is the user's
// conversation and is never stored, so a refusal that gives only a feature
// name leaves nothing to search.
func TestAnthropicBlockRefusalsNameTheBlock(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name     string
		block    string
		category llmprotocol.ErrorCategory
		code     string
		at       string
		field    string
	}{
		{
			name:     "missing_required_member",
			block:    `{"type":"text"}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_content_variant",
			at:       `content block 1 of type "text"`,
			field:    `"content.text"`,
		},
		{
			name:     "member_of_another_variant",
			block:    `{"type":"text","text":"hi","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_content_variant",
			at:       `content block 1 of type "text"`,
			field:    `"content.source"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"model":"client-model","max_tokens":32,"messages":[{"role":"user",` +
				`"content":[{"type":"text","text":"look"},` + test.block + `]}]}`)
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			assertProtocolError(t, err, test.category, test.code)
			assertRefusalNamesBlock(t, err, test.at, test.field)
		})
	}
}

// assertRefusalNamesBlock reads the cause, because the cause is the only part
// of a refusal that reaches the ingress_request_refused line.
func assertRefusalNamesBlock(t *testing.T, err error, location, field string) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Cause == nil {
		t.Fatalf("refusal carries no cause: %v", err)
	}
	detail := protocolError.Cause.Error()
	if !strings.Contains(detail, location) || !strings.Contains(detail, field) {
		t.Fatalf("cause %q does not name %s and %s", detail, location, field)
	}
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
	// Earlier messages may keep the client's string content, so only the
	// last message, which carries the citations, is read as blocks.
	var wire struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("encoded body is not a Messages request: %v", err)
	}
	var blocks []struct {
		Citations json.RawMessage `json:"citations"`
	}
	if err := json.Unmarshal(wire.Messages[len(wire.Messages)-1].Content, &blocks); err != nil {
		t.Fatalf("the cited message's content is not blocks: %v", err)
	}
	return blocks[0].Citations
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

// A text block stating an empty or null citations list -- as OpenRouter
// Messages replies do -- decodes for every client; it used to fail the turn.
func TestAnthropicResponseEmptyCitationsAreNotRefused(t *testing.T) {
	for _, citations := range []string{`[]`, `null`} {
		body := `{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
			`"content":[{"type":"text","text":"done","citations":` + citations + `}],` +
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
		for _, client := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			if out := string(translateAnthropicResponse(t, body, client)); !strings.Contains(out, "done") {
				t.Fatalf("citations %s to %s: %s", citations, client, out)
			}
		}
	}
}

// A streamed citation must be an object stating its type; anything else is a
// malformed provider stream, not a citation to carry.
func TestStreamedCitationMustBeATypedObject(t *testing.T) {
	for _, citation := range []string{`null`, `"x"`, `1`, `[]`, `{}`, `{"type":7}`, `{"type":""}`} {
		stream := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + citation + `}}

`
		s, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, err = s.Push([]byte(stream))
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_delta")
	}
}

// A document citation a Chat or Responses client cannot carry is reported as
// dropped, while the text is delivered.
func TestUncarriedCitationIsReported(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
		`"content":[{"type":"text","text":"The handbook says ten.","citations":` + charLocationCitations + `}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	engine := NewBuiltinEngine()
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		response, envelope, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		response.Generation++
		encoded, err := engine.EncodeResponse(client, response, envelope)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, diagnostic := range encoded.Diagnostics {
			found = found || (diagnostic.Field == "content.citations" && diagnostic.Action == llmprotocol.DiagnosticDropped)
		}
		if !found || !strings.Contains(string(encoded.Body), "The handbook says ten.") {
			t.Fatalf("%s: diagnostics %+v, body %s", client, encoded.Diagnostics, encoded.Body)
		}
	}
	// A web search citation reaches them, so nothing is reported.
	web := strings.Replace(body, charLocationCitations, webSearchCitations, 1)
	response, envelope, _, _ := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(web))
	response.Generation++
	encoded, _ := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, envelope)
	for _, diagnostic := range encoded.Diagnostics {
		if diagnostic.Field == "content.citations" {
			t.Fatalf("a carried web citation was reported: %+v", diagnostic)
		}
	}
}

// A buffered provider citations member must be null or a list of typed
// objects, as a streamed citation must; any type name is carried.
func TestBufferedCitationsMustBeTypedObjects(t *testing.T) {
	engine := NewBuiltinEngine()
	for citations, valid := range map[string]bool{
		`null`: true, `[]`: true, `[{"type":"future_location","x":1}]`: true,
		`[{"url":"https://example.com"}]`: false, `[{"type":7}]`: false, `[null]`: false, `["x"]`: false, `{}`: false, `"x"`: false,
	} {
		body := `{"id":"msg_1","type":"message","role":"assistant","model":"source-model",` +
			`"content":[{"type":"text","text":"done","citations":` + citations + `}],` +
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
		_, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
		if valid && err != nil {
			t.Errorf("citations %s refused: %v", citations, err)
		}
		if !valid {
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_citations")
		}
	}
}

// Streamed to a Chat or Responses client, a citation it cannot carry is
// reported as dropped, as it is buffered.
func TestStreamedUncarriedCitationIsReported(t *testing.T) {
	stream := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The handbook says ten."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + strings.TrimSuffix(strings.TrimPrefix(charLocationCitations, "["), "]") + `}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		s, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, client, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _, pushed, err := s.Push([]byte(stream))
		if err != nil {
			t.Fatal(err)
		}
		_, _, final, err := s.Finalize(nil)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, diagnostic := range append(pushed, final...) {
			found = found || (diagnostic.Field == "content.citations" && diagnostic.Action == llmprotocol.DiagnosticDropped)
		}
		if !found {
			t.Fatalf("%s: no dropped content.citations diagnostic in %+v", client, append(pushed, final...))
		}
	}
}
