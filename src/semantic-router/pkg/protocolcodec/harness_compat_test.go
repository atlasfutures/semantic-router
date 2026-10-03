package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// router-infra#90 (C1) and #91 (C2): what third-party harnesses (opencode, pi,
// omp, Hermes, OpenClaw) need from the Chat and Responses codecs. The request
// shapes they send are in testdata/client_corpus.v1.json; the goldens 055-058
// show the translations for review. These tests pin the round trips.

func renameResponseModel(response *llmprotocol.Response) error {
	response.Model = "routed-model"
	return nil
}

// openRouterReasoningDetails is the reasoning_details array the recorded
// mimo arm returned, as OpenRouter sent it.
func openRouterReasoningDetails(t *testing.T) json.RawMessage {
	t.Helper()
	var wire struct {
		Choices []struct {
			Message struct {
				ReasoningDetails json.RawMessage `json:"reasoning_details"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(loadProviderFixture(t, openRouterResponseReasoning), &wire); err != nil || len(wire.Choices) != 1 {
		t.Fatalf("fixture: %v", err)
	}
	return wire.Choices[0].Message.ReasoningDetails
}

func assertEqualReasoningJSON(t *testing.T, label string, got, want json.RawMessage) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("%s: %v in %s", label, err, got)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("%s: %v in %s", label, err, want)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("%s:\n got %s\nwant %s", label, got, want)
	}
}

// The recorded OpenRouter response reaches a Chat client with its
// reasoning_details, which the Router used to prune.
func TestOpenRouterReasoningDetailsReachAChatClient(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		loadProviderFixture(t, openRouterResponseReasoning), renameResponseModel,
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Choices []struct {
			Message struct {
				ReasoningDetails json.RawMessage `json:"reasoning_details"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil || len(wire.Choices) != 1 {
		t.Fatalf("body %s: %v", result.Body, err)
	}
	assertEqualReasoningJSON(t, "reasoning_details", wire.Choices[0].Message.ReasoningDetails, openRouterReasoningDetails(t))
}

// A streamed turn hands a Chat client each reasoning_details fragment as it
// arrives, and the neutral response it accumulates holds them merged into
// what the buffered turn returns.
func TestOpenRouterStreamedReasoningDetailsReachAChatClient(t *testing.T) {
	body, events := runProviderStream(t, openRouterStreamReasoning, llmprotocol.OpenAIChatV1)
	if got := bytes.Count(body, []byte(`"reasoning_details"`)); got != 2 {
		t.Fatalf("a Chat client saw %d reasoning_details fragments, want the 2 the provider sent:\n%s", got, body)
	}
	accumulator := newResponseAccumulator()
	if err := accumulator.apply(events); err != nil {
		t.Fatal(err)
	}
	response, err := accumulator.response()
	if err != nil {
		t.Fatal(err)
	}
	var merged json.RawMessage
	for _, item := range response.Output {
		for _, content := range item.Content {
			if details, _ := reasoningDetailsOf(content); details != nil {
				merged = details
			}
		}
	}
	assertEqualReasoningJSON(t, "merged stream reasoning_details", merged, openRouterReasoningDetails(t))
}

func mintedEncryptedContent(t *testing.T, body []byte) []string {
	t.Helper()
	var minted []string
	for _, piece := range strings.Split(string(body), `"encrypted_content":"`)[1:] {
		value, _, found := strings.Cut(piece, `"`)
		if !found || !strings.HasPrefix(value, mintedReasoningDetailsPrefix) {
			t.Fatalf("encrypted_content %q is not a minted reasoning_details blob", value)
		}
		minted = append(minted, value)
	}
	return minted
}

// A Responses client is given the reasoning_details as encrypted_content, on
// the buffered and the streamed path, and the blob reads back into exactly
// what OpenRouter sent.
func TestResponsesClientIsGivenReasoningDetailsAsEncryptedContent(t *testing.T) {
	buffered, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		loadProviderFixture(t, openRouterResponseReasoning), renameResponseModel,
	)
	if err != nil {
		t.Fatal(err)
	}
	streamed, _ := runProviderStream(t, openRouterStreamReasoning, llmprotocol.OpenAIResponsesV1)
	for label, body := range map[string][]byte{"buffered": buffered.Body, "streamed": streamed} {
		blobs := mintedEncryptedContent(t, body)
		if len(blobs) == 0 {
			t.Fatalf("%s: no encrypted_content reached the Responses client:\n%s", label, body)
		}
		for _, blob := range blobs {
			quoted, _ := json.Marshal(blob)
			details, ok := mintedReasoningDetails(quoted)
			if !ok {
				t.Fatalf("%s: %q does not read back", label, blob)
			}
			assertEqualReasoningJSON(t, label+" round trip", details, openRouterReasoningDetails(t))
		}
	}
}

// The resend of a minted reasoning item -- in each shape the harnesses send
// it -- goes back to a Chat target as the reasoning_details OpenRouter issued,
// on the assistant message holding the tool call they belong to. A Responses
// target is never sent the blob or the id that came with it, and the item
// does not count as encrypted reasoning for the issuer record.
func TestMintedReasoningItemResendGoesBackToChatOnly(t *testing.T) {
	details := json.RawMessage(`[{"type":"reasoning.encrypted","id":"call_1","data":"gemini-thought-signature","format":"google-gemini-v1","index":0}]`)
	minted := string(mintReasoningDetails(details))
	shapes := map[string]string{
		// opencode, pi, omp: the stored item, id included.
		"with_id": `{"type":"reasoning","id":"rs_1","status":"completed","summary":[],"encrypted_content":` + minted + `}`,
		// Hermes and OpenClaw strip the id.
		"without_id": `{"type":"reasoning","summary":[{"type":"summary_text","text":"Check the tool."}],"encrypted_content":` + minted + `}`,
	}
	for name, item := range shapes {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"model":"m","store":false,"include":["reasoning.encrypted_content"],"input":[` +
				`{"role":"user","content":[{"type":"input_text","text":"weather?"}]},` + item + `,` +
				`{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{}"},` +
				`{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			if err != nil {
				t.Fatal(err)
			}
			if HoldsEncryptedReasoning(request) {
				t.Fatal("a minted item counted as provider encrypted reasoning")
			}

			// Unchanged generation: the client bytes would be replayed if allowed.
			sameFormat, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			for _, leaked := range []string{mintedReasoningDetailsPrefix, `"rs_1"`, `"encrypted_content":`} {
				if bytes.Contains(sameFormat.Body, []byte(leaked)) {
					t.Fatalf("a Responses target was sent %s:\n%s", leaked, sameFormat.Body)
				}
			}

			request.Model = "routed"
			request.Generation++
			chat, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var wire chatRequestWire
			if err := json.Unmarshal(chat.Body, &wire); err != nil {
				t.Fatal(err)
			}
			var carrier *chatMessageWire
			for index := range wire.Messages {
				if len(wire.Messages[index].ReasoningDetails) > 0 {
					carrier = &wire.Messages[index]
				}
			}
			if carrier == nil || len(carrier.ToolCalls) != 1 {
				t.Fatalf("reasoning_details are not on the tool-call message:\n%s", chat.Body)
			}
			assertEqualReasoningJSON(t, "chat reasoning_details", carrier.ReasoningDetails, details)
		})
	}
}

// A provider's own blob is still the carried, issuer-gated item it was.
func TestProviderEncryptedReasoningIsNotMistakenForMinted(t *testing.T) {
	body := []byte(`{"model":"m","store":false,"input":[{"role":"user","content":"hi"},` +
		`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAABopaque"}]}`)
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if !HoldsEncryptedReasoning(request) {
		t.Fatal("a provider blob lost its carried form")
	}
	for _, value := range []string{`"gAAAAABopaque"`, `"vsr.reasoning_details.v1.!!"`, `"vsr.reasoning_details.v1.e30"`} {
		if _, minted := mintedReasoningDetails(json.RawMessage(value)); minted {
			t.Fatalf("%s read as a minted blob", value)
		}
	}
}

// Fragments extend the item their index names; string members join, the
// rest keep their first value; a fragment with a new or no index is a new
// item.
func TestMergeReasoningDetailsFragment(t *testing.T) {
	var merged json.RawMessage
	for _, fragment := range []string{
		`[{"type":"reasoning.text","text":"Check ","format":"anthropic-claude-v1","index":0}]`,
		`[{"type":"reasoning.text","text":"the tool.","index":0}]`,
		`[{"type":"reasoning.text","signature":"sig","index":0},{"type":"reasoning.encrypted","data":"abc","id":"call_1","index":1}]`,
		`[{"type":"reasoning.summary","summary":"short"}]`,
	} {
		var err error
		if merged, err = mergeReasoningDetailsFragment(merged, json.RawMessage(fragment)); err != nil {
			t.Fatal(err)
		}
	}
	assertEqualReasoningJSON(t, "merged", merged, json.RawMessage(`[`+
		`{"type":"reasoning.text","text":"Check the tool.","format":"anthropic-claude-v1","index":0,"signature":"sig"},`+
		`{"type":"reasoning.encrypted","data":"abc","id":"call_1","index":1},`+
		`{"type":"reasoning.summary","summary":"short"}]`))
}

// prompt_cache_key and prompt_cache_retention, which pi, OpenClaw and Hermes
// send on Chat Completions when configured to, no longer refuse the turn:
// a Chat target gets them back, the others drop and count them. A value of
// the wrong type is still the client's error.
func TestChatPromptCacheMembersAreCarried(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` +
		`"prompt_cache_key":"session-1","prompt_cache_retention":"24h"}`)
	rename := func(request *llmprotocol.Request) error { request.Model = "routed"; return nil }
	chat, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, rename)
	if err != nil {
		t.Fatal(err)
	}
	for _, member := range []string{`"prompt_cache_key":"session-1"`, `"prompt_cache_retention":"24h"`} {
		if !bytes.Contains(chat.Body, []byte(member)) {
			t.Fatalf("a Chat target lost %s: %s", member, chat.Body)
		}
	}
	responses, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, body, rename)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(responses.Body, []byte("prompt_cache")) {
		t.Fatalf("a foreign target was sent a Chat carrier member: %s", responses.Body)
	}
	for _, bad := range []string{`"prompt_cache_key":7`, `"prompt_cache_retention":{}`} {
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1,
			[]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],`+bad+`}`))
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest {
			t.Fatalf("%s returned %v, want invalid_request", bad, err)
		}
	}
}

// pi long retention on Responses: prompt_cache_retention is carried like
// prompt_cache_key.
func TestResponsesPromptCacheRetentionIsCarried(t *testing.T) {
	body := []byte(`{"model":"m","input":"hi","prompt_cache_key":"s","prompt_cache_retention":"24h"}`)
	result, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, body,
		func(request *llmprotocol.Request) error { request.Model = "routed"; return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(result.Body, []byte(`"prompt_cache_retention":"24h"`)) {
		t.Fatalf("prompt_cache_retention was not carried: %s", result.Body)
	}
}

// A Chat client's OpenRouter reasoning object reaches the neutral controls,
// so a bounded Chat target is sent one control and not effort beside a bound.
func TestChatReasoningObjectIsOneControl(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hi"}],` +
		`"reasoning":{"effort":"high"}}`)
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if request.ReasoningEffort != "high" {
		t.Fatalf("reasoning.effort = %q, want high", request.ReasoningEffort)
	}
	request.Generation++
	encoded, err := NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if _, both := wire["reasoning_effort"]; both && bytes.Contains(wire["reasoning"], []byte("max_tokens")) {
		t.Fatalf("effort travelled beside a bound: %s", encoded.Body)
	}
	if bytes.Contains(wire["reasoning"], []byte("effort")) && bytes.Contains(wire["reasoning"], []byte("max_tokens")) {
		t.Fatalf("effort travelled beside a bound: %s", encoded.Body)
	}
}
