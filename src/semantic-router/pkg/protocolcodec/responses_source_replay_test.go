package protocolcodec

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// routeResponsesBody decodes a client body for mutation, routes it to model,
// lets mutate change the request the way dispatch does, and encodes it for
// Responses.
func routeResponsesBody(t *testing.T, body, model string, mutate func(*llmprotocol.Request)) string {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeRequestForMutation() error = %v", err)
	}
	request.Model = model
	if mutate != nil {
		mutate(&request)
	}
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	return string(result.Body)
}

func withModel(body, from, to string) string {
	return strings.Replace(body, `"model":"`+from+`"`, `"model":"`+to+`"`, 1)
}

// A routed Responses turn that changes nothing but the model reaches the
// provider as the client's bytes with only the model id rewritten, whichever
// shorthand the client used.
func TestResponsesRouteThatOnlyChangesTheModelSendsTheClientBytes(t *testing.T) {
	cases := map[string]string{
		"instructions and string content": `{"model":"client-model","instructions":"You are Codex.",` +
			`"input":[{"type":"message","role":"user","content":"List the files."}],"stream":true}`,
		"resent assistant output_text": `{"model":"client-model","input":[{"type":"message","role":"user","content":"List the files."},` +
			`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done.","annotations":[]}]},` +
			`{"type":"message","role":"user","content":"Thanks."}]}`,
		"string input": `{"model":"client-model","input":"List the files.","store":false}`,
		"string tool output": `{"model":"client-model","input":[{"type":"message","role":"user","content":"Run ls."},` +
			`{"type":"function_call","call_id":"c1","name":"exec","arguments":"{\"cmd\":\"ls\"}"},` +
			`{"type":"function_call_output","call_id":"c1","output":"README.md\n"}],` +
			`"tools":[{"type":"function","name":"exec","parameters":{"type":"object"},"strict":false}]}`,
		"a developer message's own id": `{"model":"client-model","input":[{"type":"message","id":"msg_dev","role":"developer",` +
			`"content":[{"type":"input_text","text":"Sandbox: read-only."}]},{"type":"message","role":"user","content":"Hi"}]}`,
		"non-ASCII and markup": `{"model":"client-model","input":[{"type":"message","role":"user","content":"Überprüfe <b>die</b> Datei — 日本語 ✓ & more"}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if got, want := routeResponsesBody(t, body, "routed-model", nil), withModel(body, "client-model", "routed-model"); got != want {
				t.Fatalf("routed body =\n%s\nwant the client bytes with the model rewritten:\n%s", got, want)
			}
		})
	}
}

// Pathfinder's Responses noop_fidelity golden: with no lever on the request,
// each call reaches the provider as the client bytes with the case's wire
// model.
func TestResponsesNoopFidelityGoldenIsByteEqual(t *testing.T) {
	dir := "../selection/raylinearc/thinkingcontrol/testdata/golden/responses/noop_fidelity"
	raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		WireModel string `json:"wire_model"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil || golden.WireModel == "" {
		t.Fatalf("case.json wire_model: %v", err)
	}
	for _, call := range []string{"00", "01"} {
		client, err := os.ReadFile(filepath.Join(dir, "client-"+call+".json"))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join(dir, "provider-"+call+".json"))
		if err != nil {
			t.Fatal(err)
		}
		if got := routeResponsesBody(t, string(client), golden.WireModel, nil); got != string(want) {
			t.Fatalf("call %s =\n%s\nwant\n%s", call, got, want)
		}
	}
}

// The recorded Codex turns pass through byte for byte when the dispatch
// drops nothing: web_search is admitted and the encrypted reasoning goes back
// to its issuer.
func TestResponsesCodexTurnsReplayWhenNothingIsDropped(t *testing.T) {
	for _, turn := range []string{"turn1", "turn2"} {
		t.Run(turn, func(t *testing.T) {
			body := readCodexTurn(t, turn)
			got := routeResponsesBody(t, body, "routed-model", forwardEverything)
			if want := codexRouted(t, body); got != want {
				t.Fatalf("routed body differs from the client bytes:\n%s", got)
			}
		})
	}
}

// Whatever the dispatch changed or dropped, and whatever the decoder
// normalised, sends the canonical encoding instead of the client bytes.
func TestResponsesRouteNeverReplaysWhatChanged(t *testing.T) {
	assertCanonical := func(t *testing.T, got, client string) {
		t.Helper()
		if got == client || strings.Contains(got, `"instructions"`) {
			t.Fatalf("the client bytes were replayed:\n%s", got)
		}
	}
	t.Run("a web_search tool this target is not sent", func(t *testing.T) {
		body := readCodexTurn(t, "turn1")
		got := routeResponsesBody(t, body, "routed-model", func(request *llmprotocol.Request) {
			request.ForwardsEncryptedReasoning = true
		})
		assertCanonical(t, got, codexRouted(t, body))
		if strings.Contains(got, `"web_search"`) {
			t.Fatalf("the unadmitted web_search tool travelled:\n%s", got)
		}
	})
	t.Run("encrypted reasoning bound for another issuer", func(t *testing.T) {
		body := readCodexTurn(t, "turn2")
		got := routeResponsesBody(t, body, "routed-model", func(request *llmprotocol.Request) {
			request.HostedTools = []string{"web_search"}
		})
		assertCanonical(t, got, codexRouted(t, body))
		// The include list names reasoning.encrypted_content; the blob is
		// what must not travel.
		if strings.Contains(got, "gAAAAB-fake-encrypted-content-for-capture") {
			t.Fatalf("encrypted reasoning travelled to a non-issuer:\n%s", got)
		}
	})
	t.Run("a route that changed more than the model", func(t *testing.T) {
		body := `{"model":"client-model","instructions":"You are Codex.","input":"List the files."}`
		got := routeResponsesBody(t, body, "routed-model", func(request *llmprotocol.Request) {
			request.ReasoningEffort = "high"
		})
		assertCanonical(t, got, withModel(body, "client-model", "routed-model"))
		if !strings.Contains(got, `"reasoning":{"effort":"high"}`) {
			t.Fatalf("the routed effort is missing:\n%s", got)
		}
	})
	t.Run("an explicit null the decoder reads as absent", func(t *testing.T) {
		body := `{"model":"client-model","instructions":null,"input":"List the files."}`
		got := routeResponsesBody(t, body, "routed-model", nil)
		if strings.Contains(got, `"instructions"`) {
			t.Fatalf("the explicit null travelled:\n%s", got)
		}
	})
	for name, id := range map[string]string{"a null": `null`, "an empty": `""`} {
		t.Run(name+" instruction item id the decoder normalises away", func(t *testing.T) {
			body := `{"model":"client-model","input":[{"type":"message","id":` + id + `,"role":"developer",` +
				`"content":[{"type":"input_text","text":"Sandbox: read-only."}]},{"type":"message","role":"user","content":"Hi"}]}`
			got := routeResponsesBody(t, body, "routed-model", nil)
			if strings.Contains(got, `"id":`+id) {
				t.Fatalf("the normalised-away id travelled:\n%s", got)
			}
		})
	}
	t.Run("a developer message the decoder moves to the front", func(t *testing.T) {
		body := `{"model":"client-model","input":[{"type":"message","role":"user","content":"Hi"},` +
			`{"type":"message","role":"developer","content":"Sandbox: read-only."}]}`
		got := routeResponsesBody(t, body, "routed-model", nil)
		if got == withModel(body, "client-model", "routed-model") {
			t.Fatalf("a reordered body replayed the client bytes:\n%s", got)
		}
	})
	t.Run("assistant text whose annotations the encoder drops", func(t *testing.T) {
		body := `{"model":"client-model","input":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"See it.",` +
			`"annotations":[{"type":"url_citation","url":"https://example.com","title":"x","start_index":0,"end_index":3}]}]},` +
			`{"type":"message","role":"user","content":"Thanks."}]}`
		got := routeResponsesBody(t, body, "routed-model", nil)
		if strings.Contains(got, "url_citation") {
			t.Fatalf("the dropped annotation travelled:\n%s", got)
		}
	})
}

func forwardEverything(request *llmprotocol.Request) {
	request.HostedTools = []string{"web_search"}
	request.ForwardsEncryptedReasoning = true
}

// codexRouted is a recorded Codex turn with only its model id rewritten; the
// captures are pretty-printed.
func codexRouted(t *testing.T, body string) string {
	t.Helper()
	const recorded = `"model": "gpt-5-codex"`
	if strings.Count(body, recorded) != 1 {
		t.Fatalf("the capture no longer names its model as %s", recorded)
	}
	return strings.Replace(body, recorded, `"model": "routed-model"`, 1)
}

func readCodexTurn(t *testing.T, turn string) string {
	t.Helper()
	raw, err := os.ReadFile("testdata/codex/" + turn + "-request.json")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}
