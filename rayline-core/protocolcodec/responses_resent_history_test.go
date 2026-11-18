package protocolcodec

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// responsesInputItems returns a routed Responses body's input items.
func responsesInputItems(t *testing.T, body string) []map[string]any {
	t.Helper()
	var wire struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal([]byte(body), &wire); err != nil {
		t.Fatalf("routed body is not JSON: %v\n%s", err, body)
	}
	return wire.Input
}

// itemsOfType returns the input items of one type, and for "message" of one
// role.
func itemsOfType(items []map[string]any, kind, role string) []map[string]any {
	var found []map[string]any
	for _, item := range items {
		if item["type"] == kind && (role == "" || item["role"] == role) {
			found = append(found, item)
		}
	}
	return found
}

func jsonValue(t *testing.T, raw string) any {
	t.Helper()
	var value any
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

// A Codex turn after the first resends the earlier answer as an assistant
// message of output_text parts, and each tool result as a string. A strict
// Responses server (LM Studio's, for one) refuses an assistant message holding
// input_text, and a function_call_output whose output is a list, with
// "Invalid type for 'input'". When the dispatch changes more than the model,
// the canonical encoding is sent, and it must use the shapes Codex sent.
//
// The fixture is a Codex CLI 0.167.0 turn-2 request captured against a local
// server: a shell tool call, its result, the answer, and the next question.
// Instructions, texts and the tool list are shortened and ids replaced.
func TestCodexResentHistoryKeepsTheShapesCodexSends(t *testing.T) {
	body := readCodexTurn(t, "turn2-assistant-history")
	routed := routeResponsesBody(t, body, "routed-model", func(request *llmprotocol.Request) {
		request.ReasoningEffort = "medium"
	})
	if !strings.Contains(routed, `"effort":"medium"`) {
		t.Fatalf("the canonical encoding was not sent:\n%s", routed)
	}
	items := responsesInputItems(t, routed)

	assistant := itemsOfType(items, "message", "assistant")
	if len(assistant) != 1 {
		t.Fatalf("assistant messages = %d, want 1:\n%s", len(assistant), routed)
	}
	wantContent := jsonValue(t, `[{"type":"output_text","text":"The word printed from `+"`notes.txt`"+` is \"alpha\".","annotations":[]}]`)
	if !reflect.DeepEqual(assistant[0]["content"], wantContent) {
		t.Fatalf("assistant content = %v, want %v", assistant[0]["content"], wantContent)
	}

	outputs := itemsOfType(items, "function_call_output", "")
	if len(outputs) != 1 || outputs[0]["output"] != "Process exited with code 0\nOutput:\nalpha\n" {
		t.Fatalf("function_call_output = %v, want the string Codex sent", outputs)
	}

	for _, role := range []string{"user", "developer"} {
		for _, message := range itemsOfType(items, "message", role) {
			for _, part := range message["content"].([]any) {
				if kind := part.(map[string]any)["type"]; kind != "input_text" {
					t.Fatalf("a %s message part became %v:\n%s", role, kind, routed)
				}
			}
		}
	}
}

// History translated from Messages or Chat reaches a Responses target in the
// same shapes: assistant text as output_text, a one-text tool result as a
// string.
func TestTranslatedHistoryReachesResponsesInTheShapesCodexSends(t *testing.T) {
	cases := map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: `{"model":"m","max_tokens":256,"messages":[` +
			`{"role":"user","content":"Read notes.txt."},` +
			`{"role":"assistant","content":[{"type":"text","text":"Reading it."},{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"notes.txt"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"alpha"}]},` +
			`{"role":"assistant","content":"It says alpha."},` +
			`{"role":"user","content":"Thanks."}],` +
			`"tools":[{"name":"read","input_schema":{"type":"object"}}]}`,
		llmprotocol.OpenAIChatV1: `{"model":"m","messages":[` +
			`{"role":"user","content":"Read notes.txt."},` +
			`{"role":"assistant","content":"Reading it.","tool_calls":[{"id":"toolu_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"notes.txt\"}"}}]},` +
			`{"role":"tool","tool_call_id":"toolu_1","content":"alpha"},` +
			`{"role":"assistant","content":"It says alpha."},` +
			`{"role":"user","content":"Thanks."}],` +
			`"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}]}`,
	}
	for source, body := range cases {
		t.Run(string(source), func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequestForMutation(source, []byte(body))
			if err != nil {
				t.Fatalf("DecodeRequestForMutation() error = %v", err)
			}
			result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			items := responsesInputItems(t, string(result.Body))
			var texts []string
			for _, message := range itemsOfType(items, "message", "assistant") {
				for _, raw := range message["content"].([]any) {
					part := raw.(map[string]any)
					if part["type"] != "output_text" {
						t.Fatalf("assistant part type = %v, want output_text:\n%s", part["type"], result.Body)
					}
					texts = append(texts, part["text"].(string))
				}
			}
			if got := strings.Join(texts, "|"); got != "Reading it.|It says alpha." {
				t.Fatalf("assistant texts = %q:\n%s", got, result.Body)
			}
			outputs := itemsOfType(items, "function_call_output", "")
			if len(outputs) != 1 || outputs[0]["output"] != "alpha" {
				t.Fatalf("function_call_output = %v, want the string alpha", outputs)
			}
		})
	}
}

// Each resent message stays in one content union. Text and a refusal both
// belong to the output union; an assistant message that also holds an image
// can only be written in the input union, and a tool result that is not one
// text part keeps its list.
func TestResentHistoryKeepsOneContentUnion(t *testing.T) {
	cases := []struct {
		name, input, kind, role, member, want string
	}{
		{
			name: "assistant text and refusal",
			input: `[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Partly.","annotations":[]},{"type":"refusal","refusal":"Not the rest."}]},` +
				`{"type":"message","role":"user","content":"Why?"}]`,
			kind: "message", role: "assistant", member: "content",
			want: `[{"type":"output_text","text":"Partly.","annotations":[]},{"type":"refusal","refusal":"Not the rest."}]`,
		},
		{
			name: "assistant text with an input image",
			input: `[{"type":"message","role":"assistant","content":[{"type":"input_text","text":"Here."},{"type":"input_image","image_url":"https://example.com/a.png","detail":"auto"}]},` +
				`{"type":"message","role":"user","content":"Thanks."}]`,
			kind: "message", role: "assistant", member: "content",
			want: `[{"type":"input_text","text":"Here."},{"type":"input_image","image_url":"https://example.com/a.png","detail":"auto"}]`,
		},
		{
			name: "tool result of two text parts",
			input: `[{"type":"message","role":"user","content":"Run ls."},` +
				`{"type":"function_call","call_id":"c1","name":"exec","arguments":"{}"},` +
				`{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}]`,
			kind: "function_call_output", member: "output",
			want: `[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]`,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			body := `{"model":"client-model","input":` + test.input +
				`,"tools":[{"type":"function","name":"exec","parameters":{"type":"object"},"strict":false}]}`
			routed := routeResponsesBody(t, body, "routed-model", func(request *llmprotocol.Request) {
				request.ReasoningEffort = "low"
			})
			found := itemsOfType(responsesInputItems(t, routed), test.kind, test.role)
			if len(found) != 1 {
				t.Fatalf("%s items = %d, want 1:\n%s", test.kind, len(found), routed)
			}
			if want := jsonValue(t, test.want); !reflect.DeepEqual(found[0][test.member], want) {
				t.Fatalf("%s = %v, want %v", test.member, found[0][test.member], want)
			}
		})
	}
}
