package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const anthropicReplayPrefix = `{"model":"client-model","max_tokens":32000,"system":[{"type":"text","text":"You are Claude Code."}],"messages":[`

// routeAnthropicBody decodes a client body for mutation, lets mutate change
// the request the way routing does, and encodes it for Messages.
func routeAnthropicBody(t *testing.T, body string, mutate func(*llmprotocol.Request)) string {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeRequestForMutation() error = %v", err)
	}
	request.Model = "routed-model"
	if mutate != nil {
		mutate(&request)
	}
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	return string(result.Body)
}

// A routed turn that changes nothing but the model reaches the provider as
// the client's bytes with only the model id rewritten: string content stays a
// string, members keep the client's order, and nothing is escaped anew.
func TestAnthropicRouteThatOnlyChangesTheModelSendsTheClientBytes(t *testing.T) {
	cases := map[string]string{
		"string content": `{"role":"user","content":"List the files."},` +
			`{"role":"assistant","content":[{"type":"text","text":"Done."}]},{"role":"user","content":"Thanks."}]`,
		"cache_control tail": `{"role":"user","content":[{"type":"text","text":"List the files.","cache_control":{"type":"ephemeral"}}]}]`,
		"grouped tool results": `{"role":"user","content":"Inspect the repo."},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"ls"}},` +
			`{"type":"tool_use","id":"c2","name":"Bash","input":{"command":"pwd"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"README.md"},` +
			`{"type":"tool_result","tool_use_id":"c2","content":"/work"},{"type":"text","text":"<system-reminder>x</system-reminder>"}]}]`,
		"non-ASCII and markup": `{"role":"user","content":"Überprüfe <b>die</b> Datei — 日本語 ✓ & more"}]`,
	}
	for name, messages := range cases {
		t.Run(name, func(t *testing.T) {
			body := anthropicReplayPrefix + messages + `,"stream":true}`
			want := strings.Replace(body, `"model":"client-model"`, `"model":"routed-model"`, 1)
			if got := routeAnthropicBody(t, body, nil); got != want {
				t.Fatalf("routed body =\n%s\nwant the client bytes with the model rewritten:\n%s", got, want)
			}
		})
	}
}

// Anything else the router changed sends the canonical encoding instead of
// the client's bytes.
func TestAnthropicRouteThatChangesMoreThanTheModelIsEncoded(t *testing.T) {
	body := anthropicReplayPrefix + `{"role":"user","content":"List the files."}],"stream":true}`
	limited := routeAnthropicBody(t, body, func(request *llmprotocol.Request) {
		limit := int64(1024)
		request.Sampling.MaxOutputTokens = &limit
	})
	if !strings.Contains(limited, `"max_tokens":1024`) || strings.Contains(limited, `"content":"List the files."`) {
		t.Fatalf("a changed limit was not encoded: %s", limited)
	}
	steered := routeAnthropicBody(t, body, func(request *llmprotocol.Request) {
		tail := &request.Messages[len(request.Messages)-1]
		tail.Content = append(tail.Content, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "steer"})
	})
	if !strings.Contains(steered, `"text":"steer"`) {
		t.Fatalf("a steered tail was not encoded: %s", steered)
	}
}

// Canonical encoding keeps the client's grouping: the tool results and text
// a client sent in one user message stay one message, and results the client
// sent as separate messages stay separate.
func TestAnthropicToolResultsKeepTheClientGrouping(t *testing.T) {
	const toolUses = `{"role":"user","content":"Inspect."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{}},` +
		`{"type":"tool_use","id":"c2","name":"Bash","input":{}}]},`
	cases := map[string]struct {
		messages string
		want     []int
	}{
		"one message with results and text": {
			messages: toolUses + `{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"a"},` +
				`{"type":"tool_result","tool_use_id":"c2","content":"b"},{"type":"text","text":"note"}]}]`,
			want: []int{1, 2, 3},
		},
		"results in separate messages": {
			messages: toolUses + `{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"a"}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c2","content":"b"}]}]`,
			want: []int{1, 2, 1, 1},
		},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			body := anthropicReplayPrefix + test.messages + `}`
			// A changed limit forces the canonical encoding.
			routed := routeAnthropicBody(t, body, func(request *llmprotocol.Request) {
				limit := int64(1024)
				request.Sampling.MaxOutputTokens = &limit
			})
			var wire struct {
				Messages []struct {
					Content []json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal([]byte(routed), &wire); err != nil {
				t.Fatalf("routed body: %v", err)
			}
			blocks := make([]int, len(wire.Messages))
			for index, message := range wire.Messages {
				blocks[index] = len(message.Content)
			}
			if len(blocks) != len(test.want) {
				t.Fatalf("blocks per message = %v, want %v: %s", blocks, test.want, routed)
			}
			for index := range blocks {
				if blocks[index] != test.want[index] {
					t.Fatalf("blocks per message = %v, want %v: %s", blocks, test.want, routed)
				}
			}
		})
	}
}
