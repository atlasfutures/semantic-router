package execution

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

// A resumed session reaches a core that has never seen it: its history is
// accepted, the reply is attributed from the first turn on, and no reasoning
// the earlier turns carried reaches the model as signed or encrypted input.
func TestResumedHistoryStartsAFreshLedger(t *testing.T) {
	cases := []struct {
		name, path        string
		protocol          llmprotocol.WireFormat
		history, reply    string
		answer, next      string
		inherited, intact []string
	}{
		{
			name: "messages", path: "/v1/messages", protocol: llmprotocol.AnthropicMessagesV1,
			history: `{"model":"arc-1.0","max_tokens":64,"messages":[` +
				`{"role":"user","content":"old task"},` +
				`{"role":"assistant","content":[{"type":"thinking","thinking":"inherited-thought","signature":"sig-inherited"},{"type":"text","text":"old reply"},{"type":"tool_use","id":"toolu_old","name":"read","input":{"path":"a"}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_old","content":"file body"}]},` +
				`{"role":"assistant","content":[{"type":"redacted_thinking","data":"redacted-inherited"},{"type":"text","text":"old done"}]},` +
				`{"role":"user","content":"continue"}]}`,
			reply:     `{"id":"msg_fixture","type":"message","role":"assistant","model":"qwen-fixture","content":[{"type":"thinking","thinking":"new-thought","signature":"sig-new"},{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`,
			answer:    `{"role":"assistant","content":[{"type":"thinking","thinking":"new-thought","signature":"sig-new"},{"type":"text","text":"ready"}]}`,
			next:      `{"role":"user","content":"again"}`,
			inherited: []string{"sig-inherited", "inherited-thought", "redacted-inherited"},
			intact:    []string{"old reply", "toolu_old", "file body", "old done"},
		},
		{
			name: "chat", path: "/v1/chat/completions", protocol: llmprotocol.OpenAIChatV1,
			history: `{"model":"arc-1.0","max_tokens":64,"messages":[` +
				`{"role":"user","content":"old task"},` +
				`{"role":"assistant","content":"old reply","reasoning_details":[{"type":"reasoning.encrypted","data":"details-inherited","format":"openai-responses-v1","index":0}],"tool_calls":[{"id":"call_old","type":"function","function":{"name":"read","arguments":"{}"}}]},` +
				`{"role":"tool","tool_call_id":"call_old","content":"file body"},` +
				`{"role":"user","content":"continue"}]}`,
			reply:     `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`,
			answer:    `{"role":"assistant","content":"ready"}`,
			next:      `{"role":"user","content":"again"}`,
			inherited: []string{"details-inherited"},
			intact:    []string{"old reply", "call_old", "file body"},
		},
		{
			name: "responses", path: "/v1/responses", protocol: llmprotocol.OpenAIResponsesV1,
			history: `{"model":"arc-1.0","max_output_tokens":64,"store":false,"input":[` +
				`{"role":"user","content":"old task"},` +
				`{"type":"reasoning","id":"rs_old","summary":[],"encrypted_content":"encrypted-inherited"},` +
				`{"type":"function_call","call_id":"call_old","name":"read","arguments":"{}"},` +
				`{"type":"function_call_output","call_id":"call_old","output":"file body"},` +
				`{"role":"assistant","content":[{"type":"output_text","text":"old reply"}]},` +
				`{"role":"user","content":"continue"}]}`,
			reply:     `{"id":"resp1","object":"response","created_at":1,"status":"completed","model":"qwen-fixture","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ready","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}`,
			answer:    `{"role":"assistant","content":[{"type":"output_text","text":"ready"}]}`,
			next:      `{"role":"user","content":"again"}`,
			inherited: []string{"encrypted-inherited"},
			intact:    []string{"old reply", "call_old", "file body"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var sent []string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				mu.Lock()
				sent = append(sent, string(body))
				mu.Unlock()
				_, _ = io.WriteString(w, tc.reply)
			}))
			defer upstream.Close()
			capture := &decisionCapture{}
			_, server := executorWithHTTPDecisions(t, onProtocol(t, testConfig(t, upstream.URL, false), tc.protocol), capture)
			if status, result := executeHTTP(t, server.URL, tc.path, tc.history, "resumed-"+tc.name); status != 200 {
				t.Fatalf("resumed history returned %d: %s", status, result)
			}
			second := strings.TrimSuffix(tc.history, "]}") + "," + tc.answer + "," + tc.next + "]}"
			if status, result := executeHTTP(t, server.URL, tc.path, second, "resumed-"+tc.name); status != 200 {
				t.Fatalf("turn after the resume returned %d: %s", status, result)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(sent) != 2 {
				t.Fatalf("expected two dispatches, got %d", len(sent))
			}
			for turn, body := range sent {
				for _, token := range tc.inherited {
					if strings.Contains(body, token) {
						t.Errorf("turn %d replayed inherited reasoning %q: %s", turn+1, token, body)
					}
				}
				for _, text := range tc.intact {
					if !strings.Contains(body, text) {
						t.Errorf("turn %d lost inherited conversation %q: %s", turn+1, text, body)
					}
				}
			}
			if tc.protocol == llmprotocol.AnthropicMessagesV1 && !strings.Contains(sent[1], "sig-new") {
				t.Errorf("reasoning signed after the resume was not carried: %s", sent[1])
			}
			capture.mu.Lock()
			defer capture.mu.Unlock()
			if len(capture.contexts) != 2 {
				t.Fatalf("expected two decisions, got %d", len(capture.contexts))
			}
			if first := capture.contexts[0]; len(first.Attribution) != 0 {
				t.Errorf("inherited history was attributed: %+v", first)
			}
			roles, err := wireMessageRoles(tc.protocol, []byte(tc.history))
			if err != nil {
				t.Fatal(err)
			}
			attribution := capture.contexts[1].Attribution
			if len(attribution) != 1 || attribution[0].Message != len(roles) {
				t.Errorf("the first turn after the resume was not attributed at index %d: %+v", len(roles), attribution)
			}
			if capture.contexts[0].ContextEpoch != capture.contexts[1].ContextEpoch {
				t.Errorf("an extending turn changed the context epoch: %s then %s", capture.contexts[0].ContextEpoch, capture.contexts[1].ContextEpoch)
			}
		})
	}
}

// The core cannot tell which model produced a resumed history, so the same
// target model and a different one get the same answer: inherited signed or
// encrypted reasoning is never forwarded, while reasoning issued after the
// resume still reaches the target that issued it.
func TestInheritedReasoningIsNeverReplayedToAnyModel(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	decode := func(t *testing.T, format llmprotocol.WireFormat, body string) (llmprotocol.Request, llmprotocol.Envelope) {
		t.Helper()
		request, envelope, _, err := engine.DecodeRequestForMutation(format, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return request, envelope
	}
	encode := func(t *testing.T, format llmprotocol.WireFormat, request llmprotocol.Request, envelope llmprotocol.Envelope) string {
		t.Helper()
		request.Generation++
		encoded, err := engine.EncodeRequest(format, request, envelope)
		if err != nil {
			t.Fatal(err)
		}
		return string(encoded.Body)
	}

	t.Run("claude thinking", func(t *testing.T) {
		inherited := `{"role":"user","content":"old"},` +
			`{"role":"assistant","content":[{"type":"thinking","thinking":"inherited-thought","signature":"sig-inherited"},{"type":"redacted_thinking","data":"redacted-inherited"},{"type":"text","text":"old reply"},{"type":"tool_use","id":"toolu_old","name":"read","input":{}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_old","content":"file body"}]}`
		resumed := `{"model":"claude","max_tokens":1024,"messages":[` + inherited + `]}`
		later := `{"model":"claude","max_tokens":1024,"messages":[` + inherited +
			`,{"role":"assistant","content":[{"type":"thinking","thinking":"new-thought","signature":"sig-new"},{"type":"text","text":"new reply"}]},{"role":"user","content":"next"}]}`
		// The history may have come from either model; neither may verify it.
		for _, model := range []string{"claude-opus-4-6", "claude-sonnet-4-6"} {
			t.Run(model, func(t *testing.T) {
				state := newBranchState()
				request, _ := decode(t, llmprotocol.AnthropicMessagesV1, resumed)
				state.inheritHistory(request)
				deployment := Deployment{Kind: "cloud", ID: model, RegistryModel: model, Protocol: llmprotocol.AnthropicMessagesV1}
				for _, body := range []string{resumed, later} {
					request, envelope := decode(t, llmprotocol.AnthropicMessagesV1, body)
					prepareReasoning(&request, deployment, state)
					wire := encode(t, llmprotocol.AnthropicMessagesV1, request, envelope)
					for _, token := range []string{"sig-inherited", "inherited-thought", "redacted-inherited"} {
						if strings.Contains(wire, token) {
							t.Fatalf("inherited reasoning %q reached %s: %s", token, model, wire)
						}
					}
					for _, text := range []string{"old reply", "toolu_old", "file body"} {
						if !strings.Contains(wire, text) {
							t.Fatalf("inherited conversation %q was lost: %s", text, wire)
						}
					}
					if body == later && !strings.Contains(wire, `"signature":"sig-new"`) {
						t.Fatalf("thinking signed after the resume was dropped: %s", wire)
					}
				}
			})
		}
	})

	t.Run("encrypted responses reasoning", func(t *testing.T) {
		inherited := `{"role":"user","content":"old"},{"type":"reasoning","id":"rs_old","summary":[],"encrypted_content":"encrypted-inherited"},{"role":"assistant","content":[{"type":"output_text","text":"old reply"}]}`
		resumed := `{"model":"gpt","input":[` + inherited + `,{"role":"user","content":"continue"}]}`
		later := `{"model":"gpt","input":[` + inherited + `,{"role":"user","content":"continue"},{"type":"reasoning","id":"rs_new","summary":[],"encrypted_content":"encrypted-new"},{"role":"assistant","content":[{"type":"output_text","text":"new reply"}]},{"role":"user","content":"next"}]}`
		for _, model := range []string{"gpt-5.5", "gpt-5.5-mini"} {
			t.Run(model, func(t *testing.T) {
				deployment := Deployment{Kind: "cloud", ID: model, URL: "https://provider.example/v1/responses", WireModel: model, IssuerSHA256: "approved-profile", Protocol: llmprotocol.OpenAIResponsesV1}
				state := newBranchState()
				request, envelope := decode(t, llmprotocol.OpenAIResponsesV1, resumed)
				state.inheritHistory(request)
				// The resumed turn: nothing it holds is this target's.
				state.issuers = prepareIssuersAndCheck(t, encode, request, envelope, deployment, state, "encrypted-inherited", "")
				// After the resume the target issued encrypted-new; the client
				// resends it beside the inherited blob.
				request, envelope = decode(t, llmprotocol.OpenAIResponsesV1, later)
				prepareIssuersAndCheck(t, encode, request, envelope, deployment, state, "encrypted-inherited", "encrypted-new")
			})
		}
	})
}

func prepareIssuersAndCheck(t *testing.T, encode func(*testing.T, llmprotocol.WireFormat, llmprotocol.Request, llmprotocol.Envelope) string,
	request llmprotocol.Request, envelope llmprotocol.Envelope, deployment Deployment, state *branchState, absent, present string,
) []string {
	t.Helper()
	issuers, _ := prepareReasoning(&request, deployment, state)
	wire := encode(t, llmprotocol.OpenAIResponsesV1, request, envelope)
	if strings.Contains(wire, absent) {
		t.Fatalf("inherited encrypted reasoning reached %s: %s", deployment.ID, wire)
	}
	if !strings.Contains(wire, "old reply") {
		t.Fatalf("inherited conversation was lost: %s", wire)
	}
	if present != "" && !strings.Contains(wire, present) {
		t.Fatalf("reasoning %s issued after the resume was not forwarded: %s", deployment.ID, wire)
	}
	return issuers
}
