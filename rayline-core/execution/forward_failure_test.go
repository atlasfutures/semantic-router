package execution

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// forwardFixture is one inbound protocol served by a local deployment of the
// same protocol, with the upstream bodies the tests below cut short.
type forwardFixture struct {
	path   string
	wire   llmprotocol.WireFormat
	body   string   // the client request; %v is its stream flag
	prefix string   // the stream's opening events, ending with PARTIAL_ANSWER
	fail   string   // the provider's in-stream error event
	failed []string // the client's stream must carry one of these error events
	done   []string // success terminals the client must never see after a failure
	more   func(sequence int) string
	json   string // a complete buffered answer
}

func sseFrame(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }

var forwardFixtures = []forwardFixture{
	{
		path: "/v1/messages", wire: llmprotocol.AnthropicMessagesV1,
		body: `{"model":"arc-1.0","max_tokens":64,"stream":%v,"messages":[{"role":"user","content":"hello"}]}`,
		prefix: sseFrame("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"qwen-fixture","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}`) +
			sseFrame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
			sseFrame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"PARTIAL_ANSWER"}}`),
		fail:   sseFrame("error", `{"type":"error","error":{"type":"overloaded_error","message":"mock provider stream failed"}}`),
		failed: []string{"event: error\n"},
		done:   []string{"message_stop"},
		more: func(int) string {
			return sseFrame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" MORE"}}`)
		},
		json: `{"id":"msg_1","type":"message","role":"assistant","model":"qwen-fixture","content":[{"type":"text","text":"PARTIAL_ANSWER and the rest"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":3}}`,
	},
	{
		path: "/v1/chat/completions", wire: llmprotocol.OpenAIChatV1,
		body:   `{"model":"arc-1.0","max_tokens":64,"stream":%v,"messages":[{"role":"user","content":"hello"}]}`,
		prefix: "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"PARTIAL_ANSWER\"},\"finish_reason\":null}]}\n\n",
		fail:   "data: {\"error\":{\"message\":\"mock provider stream failed\",\"type\":\"server_error\",\"param\":null,\"code\":\"provider_overloaded\"}}\n\n",
		failed: []string{"data: {\"error\":"},
		done:   []string{`"finish_reason":"stop"`, "[DONE]"},
		more: func(int) string {
			return "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" MORE\"},\"finish_reason\":null}]}\n\n"
		},
		json: `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"PARTIAL_ANSWER and the rest"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`,
	},
	{
		path: "/v1/responses", wire: llmprotocol.OpenAIResponsesV1,
		body: `{"model":"arc-1.0","max_output_tokens":64,"stream":%v,"input":[{"role":"user","content":"hello"}]}`,
		prefix: sseFrame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"model":"qwen-fixture","status":"in_progress","output":[]}}`) +
			sseFrame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`) +
			sseFrame("response.content_part.added", `{"type":"response.content_part.added","sequence_number":2,"output_index":0,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`) +
			sseFrame("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"item_id":"msg_1","content_index":0,"delta":"PARTIAL_ANSWER"}`),
		fail:   sseFrame("response.failed", `{"type":"response.failed","sequence_number":4,"response":{"id":"resp_1","object":"response","created_at":1,"model":"qwen-fixture","status":"failed","output":[],"error":{"code":"provider_overloaded","message":"mock provider stream failed"}}}`),
		failed: []string{"event: response.failed\n", "event: error\n"},
		done:   []string{"response.completed"},
		more: func(sequence int) string {
			return sseFrame("response.output_text.delta", fmt.Sprintf(`{"type":"response.output_text.delta","sequence_number":%d,"output_index":0,"item_id":"msg_1","content_index":0,"delta":" MORE"}`, sequence))
		},
		json: `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"qwen-fixture","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"PARTIAL_ANSWER and the rest","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`,
	},
}

func forwardServer(t *testing.T, fixture forwardFixture, handler http.HandlerFunc) (*Server, *httptest.Server) {
	t.Helper()
	upstream := httptest.NewServer(handler)
	t.Cleanup(upstream.Close)
	return executorWithHTTPDecisions(t, onProtocol(t, testConfig(t, upstream.URL, false), fixture.wire), &decisionCapture{})
}

// A buffered answer whose upstream body ended early (JSON short of its
// Content-Length) reaches the client as a 502 without the partial text, and
// commits nothing. A buffered request is never answered from an event stream:
// a stream body, whether it carries an error event or ends before its
// terminal event, is refused as an invalid body, so neither can become a
// finished answer.
func TestBufferedUpstreamFailureIsA502WithoutPartialText(t *testing.T) {
	reasons := map[string]string{
		"stream-error":     "upstream response body must be a JSON object",
		"stream-truncated": "upstream response body must be a JSON object",
		"json-truncated":   "selected endpoint response was incomplete",
	}
	for _, fixture := range forwardFixtures {
		for _, shape := range []string{"stream-error", "stream-truncated", "json-truncated"} {
			t.Run(fixture.path+"/"+shape, func(t *testing.T) {
				server, core := forwardServer(t, fixture, func(w http.ResponseWriter, r *http.Request) {
					switch shape {
					case "stream-error":
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, fixture.prefix+fixture.fail)
					case "stream-truncated":
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, fixture.prefix)
					case "json-truncated":
						w.Header().Set("Content-Type", "application/json")
						w.Header().Set("Content-Length", fmt.Sprint(len(fixture.json)))
						// Cut just past the partial text, short of Content-Length.
						_, _ = io.WriteString(w, fixture.json[:strings.Index(fixture.json, "PARTIAL_ANSWER")+len("PARTIAL_ANSWER")])
					}
				})
				status, body := executeHTTP(t, core.URL, fixture.path, fmt.Sprintf(fixture.body, false), "buffered-"+shape)
				if status != http.StatusBadGateway || strings.Contains(string(body), "PARTIAL_ANSWER") || !strings.Contains(string(body), reasons[shape]) {
					t.Fatalf("a failed buffered upstream reached the client as %d: %s", status, body)
				}
				state, err := server.states.acquire(t.Context(), "buffered-"+shape, "main")
				if err != nil {
					t.Fatal(err)
				}
				defer state.release()
				if state.completed != 0 {
					t.Error("a failed buffered turn committed as completed")
				}
			})
		}
	}
}

// A streamed answer whose upstream failed or ended early ends with the
// listener's error event, never with a success terminal; a provider's error
// reaches the client as that provider's error.
func TestStreamedUpstreamFailureEndsWithAnError(t *testing.T) {
	for _, fixture := range forwardFixtures {
		for _, shape := range []string{"stream-error", "stream-truncated"} {
			t.Run(fixture.path+"/"+shape, func(t *testing.T) {
				_, core := forwardServer(t, fixture, func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, fixture.prefix)
					if shape == "stream-error" {
						_, _ = io.WriteString(w, fixture.fail)
					}
				})
				_, body := executeHTTP(t, core.URL, fixture.path, fmt.Sprintf(fixture.body, true), "streamed-"+shape)
				text := string(body)
				failed := false
				for _, marker := range fixture.failed {
					failed = failed || strings.Contains(text, marker)
				}
				for _, marker := range fixture.done {
					failed = failed && !strings.Contains(text, marker)
				}
				if shape == "stream-error" && !strings.Contains(text, "mock provider stream failed") {
					failed = false
				}
				if !strings.Contains(text, "PARTIAL_ANSWER") || !failed {
					t.Fatalf("a failed stream did not end with its error: %s", text)
				}
			})
		}
	}
}

// A client that leaves a streamed answer cancels the upstream request, both
// while the upstream is still generating and while it is stalled, and the
// turn is not committed.
func TestClientLeavingAStreamCancelsTheUpstream(t *testing.T) {
	for _, fixture := range forwardFixtures {
		for _, mode := range []string{"generating", "stalled"} {
			t.Run(fixture.path+"/"+mode, func(t *testing.T) {
				cancelled := make(chan bool, 1)
				server, core := forwardServer(t, fixture, func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, fixture.prefix)
					w.(http.Flusher).Flush()
					deadline := time.After(5 * time.Second)
					tick := time.NewTicker(10 * time.Millisecond)
					defer tick.Stop()
					for sequence := 4; ; sequence++ {
						select {
						case <-r.Context().Done():
							cancelled <- true
							return
						case <-deadline:
							cancelled <- false
							return
						case <-tick.C:
							if mode == "generating" {
								_, _ = io.WriteString(w, fixture.more(sequence))
								w.(http.Flusher).Flush()
							}
						}
					}
				})
				session := "leaving-" + mode
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				request, err := http.NewRequestWithContext(ctx, http.MethodPost, core.URL+fixture.path, strings.NewReader(fmt.Sprintf(fixture.body, true)))
				if err != nil {
					t.Fatal(err)
				}
				request.Header.Set("Authorization", "Bearer "+testCoreToken)
				request.Header.Set("X-Rayline-Session-Id", session)
				response, err := http.DefaultClient.Do(request)
				if err != nil {
					t.Fatal(err)
				}
				// Generating: wait until the upstream's later output reaches the
				// client, so the client leaves a stream that is still running.
				want := map[string]string{"generating": " MORE", "stalled": "PARTIAL_ANSWER"}[mode]
				reader := bufio.NewReader(response.Body)
				for {
					line, err := reader.ReadString('\n')
					if err != nil {
						t.Fatalf("the stream ended before %q: %v", want, err)
					}
					if strings.Contains(line, want) {
						break
					}
				}
				cancel()
				_ = response.Body.Close()
				select {
				case ok := <-cancelled:
					if !ok {
						t.Fatal("the upstream request outlived the client by 5s")
					}
				case <-time.After(10 * time.Second):
					t.Fatal("the upstream handler never reported")
				}
				// The handler holds the branch until it returns.
				state, err := server.states.acquire(t.Context(), session, "main")
				if err != nil {
					t.Fatal(err)
				}
				defer state.release()
				if state.completed != 0 {
					t.Error("an abandoned turn committed as completed")
				}
			})
		}
	}
}

// A client that leaves a buffered answer cancels the upstream request while
// the upstream stalls on its body.
func TestClientLeavingABufferedAnswerCancelsTheUpstream(t *testing.T) {
	for _, fixture := range forwardFixtures {
		t.Run(fixture.path, func(t *testing.T) {
			started := make(chan struct{})
			cancelled := make(chan bool, 1)
			_, core := forwardServer(t, fixture, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = io.WriteString(w, fixture.json[:10])
				w.(http.Flusher).Flush()
				close(started)
				select {
				case <-r.Context().Done():
					cancelled <- true
				case <-time.After(5 * time.Second):
					cancelled <- false
				}
			})
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			request, err := http.NewRequestWithContext(ctx, http.MethodPost, core.URL+fixture.path, strings.NewReader(fmt.Sprintf(fixture.body, false)))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+testCoreToken)
			request.Header.Set("X-Rayline-Session-Id", "leaving-buffered")
			returned := make(chan error, 1)
			go func() {
				response, err := http.DefaultClient.Do(request)
				if err == nil {
					_ = response.Body.Close()
				}
				returned <- err
			}()
			select {
			case <-started:
			case <-time.After(10 * time.Second):
				t.Fatal("the request never reached the upstream")
			}
			cancel()
			select {
			case ok := <-cancelled:
				if !ok {
					t.Fatal("the upstream request outlived the client by 5s")
				}
			case <-time.After(10 * time.Second):
				t.Fatal("the upstream handler never reported")
			}
			if err := <-returned; err == nil {
				t.Fatal("a stalled buffered answer returned a response")
			}
		})
	}
}
