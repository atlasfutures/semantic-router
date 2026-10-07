package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Anthropic accepts arguments with a repeated member (the last wins) or an
// unpaired surrogate escape (decoded as U+FFFD). A streamed call that writes
// them completes, under every client format, with the arguments settled as
// Anthropic settles them; the turn does not fail.
func TestStreamedToolArgumentsAnthropicAcceptsComplete(t *testing.T) {
	for name, test := range map[string]struct {
		chunks   []string
		settled  string
		splitsAt string
	}{
		"repeated member": {
			chunks:  []string{`{"path":"a.py","edits":[{"oldText":"x ≥ y","newText":"z"}],`, `"path":"b.py"}`},
			settled: `{"path":"b.py","edits":[{"oldText":"x ≥ y","newText":"z"}]}`,
		},
		"unpaired high surrogate": {
			chunks:  []string{`{"path":"a.py","edits":[{"oldText":"x \ud83d`, ` y","newText":"z"}]}`},
			settled: "{\"path\":\"a.py\",\"edits\":[{\"oldText\":\"x � y\",\"newText\":\"z\"}]}",
		},
		"unpaired low surrogate": {
			chunks:  []string{`{"path":"a.py","edits":[{"oldText":"\ude00"}]}`},
			settled: "{\"path\":\"a.py\",\"edits\":[{\"oldText\":\"�\"}]}",
		},
	} {
		body := anthropicToolStream(test.chunks, anthropicStop("tool_use"))
		t.Run(name+"/responses", func(t *testing.T) {
			wire, events, diagnostics := runSettledToolStream(t, llmprotocol.OpenAIResponsesV1, body)
			requireSettledCall(t, events, diagnostics, test.settled)
			done := responsesWireEvent(t, wire, "response.function_call_arguments.done")
			if done["arguments"] != test.settled {
				t.Fatalf("function_call_arguments.done arguments = %v, want %s", done["arguments"], test.settled)
			}
			if !strings.Contains(wire, "event: response.completed") {
				t.Fatalf("the turn did not complete:\n%s", wire)
			}
		})
		t.Run(name+"/anthropic", func(t *testing.T) {
			wire, events, diagnostics := runSettledToolStream(t, llmprotocol.AnthropicMessagesV1, body)
			requireSettledCall(t, events, diagnostics, test.settled)
			if !strings.Contains(wire, `"stop_reason":"tool_use"`) || !strings.Contains(wire, "event: message_stop") {
				t.Fatalf("the turn did not end as tool_use:\n%s", wire)
			}
		})
		t.Run(name+"/chat", func(t *testing.T) {
			wire, events, diagnostics := runSettledToolStream(t, llmprotocol.OpenAIChatV1, body)
			requireSettledCall(t, events, diagnostics, test.settled)
			if !strings.Contains(wire, `"finish_reason":"tool_calls"`) || !strings.Contains(wire, "data: [DONE]") {
				t.Fatalf("the turn did not end with its tool call:\n%s", wire)
			}
		})
	}
}

func runSettledToolStream(t *testing.T, target llmprotocol.WireFormat, body string) (string, []llmprotocol.Event, llmprotocol.Diagnostics) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, diagnostics, err := stream.Push([]byte(body))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	final, finalEvents, finalDiagnostics, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	return string(bytes.Join(append(frames, final...), nil)), append(events, finalEvents...), append(diagnostics, finalDiagnostics...)
}

func requireSettledCall(t *testing.T, events []llmprotocol.Event, diagnostics llmprotocol.Diagnostics, settled string) {
	t.Helper()
	completed := false
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseFailed {
			t.Fatalf("the turn failed: %+v", event.Error)
		}
		if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
			if event.ToolCall.Arguments != settled || event.ToolCall.Incomplete {
				t.Fatalf("completed call = %+v, want whole arguments %s", event.ToolCall, settled)
			}
			completed = true
		}
	}
	if !completed {
		t.Fatal("no tool call completed")
	}
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "tool_call.arguments" && diagnostic.Action == llmprotocol.DiagnosticApproximated {
			return
		}
	}
	t.Fatalf("no diagnostic says the arguments were settled: %+v", diagnostics)
}

func responsesWireEvent(t *testing.T, wire, eventType string) map[string]any {
	t.Helper()
	for _, frame := range strings.Split(wire, "\n\n") {
		if !strings.HasPrefix(frame, "event: "+eventType+"\n") {
			continue
		}
		var data map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(frame[strings.Index(frame, "\n")+1:], "data: ")), &data); err != nil {
			t.Fatal(err)
		}
		return data
	}
	t.Fatalf("no %s event:\n%s", eventType, wire)
	return nil
}

// Settled arguments are bounded like streamed ones: the re-encoding writes a
// raw U+2028 as a six-byte escape, so arguments within the limit as streamed
// can exceed it once settled. They are refused at completion, where the
// settlement happens, rather than by whichever later stage measures them.
func TestSettledToolArgumentsAreBoundedAgain(t *testing.T) {
	streamed := `{"a":"x","a":"` + strings.Repeat(" ", 10) + `"}`
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.ToolArgumentsBytes = len(streamed) + 4
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push([]byte(anthropicToolStream([]string{streamed}, anthropicStop("tool_use"))))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "tool_arguments_limit" {
		t.Fatalf("push error = %v, want tool_arguments_limit", err)
	}
	if !strings.Contains(protocolError.Message, "settled") {
		t.Fatalf("refused as %q; want the settled-arguments bound, at completion", protocolError.Message)
	}
}

// The shape #225's live failure was suspected of (harness run13, an edit call
// whose text wrote "planarity ≥1.0"): a multi-byte character split across
// input_json_delta chunks, as Anthropic can split it. Each split completes
// under every client format and every transport chunking, with the
// arguments byte for byte, so this shape is not the cause; it is pinned so
// it stays so. A split inside the UTF-8 bytes of one frame is not something
// Anthropic's JSON frames can carry, and the decoder refuses it as
// invalid_upstream_utf8, which is not the code the live turn failed with.
func TestStreamedToolArgumentsWithASplitMultibyteCharacterComplete(t *testing.T) {
	arguments := `{"path":"planar.py","edits":[{"oldText":"pos = data['pos']\n","newText":"# planarity ≥1.0\n"}]}`
	at := strings.Index(arguments, "≥")
	escaped := strings.Replace(arguments, "≥", `≥`, 1)
	escapedAt := strings.Index(escaped, `≥`)
	for name, test := range map[string]struct {
		chunks []string
		want   string
	}{
		"split before the character": {chunks: []string{arguments[:at], arguments[at:]}, want: arguments},
		"split after the character":  {chunks: []string{arguments[:at+len("≥")], arguments[at+len("≥"):]}, want: arguments},
		"split inside its escape":    {chunks: []string{escaped[:escapedAt+4], escaped[escapedAt+4:]}, want: escaped},
	} {
		body := anthropicToolStream(test.chunks, anthropicStop("tool_use"))
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1} {
			for _, step := range []int{0, 1, 13} {
				t.Run(fmt.Sprintf("%s/%s/%d", name, target, step), func(t *testing.T) {
					stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
						llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
					if err != nil {
						t.Fatal(err)
					}
					var events []llmprotocol.Event
					data := []byte(body)
					size := step
					if size == 0 {
						size = len(data)
					}
					for offset := 0; offset < len(data); offset += size {
						end := min(offset+size, len(data))
						_, pushed, _, pushErr := stream.Push(data[offset:end])
						if pushErr != nil {
							t.Fatalf("push: %v", pushErr)
						}
						events = append(events, pushed...)
					}
					_, final, _, err := stream.Finalize(nil)
					if err != nil {
						t.Fatalf("finalize: %v", err)
					}
					completed := false
					for _, event := range append(events, final...) {
						if event.Type == llmprotocol.EventResponseFailed {
							t.Fatalf("the turn failed: %+v", event.Error)
						}
						if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
							if event.ToolCall.Arguments != test.want || event.ToolCall.Incomplete {
								t.Fatalf("completed call arguments = %q, want %q", event.ToolCall.Arguments, test.want)
							}
							completed = true
						}
					}
					if !completed {
						t.Fatal("no tool call completed")
					}
				})
			}
		}
	}
}
