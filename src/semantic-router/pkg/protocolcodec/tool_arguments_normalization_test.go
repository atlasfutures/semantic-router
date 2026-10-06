package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
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
