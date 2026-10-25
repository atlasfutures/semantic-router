package protocolcodec

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func TestResponseAccumulatorRequiresTerminalAndIsRepeatable(t *testing.T) {
	accumulator := NewResponseAccumulator()
	if _, err := accumulator.Response(); err == nil {
		t.Fatal("incomplete stream produced a completed response")
	}
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted, ResponseID: "response"},
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, Role: llmprotocol.RoleAssistant},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0, Delta: "answer"},
		{Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn},
	}
	if err := accumulator.Apply(events); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		response, err := accumulator.Response()
		if err != nil || len(response.Output) != 1 || response.Output[0].Content[0].Text != "answer" {
			t.Fatalf("response = %+v, %v", response, err)
		}
	}
}
