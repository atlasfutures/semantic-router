package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// emptyCompletionState is a stream that reached its terminal event with one
// output item the stream completed and no content on it. An upstream model
// that spends its turn on a stop token ends this way: the client is served a
// finished message, the usage counts a completion token, and no delta ever
// arrives.
func emptyCompletionState() *semanticResponseStreamState {
	state := &semanticResponseStreamState{
		responseID: "response",
		model:      "provider-model",
		stop:       llmprotocol.StopEndTurn,
		items:      map[int]*semanticStreamItem{},
		terminal:   true,
	}
	state.item(0).completed = true
	return state
}

// TestEmptyCompletionReconstructsInsteadOfFailing is the ask.
//
// response() refuses an output item the stream completed with no content
// (processor_res_semantic_stream.go:328). finalizeSemanticStreamingResponse
// reports the turn's usage first (line 371) and then returns on that error
// (line 374), so updateResponseCache, scheduleSemanticResponseMemoryStore and
// persistResponseObject (lines 390-392) never run. The turn is charged and
// counted while no record of what was served survives it.
//
// A completed item with no content is an empty answer, not a broken stream,
// and should reconstruct as one.
func TestEmptyCompletionReconstructsInsteadOfFailing(t *testing.T) {
	response, err := emptyCompletionState().response()
	if err != nil {
		t.Fatalf("an empty completion failed reconstruction: %v", err)
	}
	if len(response.Output) != 1 {
		t.Fatalf("reconstructed %d output items, want 1", len(response.Output))
	}
	contents := response.Output[0].Content
	if len(contents) != 1 || contents[0].Kind != llmprotocol.ContentText || contents[0].Text != "" {
		t.Fatalf("reconstructed content = %+v, want one empty text block", contents)
	}
	if response.StopReason != llmprotocol.StopEndTurn {
		t.Fatalf("stop reason = %q, want %q", response.StopReason, llmprotocol.StopEndTurn)
	}
}

// TestIncompleteItemStillFailsReconstruction guards the line that stays. An
// item the stream never completed is a truncated turn, and reconstructing it
// as an empty answer would report a real failure as a success. It passes
// today and must keep passing.
func TestIncompleteItemStillFailsReconstruction(t *testing.T) {
	state := emptyCompletionState()
	state.item(0).completed = false

	if _, err := state.response(); err == nil {
		t.Fatal("an output item the stream never completed was reconstructed")
	}
}

// TestEmptyCompletionIsRefusedToday pins the present behaviour so a change to
// it shows up in the diff rather than only in the test above.
func TestEmptyCompletionIsRefusedToday(t *testing.T) {
	response, err := emptyCompletionState().response()
	if err == nil {
		t.Fatalf("the recorded behaviour has changed: reconstruction returned %+v", response)
	}
	if err.Error() != "semantic stream output item is empty" {
		t.Fatalf("reconstruction failed with %q, want the recorded refusal", err)
	}
}
