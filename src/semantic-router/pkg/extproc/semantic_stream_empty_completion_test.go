package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An upstream completion that produces only a stop token is a legal outcome,
// not a broken stream. On 2026-09-03 a turn on deepseek-v4-flash@thinking-on
// ended that way: the client was served content_block_start with text "",
// no deltas, content_block_stop, stop_reason end_turn and output_tokens 1.
// Reconstruction refused it -- "semantic stream output item is empty" -- so
// the turn was billed and reported successful while settlement held no
// response at all.
func TestEmptyCompletionReconstructsInsteadOfFailing(t *testing.T) {
	state := &semanticResponseStreamState{
		responseID: "gen-1788474769-7MLMxl4euvXqO1r4CdYp",
		model:      "deepseek/deepseek-v4-flash@thinking-on",
		stop:       llmprotocol.StopEndTurn,
		items:      map[int]*semanticStreamItem{},
		terminal:   true,
	}
	item := state.item(0)
	item.completed = true

	response, err := state.response()
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
		t.Fatalf("stop reason = %q, want end_turn", response.StopReason)
	}
}

// An item the stream never completed is still a broken stream. Reconstructing
// a truncated turn as an empty answer would hide a real failure.
func TestIncompleteItemStillFailsReconstruction(t *testing.T) {
	state := &semanticResponseStreamState{
		responseID: "gen-1", model: "m", stop: llmprotocol.StopEndTurn,
		items: map[int]*semanticStreamItem{}, terminal: true,
	}
	state.item(0)

	if _, err := state.response(); err == nil {
		t.Fatal("an incomplete output item was reconstructed")
	}
}
