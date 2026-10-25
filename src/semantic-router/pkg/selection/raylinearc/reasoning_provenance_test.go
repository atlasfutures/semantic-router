package raylinearc

import (
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
)

// The record survives the stored episode state, and a malformed one is
// refused there like any other corrupt state.
func TestReasoningProvenancePersists(t *testing.T) {
	state, _ := NewEpisodeState(2)
	state.ReasoningProvenance = reasoning.WithReasoningProvenance(nil, "gpt", []string{"blob-a"})
	now := time.Now()
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, _, err := unmarshalEpisodeState(payload, 2, now)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(decoded.ReasoningProvenance) != 1 || decoded.ReasoningProvenance[0] != state.ReasoningProvenance[0] {
		t.Fatalf("decoded provenance = %+v", decoded.ReasoningProvenance)
	}
	state.ReasoningProvenance[0].Worker = "not-a-digest"
	if _, err := marshalEpisodeState(state, 1, now); err == nil {
		t.Fatal("a malformed provenance entry was persisted")
	}
}
