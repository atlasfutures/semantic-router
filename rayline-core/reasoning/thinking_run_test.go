package reasoning

import (
	"fmt"
	"testing"
)

func TestThinkingRunAnchorsOnceAndEndsOtherRuns(t *testing.T) {
	sonnet, opus := ThinkingRunWorker("sonnet"), ThinkingRunWorker("opus")
	opaque := WithReasoningProvenance(nil, "claude", []string{"redacted-blob"})

	provenance := WithThinkingRun(opaque, sonnet, []string{opus}, false, "sig-1")
	if got := ThinkingRunAnchors(provenance, sonnet); fmt.Sprint(got) != fmt.Sprint([]string{ThinkingRunAnchorDigest("sig-1")}) {
		t.Fatalf("anchors = %v", got)
	}
	// An anchored turn and a turn that signed nothing keep the anchor.
	provenance = WithThinkingRun(provenance, sonnet, []string{opus}, true, "sig-2")
	provenance = WithThinkingRun(provenance, sonnet, []string{opus}, false, "")
	if got := ThinkingRunAnchors(provenance, sonnet); len(got) != 1 || got[0] != ThinkingRunAnchorDigest("sig-1") {
		t.Fatalf("anchors moved: %v", got)
	}
	// A turn on another model ends the run; opaque provenance is untouched.
	provenance = WithThinkingRun(provenance, opus, []string{sonnet}, false, "sig-3")
	if got := ThinkingRunAnchors(provenance, sonnet); len(got) != 0 {
		t.Fatalf("the ended run kept %v", got)
	}
	if !ReasoningIssuedElsewhere(provenance, "redacted-blob", "other") {
		t.Fatalf("opaque provenance was lost: %+v", provenance)
	}
	if err := ValidateReasoningProvenance(provenance); err != nil {
		t.Fatal(err)
	}
}

func TestThinkingRunKeepsItsAnchorsWithinBounds(t *testing.T) {
	worker := ThinkingRunWorker("sonnet")
	var provenance []ReasoningProvenance
	for index := range MaxThinkingRunAnchors + 2 {
		provenance = WithThinkingRun(provenance, worker, nil, false, fmt.Sprintf("sig-%d", index))
	}
	anchors := ThinkingRunAnchors(provenance, worker)
	if len(anchors) != MaxThinkingRunAnchors || anchors[len(anchors)-1] != ThinkingRunAnchorDigest(fmt.Sprintf("sig-%d", MaxThinkingRunAnchors+1)) {
		t.Fatalf("anchors = %v", anchors)
	}
	// Opaque blocks recorded after the anchor never push it out: the run's
	// next turn moves it to the newest end.
	blocks := make([]string, MaxReasoningProvenance)
	for index := range blocks {
		blocks[index] = fmt.Sprintf("blob-%d", index)
	}
	provenance = WithReasoningProvenance(provenance, "claude", blocks[:MaxReasoningProvenance-MaxThinkingRunAnchors])
	provenance = WithThinkingRun(provenance, worker, nil, true, "")
	if len(provenance) > MaxReasoningProvenance || len(ThinkingRunAnchors(provenance, worker)) != MaxThinkingRunAnchors {
		t.Fatalf("provenance = %d entries, anchors %v", len(provenance), ThinkingRunAnchors(provenance, worker))
	}
}

// An anchor at a redacted_thinking block is never read as that block's
// issuer record: the opaque layer leaves the block to its own rules.
func TestThinkingRunAnchorIsNotAnOpaqueIssuerRecord(t *testing.T) {
	worker := ThinkingRunWorker("opus")
	provenance := WithThinkingRun(nil, worker, nil, false, "redacted-data")
	if ReasoningIssuedElsewhere(provenance, "redacted-data", "opus-target") {
		t.Fatalf("a run anchor read as an opaque issuer: %+v", provenance)
	}
}
