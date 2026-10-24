package reasoning

import (
	"testing"
)

// A block is known to be another worker's only when the episode recorded it;
// the record keeps each block once, newest last, within its bound.
func TestReasoningProvenance(t *testing.T) {
	provenance := WithReasoningProvenance(nil, "gpt", []string{"blob-a", "blob-b"})
	provenance = WithReasoningProvenance(provenance, "claude", []string{"blob-b"})
	if len(provenance) != 2 {
		t.Fatalf("provenance = %+v, want blob-a and blob-b once each", provenance)
	}
	switch {
	case !ReasoningIssuedElsewhere(provenance, "blob-a", "claude"):
		t.Fatal("gpt's block is not known as issued elsewhere for claude")
	case ReasoningIssuedElsewhere(provenance, "blob-a", "gpt"):
		t.Fatal("gpt's block counts as elsewhere for gpt")
	case ReasoningIssuedElsewhere(provenance, "blob-b", "claude"):
		t.Fatal("blob-b, last issued by claude, counts as elsewhere for claude")
	case ReasoningIssuedElsewhere(provenance, "unrecorded", "claude"):
		t.Fatal("an unrecorded block counts as issued elsewhere")
	}
	var many []string
	for index := range MaxReasoningProvenance + 5 {
		many = append(many, string(rune('a'+index%26))+string(rune('0'+index/26)))
	}
	if bounded := WithReasoningProvenance(nil, "gpt", many); len(bounded) != MaxReasoningProvenance {
		t.Fatalf("provenance holds %d entries, want %d", len(bounded), MaxReasoningProvenance)
	}
}

// A block from an issuer whose provider can vary, or headed to such a target,
// is never resent: neither side can be shown to read it.
func TestReasoningIssuedElsewhereWithAnUnknownIssuer(t *testing.T) {
	unknownBlock := WithReasoningProvenance(nil, ReasoningIssuerUnknown, []string{"blob"})
	if !ReasoningIssuedElsewhere(unknownBlock, "blob", "claude") || !ReasoningIssuedElsewhere(unknownBlock, "blob", ReasoningIssuerUnknown) {
		t.Fatal("a block from an unknown issuer was resent")
	}
	known := WithReasoningProvenance(nil, "claude", []string{"blob"})
	if !ReasoningIssuedElsewhere(known, "blob", ReasoningIssuerUnknown) {
		t.Fatal("a block was resent to a target whose provider can vary")
	}
}
