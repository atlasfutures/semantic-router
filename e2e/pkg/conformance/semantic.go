package conformance

import (
	"fmt"
	"sort"
)

// compareSemantic requires typed equivalence rather than wire identity.
//
// A buffered body is compared as decoded JSON, so key order and insignificant
// whitespace never fail. A stream is compared as its ordered dispatched events, so
// chunk boundaries, CRLF, comments, keepalives, and a missing final newline never
// fail either. Pointers declared volatile must match by JSON type, not by value.
//
// Headers are not compared: a cross-protocol path legitimately rewrites them, and
// the header contract belongs to the exact and exact-except boundaries.
func compareSemantic(cmp Comparison, want, got Payload) ([]Mismatch, error) {
	volatile := volatileSet(cmp.Volatile)

	mismatches := compareStatus(want.Status, got.Status)
	body, err := compareBodies(want, got, func(path string, wantValue, gotValue any, out *[]Mismatch) error {
		diffJSON(path, wantValue, gotValue, volatile, out)
		return nil
	}, func(wantEvent, gotEvent SSEEvent, coordinate string, out *[]Mismatch) {
		// The SSE id field is reconnection transport metadata, not protocol
		// semantics, so semantic mode ignores it. exact and exact-except still check it.
		compareEventData(wantEvent, gotEvent, coordinate, volatile, out)
	})
	if err != nil {
		return nil, err
	}
	return append(mismatches, body...), nil
}

// CompareFidelity checks an observed fidelity ledger against the case expectation.
//
// The ledger is the second half of a semantic comparison: typed equivalence proves
// the far side received the right content, and the ledger proves every field that
// did not survive was declared. The router emits the observed ledger, so a caller
// supplies it separately rather than reading it from the fixture tree.
func (c *Case) CompareFidelity(got map[string]FidelityAction) (Result, error) {
	result := Result{Case: c.ID, Mode: ModeSemantic}
	want := c.Expectation.Fidelity

	for _, pointer := range sortedLedgerPointers(want, got) {
		wantAction, inWant := want[pointer]
		gotAction, inGot := got[pointer]

		switch {
		case !inGot:
			result.Mismatches = append(result.Mismatches, Mismatch{
				Path: pointer, Want: string(wantAction), Got: "<absent>", Reason: "ledger entry missing",
			})
			continue
		case !inWant:
			result.Mismatches = append(result.Mismatches, Mismatch{
				Path: pointer, Want: "<absent>", Got: string(gotAction), Reason: "undeclared ledger entry",
			})
			continue
		}

		wantTier, err := wantAction.Tier()
		if err != nil {
			return Result{}, fmt.Errorf("case %q: expected fidelity %q: %w", c.ID, pointer, err)
		}
		gotTier, err := gotAction.Tier()
		if err != nil {
			return Result{}, fmt.Errorf("case %q: observed fidelity %q: %w", c.ID, pointer, err)
		}

		if wantAction == gotAction {
			continue
		}
		reason := "fidelity action differs within tier " + string(wantTier)
		if wantTier != gotTier {
			reason = fmt.Sprintf("fidelity tier differs: %s vs %s", wantTier, gotTier)
		}
		result.Mismatches = append(result.Mismatches, Mismatch{
			Path: pointer, Want: string(wantAction), Got: string(gotAction), Reason: reason,
		})
	}
	return result, nil
}

func sortedLedgerPointers(ledgers ...map[string]FidelityAction) []string {
	seen := map[string]struct{}{}
	for _, ledger := range ledgers {
		for pointer := range ledger {
			seen[pointer] = struct{}{}
		}
	}

	pointers := make([]string, 0, len(seen))
	for pointer := range seen {
		pointers = append(pointers, pointer)
	}
	sort.Strings(pointers)
	return pointers
}
