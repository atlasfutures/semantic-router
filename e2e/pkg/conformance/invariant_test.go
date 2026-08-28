package conformance

import "testing"

// Tests for the enforced invariants and the closed invariant vocabulary. The
// rules themselves live in invariant.go.

// The enforced invariants. Each one names an encoding the public API contracts
// declare interchangeable, so the table below pins both halves: the difference the
// invariant erases, and the difference it must still report.
func TestCompareEnforcedInvariants(t *testing.T) {
	const (
		compactArguments = `{"messages":[{"role":"assistant","tool_calls":[{"id":"t1","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}]}]}`
		spacedArguments  = `{"messages":[{"role":"assistant","tool_calls":[{"id":"t1","type":"function","function":{"name":"add","arguments":"{ \"b\": 3, \"a\": 2 }"}}]}]}`
		changedArguments = `{"messages":[{"role":"assistant","tool_calls":[{"id":"t1","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":4}"}}]}]}`

		stringContent = `{"messages":[{"role":"user","content":"Add 2 and 3."}]}`
		partContent   = `{"messages":[{"role":"user","content":[{"type":"text","text":"Add 2 and 3."}]}]}`
		markedContent = `{"messages":[{"role":"user","content":[{"type":"text","text":"Add 2 and 3.","cache_control":{"type":"ephemeral"}}]}]}`

		explicitNull = `{"messages":[{"role":"assistant","content":null,"tool_calls":[]}]}`
		omittedNull  = `{"messages":[{"role":"assistant","tool_calls":[]}]}`
	)

	semantic := func(invariants ...Invariant) Comparison {
		return Comparison{Mode: ModeSemantic, Invariants: invariants}
	}

	runCompareCases(t, []compareCase{
		{
			name:      "spacing inside a tool-call arguments string fails without the invariant",
			cmp:       semantic(),
			want:      Payload{Body: []byte(compactArguments)},
			got:       Payload{Body: []byte(spacedArguments)},
			wantPaths: []string{"/messages/0/tool_calls/0/function/arguments"},
		},
		{
			name: "argument-json-equivalence compares the arguments string as JSON",
			cmp:  semantic(InvariantArgumentJSON),
			want: Payload{Body: []byte(compactArguments)},
			got:  Payload{Body: []byte(spacedArguments)},
		},
		{
			name:      "argument-json-equivalence still reports a changed argument value",
			cmp:       semantic(InvariantArgumentJSON),
			want:      Payload{Body: []byte(compactArguments)},
			got:       Payload{Body: []byte(changedArguments)},
			wantPaths: []string{"/messages/0/tool_calls/0/function/arguments/b"},
		},
		{
			name:      "a bare string content fails against a text part without the invariant",
			cmp:       semantic(),
			want:      Payload{Body: []byte(stringContent)},
			got:       Payload{Body: []byte(partContent)},
			wantPaths: []string{"/messages/0/content"},
		},
		{
			name: "content-encoding-equivalence accepts a bare string for a single text part",
			cmp:  semantic(InvariantContentEncoding),
			want: Payload{Body: []byte(stringContent)},
			got:  Payload{Body: []byte(partContent)},
		},
		{
			name:      "content-encoding-equivalence still reports a part that carries more than text",
			cmp:       semantic(InvariantContentEncoding),
			want:      Payload{Body: []byte(stringContent)},
			got:       Payload{Body: []byte(markedContent)},
			wantPaths: []string{"/messages/0/content/0/cache_control"},
		},
		{
			name:      "content-encoding-equivalence leaves an Anthropic block list alone",
			cmp:       semantic(InvariantContentEncoding),
			want:      Payload{Body: []byte(`{"content":[{"type":"text","text":"hi"}]}`)},
			got:       Payload{Body: []byte(`{"content":"hi"}`)},
			wantPaths: []string{"/content"},
		},
		{
			name:      "an explicit null fails against an omitted key without the invariant",
			cmp:       semantic(),
			want:      Payload{Body: []byte(explicitNull)},
			got:       Payload{Body: []byte(omittedNull)},
			wantPaths: []string{"/messages/0/content"},
		},
		{
			name: "null-vs-omitted-equivalence accepts an omitted key for an expected null",
			cmp:  semantic(InvariantNullVsOmitted),
			want: Payload{Body: []byte(explicitNull)},
			got:  Payload{Body: []byte(omittedNull)},
		},
		{
			name: "null-vs-omitted-equivalence accepts an emitted null for an omitted key",
			cmp:  semantic(InvariantNullVsOmitted),
			want: Payload{Body: []byte(omittedNull)},
			got:  Payload{Body: []byte(explicitNull)},
		},
		{
			name:      "null-vs-omitted-equivalence still reports a dropped value",
			cmp:       semantic(InvariantNullVsOmitted),
			want:      Payload{Body: []byte(`{"temperature":0,"top_p":null}`)},
			got:       Payload{Body: []byte(`{}`)},
			wantPaths: []string{"/temperature"},
		},
	})
}

// TestCaseComparisonCarriesOnlyEnforcedInvariants keeps the comparator from acting
// on a name the corpus declares for documentation. A covered invariant is held by
// the authored artifacts, so handing it to the comparator would be a second,
// invisible source of relaxation.
func TestCaseComparisonCarriesOnlyEnforcedInvariants(t *testing.T) {
	c := &Case{
		ID: "unit-01",
		Expectation: Expectation{
			ProviderRequest: ModeSemantic,
			ClientResponse:  ModeSemantic,
			Invariants:      []Invariant{"tool-id-pairing", InvariantArgumentJSON, "ordered-blocks"},
		},
	}

	cmp, err := c.Comparison(BoundaryProviderRequest)
	if err != nil {
		t.Fatalf("Comparison() error = %v", err)
	}
	if len(cmp.Invariants) != 1 || cmp.Invariants[0] != InvariantArgumentJSON {
		t.Fatalf("resolved invariants = %v, want only %q", cmp.Invariants, InvariantArgumentJSON)
	}
}

// TestLoadAcceptsADeferredInvariantOutsideThePromotedTranche is the other half of
// the promoted-case rule: a name nothing asserts yet is legal on a contract-only
// case, which is how the deferred tranche keeps declaring its intent.
func TestLoadAcceptsADeferredInvariantOutsideThePromotedTranche(t *testing.T) {
	dir := writeTree(t, caseYAML(map[string]string{
		"tranche":    "firebase-derived-deferred",
		"invariants": "[cache-read-preserved]",
	}))

	if _, err := Load(dir); err != nil {
		t.Fatalf("Load() error = %v, want success", err)
	}
}

// TestFrozenInventoryInvariantVocabulary pins the whole corpus against the closed
// vocabulary: every declared name is known, and no promoted case carries one that
// nothing asserts.
func TestFrozenInventoryInvariantVocabulary(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}

	enforced := 0
	for _, c := range inv.Cases {
		for _, invariant := range c.Expectation.Invariants {
			support, known := invariant.Support()
			if !known {
				t.Errorf("case %q declares unknown invariant %q", c.ID, invariant)
				continue
			}
			if support == SupportDeferred && c.Tranche == PromotedTranche {
				t.Errorf("promoted case %q declares deferred invariant %q", c.ID, invariant)
			}
			if support == SupportEnforced {
				enforced++
			}
		}
	}
	if enforced == 0 {
		t.Error("no case declares an enforced invariant; the enforcement path is unreachable from the corpus")
	}
}
