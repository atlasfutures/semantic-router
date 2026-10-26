package raylinearc

import (
	"bytes"
	"strings"
	"testing"
)

// The signals block is opaque: any object decodes and is kept byte for
// byte; anything else is not a decision. Its absence decodes as before.
func TestPolicyDecisionResponseSignals(t *testing.T) {
	fixture := readPolicyFixture(t, "decision_response.v1.json")
	plain, err := DecodePolicyDecisionResponse(fixture)
	if err != nil || plain.Signals != nil {
		t.Fatalf("plain: %v %s", err, plain.Signals)
	}
	trimmed := bytes.TrimRight(bytes.TrimSpace(fixture), "}")
	with := func(block string) []byte {
		return append(append(append([]byte(nil), trimmed...), []byte(`,"signals":`+block)...), '}')
	}
	block := `{"pii.payload":{"anything":[1,2]}}`
	response, err := DecodePolicyDecisionResponse(with(block))
	if err != nil || string(response.Signals) != block {
		t.Fatalf("signals: %v %s", err, response.Signals)
	}
	for _, bad := range []string{`null`, `[]`, `"x"`} {
		if _, err := DecodePolicyDecisionResponse(with(bad)); err == nil {
			t.Errorf("signals %s decoded", bad)
		}
	}
}

// A request without signals encodes no member; with them, the names as sent.
func TestPolicyDecisionRequestSignals(t *testing.T) {
	plain, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{})
	if err != nil || strings.Contains(string(plain), "signals") {
		t.Fatalf("plain: %v %s", err, plain)
	}
	signalled, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{Signals: []string{"topic.domain", "pii.payload"}})
	if err != nil || !strings.Contains(string(signalled), `"signals":["topic.domain","pii.payload"]`) {
		t.Fatalf("signalled: %v %s", err, signalled)
	}
}

// Only an unsupported_request whose reason is a bounded signal_* reason is
// a signal refusal; its reason is kept for the public code.
func TestPolicySignalRefusalClass(t *testing.T) {
	for reason, want := range map[string]string{
		"signal_not_supported":     PolicySignalRefusedClass,
		"signal_not_admitted":      PolicySignalRefusedClass,
		"signal_heads_unavailable": PolicySignalRefusedClass,
		"projection_refused":       "unsupported_request",
		"signal_<script>":          "unsupported_request",
	} {
		got := policyFailureClass(PolicyErrorResponse{Error: "unsupported_request", Detail: map[string]any{"reason": reason}})
		if got != want {
			t.Errorf("%s: class %s, want %s", reason, got, want)
		}
	}
	if got := policyFailureClass(PolicyErrorResponse{Error: "selection_refused", Detail: map[string]any{"reason": "signal_not_supported"}}); got == PolicySignalRefusedClass {
		t.Error("a selection refusal read as a signal refusal")
	}
}
