package testcases

import "testing"

// The deployed half of this testcase needs the anthropic-shim profile. The header
// reader does not, and it is the part that can silently rot: the shim's debug
// payload shape is the contract between the two, so pin it here rather than
// discover a mismatch inside a cluster run.
func TestProviderInboundHeadersReadsTheShimDebugPayload(t *testing.T) {
	payload := []byte(`{
	  "session_id": "s1",
	  "body": {"model": "MoM"},
	  "headers": {"Anthropic-Version": "2023-06-01", "content-type": "application/json"}
	}`)

	headers, err := providerInboundHeaders(payload)
	if err != nil {
		t.Fatalf("providerInboundHeaders() error = %v", err)
	}
	// Folded to lower case, because the recorder preserves whatever case arrived.
	if got := headers["anthropic-version"]; got != "2023-06-01" {
		t.Errorf("anthropic-version = %q, want %q", got, "2023-06-01")
	}
	if got := headers["content-type"]; got != "application/json" {
		t.Errorf("content-type = %q, want %q", got, "application/json")
	}
}

// A payload with no headers must fail loudly. Treating it as an empty set would
// make the assertion vacuous, which is the failure mode that lets a dropped
// header pass unnoticed.
func TestProviderInboundHeadersRejectsAPayloadWithNoHeaders(t *testing.T) {
	for name, payload := range map[string]string{
		"headers absent": `{"session_id": "s1", "body": {}}`,
		"headers empty":  `{"session_id": "s1", "body": {}, "headers": {}}`,
	} {
		if _, err := providerInboundHeaders([]byte(payload)); err == nil {
			t.Errorf("%s: expected an error, got none", name)
		}
	}
}

func TestProviderInboundHeadersRejectsMalformedPayload(t *testing.T) {
	if _, err := providerInboundHeaders([]byte("not json")); err == nil {
		t.Error("expected a decode error, got none")
	}
}
