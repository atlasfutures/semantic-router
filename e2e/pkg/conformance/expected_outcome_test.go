package conformance

import "testing"

// Tests for the expected-failure markers the corpus carries, and for the
// fixture correction that keeps seed-06's remaining failure the marked one.

// TestFrozenInventoryExpectedOutcomes pins the three markers the live kind run
// justified. Each one has to name a gap a reader can chase and a signature tight
// enough that a different regression cannot hide behind it.
func TestFrozenInventoryExpectedOutcomes(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}

	// One marker left, and both retirements were the corpus's own fault rather
	// than the Router's.
	//
	// seed-02 assumed the Router could not reach an Anthropic backend. It could;
	// the profile had simply never declared api_format. A Cloud Run run on
	// 2026-08-29, before the codec work merged, already passed this case once its
	// config set the field.
	//
	// seed-06 assumed the Router synthesizes no terminal event after a provider
	// dies mid-stream. It does. The old script reset the TCP connection, and an
	// in-cluster proxy forwards a reset as a stream reset, tearing the response
	// path down before anything can be appended -- so the case measured the proxy.
	// Ending the provider body instead is the shape a real provider failure
	// arrives in, and the Router answers it with a terminal error event.
	//
	// seed-03 kept its marker but changed cause: the Chat short-circuit is gone
	// and the missing provider version header is what remains.
	want := map[string]string{
		"seed-03-responses-to-anthropic-stream": "PL-0042#seed-03-cross-protocol-anthropic-version",
	}

	marked := map[string]string{}
	for _, c := range inv.Cases {
		if c.ExpectsFailure() {
			marked[c.ID] = c.ExpectedOutcome.Reference
			if len(c.ExpectedOutcome.Signature) == 0 {
				t.Errorf("case %q is marked expected-fail with no signature", c.ID)
			}
		}
	}
	if len(marked) != len(want) {
		t.Fatalf("marked cases = %v, want exactly %v", marked, want)
	}
	for id, reference := range want {
		if marked[id] != reference {
			t.Errorf("case %q reference = %q, want %q", id, marked[id], reference)
		}
	}
}

// TestSeed06FixtureDropsTheUnemittableProviderBlock pins the fixture correction:
// nothing in the conformance profile can emit an OpenRouter routing block, and the
// router does inject stream_options on a streaming Chat request.
func TestSeed06FixtureDropsTheUnemittableProviderBlock(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}
	c, ok := inv.Case("seed-06-anthropic-openrouter-midstream-truncation")
	if !ok || !c.Loaded() {
		t.Fatal("seed-06 must be authored")
	}

	decoded, err := decodeJSON(c.Fixtures.ExpectedProviderRequest)
	if err != nil {
		t.Fatalf("decode seed-06 expected-provider-request.json: %v", err)
	}
	body, ok := decoded.(map[string]any)
	if !ok {
		t.Fatal("seed-06 expected-provider-request.json is not an object")
	}
	if _, present := body["provider"]; present {
		t.Error("seed-06 still expects an OpenRouter provider block that nothing in the profile can emit")
	}
	options, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("seed-06 must expect the router-injected stream_options")
	}
	if include, ok := options["include_usage"].(bool); !ok || !include {
		t.Errorf("stream_options.include_usage = %v, want true", options["include_usage"])
	}
}
