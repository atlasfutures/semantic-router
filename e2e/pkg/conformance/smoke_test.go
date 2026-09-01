package conformance

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestFrozenSmokeTierCoversEveryIngressProtocol pins what makes the PR gate worth
// gating on: the compact tier still touches every ingress protocol the router
// accepts and both client modes. Narrowing the tier past that point is the failure
// this test exists to catch.
func TestFrozenSmokeTierCoversEveryIngressProtocol(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}

	smoke := inv.Smoke()
	if len(smoke) == 0 {
		t.Fatal("the frozen inventory declares no smoke tier")
	}

	protocols := map[string]bool{}
	modes := map[string]bool{}
	for _, c := range smoke {
		if !c.Loaded() {
			t.Errorf("smoke case %q reports no fixtures, so the gate would skip it", c.ID)
		}
		protocols[c.Client.Protocol] = true
		modes[c.Client.Mode] = true
	}

	for _, want := range []string{"openai-chat", "anthropic-messages", "openai-responses"} {
		if !protocols[want] {
			t.Errorf("smoke tier does not exercise ingress protocol %q", want)
		}
	}
	for _, want := range []string{"buffered", "streaming"} {
		if !modes[want] {
			t.Errorf("smoke tier does not exercise client mode %q", want)
		}
	}

	// Compactness is the other half of the contract: a tier that grew to the whole
	// tranche is no longer a distinct PR gate.
	if got, full := len(smoke), len(inv.Tranche(PromotedTranche)); got >= full {
		t.Errorf("smoke tier holds %d of %d promoted cases; it is no longer compact", got, full)
	}
}

func TestLoadRejectsUnknownSmokeCase(t *testing.T) {
	dir := writeSmokeTree(t, []string{"unit-99"})

	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), `smoke_tier names unknown case "unit-99"`) {
		t.Fatalf("Load() error = %v, want an unknown-smoke-case error", err)
	}
}

func TestLoadRejectsUnpromotedSmokeCase(t *testing.T) {
	dir := writeSmokeTree(t, []string{"unit-02"})

	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "only the \"first-six\" tranche is promoted") {
		t.Fatalf("Load() error = %v, want an unpromoted-smoke-case error", err)
	}
}

func TestLoadRejectsDuplicateSmokeCase(t *testing.T) {
	dir := writeSmokeTree(t, []string{"unit-01", "unit-01"})

	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), `smoke_tier lists case "unit-01" twice`) {
		t.Fatalf("Load() error = %v, want a duplicate-smoke-case error", err)
	}
}

// writeSmokeTree renders a two-case inventory, one promoted and one deferred, with
// the given smoke tier.
func writeSmokeTree(t *testing.T, smoke []string) string {
	t.Helper()

	dir := t.TempDir()
	inventory := `schema_version: ` + SchemaVersion + `
status: unit
comparison_modes:
  exact: Raw bytes identical.
  exact-except: Identical except declared exclusions.
  semantic: Typed equivalence.
  reject: Typed rejection.
fidelity_actions: [preserved, patched, mapped, synthesized, coerced, omitted, rejected]
normative_sources:
  openai_chat: https://platform.openai.com/docs/api-reference/chat/create
smoke_tier: [` + strings.Join(smoke, ", ") + `]
cases:` + smokeCaseYAML("unit-01", PromotedTranche) + smokeCaseYAML("unit-02", "deferred")

	write(t, filepath.Join(dir, CasesFile), inventory)
	return dir
}

func smokeCaseYAML(id, tranche string) string {
	return `
  - id: ` + id + `
    tranche: ` + tranche + `
    contract: A unit fixture.
    client: {protocol: openai-chat, path: /v1/chat/completions, mode: buffered}
    provider: {protocol: openai-chat, dialect: openai}
    mutation_mode: patch
    features: [usage]
    synthetic_shape: A short text request.
    expectation:
      outcome: supported
      provider_request: exact-except
      client_response: exact-except
      allowed_patches: [/model]
      fidelity: {/model: patched}
      loss: none
      dispatch_attempts: 1
    provenance:
      origin: unit
      sources: [openai_chat]
    ownership:
      primary: Data Plane & Networking
      reviewers: [Evaluation & Quality]
      upstream_issues: [1138]
`
}
