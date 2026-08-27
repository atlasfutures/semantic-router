package conformance

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// frozenTree is the versioned fixture tree this loader is the reader for.
const frozenTree = "../../testcases/testdata/protocol-conformance/v1"

func TestLoadFrozenInventory(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}

	if got, want := len(inv.Cases), 31; got != want {
		t.Errorf("case count = %d, want %d", got, want)
	}
	if got, want := len(inv.Tranche("first-six")), 6; got != want {
		t.Errorf("first-six tranche = %d cases, want %d", got, want)
	}

	seed, ok := inv.Case("seed-01-chat-identity-model-patch")
	if !ok {
		t.Fatal("seed-01-chat-identity-model-patch is absent")
	}
	if seed.Expectation.ProviderRequest != ModeExactExcept {
		t.Errorf("seed-01 provider_request = %q, want %q", seed.Expectation.ProviderRequest, ModeExactExcept)
	}
	if got, want := seed.Expectation.Fidelity["/model"], ActionPatched; got != want {
		t.Errorf("seed-01 fidelity /model = %q, want %q", got, want)
	}
	if !seed.Loaded() {
		t.Error("seed-01 reports no fixtures, but its case directory is authored")
	}

	// The whole promoted tranche is authored, so every one of its cases must carry
	// artifacts. A case that silently lost its directory would otherwise be reported
	// as a skip by the runner rather than as the regression it is.
	for _, c := range inv.Tranche("first-six") {
		if !c.Loaded() {
			t.Errorf("case %q reports no fixtures, but the first-six tranche is authored", c.ID)
		}
	}
}

// TestFrozenInventoryFidelityTiers pins the three-tier vocabulary against the real
// corpus: every declared action maps to a tier, and no case can lose semantics
// without naming the loss.
func TestFrozenInventoryFidelityTiers(t *testing.T) {
	inv, err := Load(frozenTree)
	if err != nil {
		t.Fatalf("load frozen inventory: %v", err)
	}

	tiers := map[FidelityTier]int{}
	for _, c := range inv.Cases {
		for pointer, action := range c.Expectation.Fidelity {
			tier, err := action.Tier()
			if err != nil {
				t.Fatalf("case %q pointer %q: %v", c.ID, pointer, err)
			}
			tiers[tier]++
		}
	}
	for _, tier := range []FidelityTier{TierLossless, TierVisibleNotEchoable, TierStatefulOrUnsupported} {
		if tiers[tier] == 0 {
			t.Errorf("tier %q has no entry in the frozen corpus", tier)
		}
	}
}

func TestFidelityActionTier(t *testing.T) {
	tests := []struct {
		action FidelityAction
		want   FidelityTier
		bad    bool
	}{
		{action: ActionPreserved, want: TierLossless},
		{action: ActionPatched, want: TierLossless},
		{action: ActionMapped, want: TierLossless},
		{action: ActionSynthesized, want: TierVisibleNotEchoable},
		{action: ActionCoerced, want: TierVisibleNotEchoable},
		{action: ActionOmitted, want: TierStatefulOrUnsupported},
		{action: ActionRejected, want: TierStatefulOrUnsupported},
		{action: "dropped", bad: true},
	}

	for _, tt := range tests {
		t.Run(string(tt.action), func(t *testing.T) {
			got, err := tt.action.Tier()
			if tt.bad {
				if err == nil {
					t.Fatalf("Tier() = %q, want an error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Tier() error: %v", err)
			}
			if got != tt.want {
				t.Errorf("Tier() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLoadValidationFailures(t *testing.T) {
	tests := []struct {
		name    string
		cases   string
		wantErr string
	}{
		{
			name:    "unknown comparator mode",
			cases:   caseYAML(map[string]string{"provider_request": "almost-exact"}),
			wantErr: `unknown comparison mode "almost-exact"`,
		},
		{
			name:    "unknown fidelity action",
			cases:   caseYAML(map[string]string{"fidelity": "{/model: shortened}"}),
			wantErr: `unknown fidelity action "shortened"`,
		},
		{
			name:    "undeclared provenance source",
			cases:   caseYAML(map[string]string{"sources": "[made_up_doc]"}),
			wantErr: `provenance source "made_up_doc" is not declared`,
		},
		{
			name:    "silent stateful loss",
			cases:   caseYAML(map[string]string{"fidelity": "{/thinking: omitted}"}),
			wantErr: "stateful-or-unsupported but the case declares loss",
		},
		{
			name:    "half-declared rejection",
			cases:   caseYAML(map[string]string{"mutation_mode": "reject"}),
			wantErr: "inconsistent rejection",
		},
		{
			name:    "duplicate case id",
			cases:   caseYAML(nil) + caseYAML(nil),
			wantErr: "duplicates index 0",
		},
		{
			name:    "unknown case field",
			cases:   caseYAML(map[string]string{"__extra": "surprise: true"}),
			wantErr: "unknown field",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := writeTree(t, tt.cases)
			_, err := Load(dir)
			if err == nil {
				t.Fatal("Load() succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Load() error = %v\nwant it to contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestLoadRejectsWrongSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, CasesFile), "schema_version: dpc-003-inventory-v2\ncases: []\n")

	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "unsupported schema_version") {
		t.Fatalf("Load() error = %v, want an unsupported schema_version error", err)
	}
}

func TestLoadFixtureDirectory(t *testing.T) {
	dir := writeTree(t, caseYAML(nil))
	caseDir := filepath.Join(dir, "unit-01")
	if err := os.MkdirAll(caseDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(caseDir, "client-request.json"), `{"model":"a"}`)

	// A directory that exists must be complete: the provider artifacts are missing.
	_, err := Load(dir)
	if err == nil || !strings.Contains(err.Error(), "missing required artifact expected-provider-request.json") {
		t.Fatalf("Load() error = %v, want a missing-artifact error", err)
	}

	write(t, filepath.Join(caseDir, "expected-provider-request.json"), `{"model":"b"}`)
	write(t, filepath.Join(caseDir, "provider-response.json"), `{"id":"r1"}`)
	write(t, filepath.Join(caseDir, "expected-client-response.sse"), "event: ping\ndata: {}\n\n")

	inv, err := Load(dir)
	if err != nil {
		t.Fatalf("Load() error = %v, want success", err)
	}
	c, _ := inv.Case("unit-01")
	if !c.Loaded() {
		t.Fatal("case reports no fixtures after its directory was authored")
	}
	if c.Fixtures.ProviderResponseStream {
		t.Error("provider-response.json was read as a stream")
	}
	if !c.Fixtures.ExpectedClientResponseStream {
		t.Error("expected-client-response.sse was not read as a stream")
	}
}

func TestLoadRejectsAmbiguousEncoding(t *testing.T) {
	dir := writeTree(t, caseYAML(nil))
	caseDir := filepath.Join(dir, "unit-01")
	if err := os.MkdirAll(caseDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(caseDir, "client-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "expected-provider-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "provider-response.json"), `{}`)
	write(t, filepath.Join(caseDir, "provider-response.sse"), "data: {}\n\n")
	write(t, filepath.Join(caseDir, "expected-client-response.json"), `{}`)

	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "declare exactly one encoding") {
		t.Fatalf("Load() error = %v, want an ambiguous-encoding error", err)
	}
}

// caseYAML renders one minimal valid case, with the named fields overridden. An
// "__extra" override is appended verbatim so a test can inject an unknown field.
func caseYAML(override map[string]string) string {
	field := func(name, fallback string) string {
		if value, ok := override[name]; ok {
			return value
		}
		return fallback
	}

	body := `
  - id: unit-01
    tranche: first-six
    contract: A unit fixture.
    client: {protocol: openai-chat, path: /v1/chat/completions, mode: buffered}
    provider: {protocol: openai-chat, dialect: openai}
    mutation_mode: ` + field("mutation_mode", "patch") + `
    features: [usage]
    synthetic_shape: A short text request.
    expectation:
      outcome: supported
      provider_request: ` + field("provider_request", "exact-except") + `
      client_response: ` + field("client_response", "exact-except") + `
      allowed_patches: [/model]
      fidelity: ` + field("fidelity", "{/model: patched}") + `
      loss: none
      dispatch_attempts: 1
    provenance:
      origin: unit
      sources: ` + field("sources", "[openai_chat]") + `
    ownership:
      primary: Data Plane & Networking
      reviewers: [Evaluation & Quality]
      upstream_issues: [1138]
`
	if extra, ok := override["__extra"]; ok {
		body += "    " + extra + "\n"
	}
	return body
}

func writeTree(t *testing.T, cases string) string {
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
cases:` + cases

	write(t, filepath.Join(dir, CasesFile), inventory)
	return dir
}

func write(t *testing.T, path, content string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
