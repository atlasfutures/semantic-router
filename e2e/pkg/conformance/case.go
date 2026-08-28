// Package conformance loads the versioned protocol-conformance fixture tree and
// compares observed wire artifacts against its expectations.
//
// The tree and its vocabulary are defined by
// e2e/testcases/testdata/protocol-conformance/v1/SCHEMA.md. This package owns the
// schema (case.go), the loader (load.go), the provider replay script (replay.go),
// SSE parsing (sse.go), the comparator API (compare.go), and the four comparison
// modes (modes.go, semantic.go, jsondiff.go). It does not run the provider fixture
// server and does not drive the router; DPC-101 and DPC-103 own those.
package conformance

import (
	"fmt"
	"strings"
)

// SchemaVersion is the only cases.yaml schema_version this loader accepts.
const SchemaVersion = "dpc-003-inventory-v1alpha1"

// Mode is a comparison mode applied to one wire boundary.
type Mode string

const (
	// ModeExact requires raw byte identity of the body plus the declared headers.
	ModeExact Mode = "exact"
	// ModeExactExcept requires structural identity except declared JSON pointers and header names.
	ModeExactExcept Mode = "exact-except"
	// ModeSemantic requires typed block or event equivalence plus an exact fidelity ledger.
	ModeSemantic Mode = "semantic"
	// ModeReject requires a typed status, error body, and zero dispatch.
	ModeReject Mode = "reject"
)

// Valid reports whether m is a declared comparison mode.
func (m Mode) Valid() bool {
	switch m {
	case ModeExact, ModeExactExcept, ModeSemantic, ModeReject:
		return true
	}
	return false
}

// FidelityAction is one cases.yaml disposition for a single field or block.
type FidelityAction string

const (
	ActionPreserved   FidelityAction = "preserved"
	ActionPatched     FidelityAction = "patched"
	ActionMapped      FidelityAction = "mapped"
	ActionSynthesized FidelityAction = "synthesized"
	ActionCoerced     FidelityAction = "coerced"
	ActionOmitted     FidelityAction = "omitted"
	ActionRejected    FidelityAction = "rejected"
)

// FidelityTier is the coarse disposition every ledger entry carries. It groups the
// seven cases.yaml actions by what a test may assert about the value afterwards.
type FidelityTier string

const (
	// TierLossless means the semantics crossed the boundary intact, so a later turn
	// may replay the value in its original carrier.
	TierLossless FidelityTier = "lossless"
	// TierVisibleNotEchoable means the value is observable in the response but was not
	// client-authored, so it must not be fed back into a later request as if it were.
	TierVisibleNotEchoable FidelityTier = "visible-but-not-echoable"
	// TierStatefulOrUnsupported means the semantics did not cross the boundary at all.
	// The case must name the exact loss or reject before dispatch.
	TierStatefulOrUnsupported FidelityTier = "stateful-or-unsupported"
)

// Tier maps a cases.yaml fidelity action onto its tier. The mapping is fixed by
// SCHEMA.md; see its "Fidelity tiers" section for the rationale of each row.
func (a FidelityAction) Tier() (FidelityTier, error) {
	switch a {
	case ActionPreserved, ActionPatched, ActionMapped:
		return TierLossless, nil
	case ActionSynthesized, ActionCoerced:
		return TierVisibleNotEchoable, nil
	case ActionOmitted, ActionRejected:
		return TierStatefulOrUnsupported, nil
	}
	return "", fmt.Errorf("unknown fidelity action %q", a)
}

// LossNone is the cases.yaml spelling for "this case declares no semantic loss".
const LossNone = "none"

// Inventory is a parsed cases.yaml.
type Inventory struct {
	SchemaVersion     string            `json:"schema_version"`
	Status            string            `json:"status"`
	Scope             map[string]string `json:"scope"`
	PublicationPolicy map[string]string `json:"publication_policy"`
	ComparisonModes   map[Mode]string   `json:"comparison_modes"`
	FidelityActions   []FidelityAction  `json:"fidelity_actions"`
	NormativeSources  map[string]string `json:"normative_sources"`

	// SmokeTier names the promoted cases that CI runs on pull requests. It is the
	// smallest subset that still covers every ingress protocol and both client
	// modes; the loader rejects a tier that names an unknown or unpromoted case.
	SmokeTier []string `json:"smoke_tier"`

	Cases []Case `json:"cases"`
}

// Case is one conformance contract: a client protocol, a provider route, and the
// expectations both wire boundaries must satisfy.
type Case struct {
	ID             string      `json:"id"`
	Tranche        string      `json:"tranche"`
	Contract       string      `json:"contract"`
	Client         ClientSpec  `json:"client"`
	Provider       RouteSpec   `json:"provider"`
	MutationMode   string      `json:"mutation_mode"`
	Features       []string    `json:"features"`
	SyntheticShape string      `json:"synthetic_shape"`
	Expectation    Expectation `json:"expectation"`
	Provenance     Provenance  `json:"provenance"`
	Ownership      Ownership   `json:"ownership"`

	// ExpectedOutcome marks the case as a known failure against a named router
	// gap. Absent means the case is expected to pass, which is every case that
	// does not name one.
	ExpectedOutcome *ExpectedOutcome `json:"expected_outcome,omitempty"`

	// Fixtures holds the per-case artifact files. It is nil until DPC-104 authors
	// the case directory; Loaded reports which state a case is in.
	Fixtures *Fixtures `json:"-"`
}

// Loaded reports whether the case directory exists and its fixtures were read.
// Cases in the frozen inventory that have no directory yet are contract-only.
func (c *Case) Loaded() bool { return c.Fixtures != nil }

// OutcomeFail is the only expected_outcome status v1 declares. A case that is
// expected to pass says so by carrying no marker at all.
const OutcomeFail = "fail"

// ExpectedOutcome records that a case is known to fail, why, and how the failure
// is recognized.
//
// The marker exists so a green run stays honest. A run that reports a marked case
// as an expected failure has still proved something: the gap is exactly the one
// the marker names. A marked case that passes, or that fails some other way, fails
// the run, because either is news.
type ExpectedOutcome struct {
	// Status is always OutcomeFail in v1.
	Status string `json:"status"`
	// Reference identifies the gap: an upstream issue, or a stable plan-document
	// token when no upstream issue covers it yet.
	Reference string `json:"reference"`
	// Reason is one sentence on what the router does instead.
	Reason string `json:"reason"`
	// Signature holds substrings that must ALL appear among the case's failure
	// messages for the failure to count as the known one. It is what keeps a
	// different regression from hiding behind the marker.
	Signature []string `json:"signature"`
}

// ExpectsFailure reports whether the case carries an expected-failure marker.
func (c *Case) ExpectsFailure() bool {
	return c.ExpectedOutcome != nil && c.ExpectedOutcome.Status == OutcomeFail
}

// UnmatchedSignature returns the signature substrings that no failure message
// contains. An empty result means the observed failure is the known one.
func (e *ExpectedOutcome) UnmatchedSignature(failures []string) []string {
	var missing []string
	for _, want := range e.Signature {
		found := false
		for _, failure := range failures {
			if strings.Contains(failure, want) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, want)
		}
	}
	return missing
}

// validateExpectedOutcome keeps an expected-failure marker reviewable and tight.
// A marker with no reference cannot be chased, a marker with no reason cannot be
// reviewed, and a marker with no signature would let any regression hide behind it.
func validateExpectedOutcome(c *Case) []error {
	marker := c.ExpectedOutcome
	if marker == nil {
		return nil
	}

	var problems []error
	fail := func(format string, args ...any) {
		problems = append(problems, fmt.Errorf("case %q: expected_outcome "+format, append([]any{c.ID}, args...)...))
	}

	if marker.Status != OutcomeFail {
		fail("status is %q; the only status v1 declares is %q, and an expected pass carries no marker", marker.Status, OutcomeFail)
	}
	if strings.TrimSpace(marker.Reference) == "" {
		fail("names no reference; cite the upstream issue or a stable plan-document token")
	}
	if strings.TrimSpace(marker.Reason) == "" {
		fail("names no reason; say in one sentence what the router does instead")
	}
	if len(marker.Signature) == 0 {
		fail("names no signature; without one any regression would count as the known failure")
	}
	for i, entry := range marker.Signature {
		if strings.TrimSpace(entry) == "" {
			fail("signature entry %d is empty", i)
		}
	}
	return problems
}

// ClientSpec is the inbound half of a case.
type ClientSpec struct {
	Protocol string `json:"protocol"`
	Path     string `json:"path"`
	Mode     string `json:"mode"`
}

// RouteSpec is the selected provider wire protocol and dialect profile.
type RouteSpec struct {
	Protocol string `json:"protocol"`
	Dialect  string `json:"dialect"`
}

// Expectation is what both boundaries must satisfy for the case to pass.
type Expectation struct {
	Outcome          string                    `json:"outcome"`
	ProviderRequest  Mode                      `json:"provider_request"`
	ClientResponse   Mode                      `json:"client_response"`
	AllowedPatches   []string                  `json:"allowed_patches"`
	Invariants       []Invariant               `json:"invariants"`
	Fidelity         map[string]FidelityAction `json:"fidelity"`
	Loss             string                    `json:"loss"`
	DispatchAttempts int                       `json:"dispatch_attempts"`
}

// DeclaresLoss reports whether the case names a semantic loss.
func (e Expectation) DeclaresLoss() bool { return e.Loss != "" && e.Loss != LossNone }

// Provenance records where the case came from and which public sources authorize it.
type Provenance struct {
	Origin           string   `json:"origin"`
	Sources          []string `json:"sources"`
	PrivateReference string   `json:"private_reference"`
	Note             string   `json:"note"`
}

// Ownership records the accountable Workgroup and the upstream issues the case serves.
type Ownership struct {
	Primary        string   `json:"primary"`
	Reviewers      []string `json:"reviewers"`
	UpstreamIssues []int    `json:"upstream_issues"`
}

// Fixtures are the artifact files read from a case directory.
type Fixtures struct {
	// Dir is the path to the case directory.
	Dir string

	// ClientRequest is the inbound request body sent to the router.
	ClientRequest []byte
	// ExpectedProviderRequest is the request body the provider fixture must observe.
	ExpectedProviderRequest []byte
	// ProviderResponse is what the provider fixture replays. Streaming reports its encoding.
	ProviderResponse []byte
	// ExpectedClientResponse is what the router must return to the client.
	ExpectedClientResponse []byte

	// ProviderResponseStream is true when provider-response is an SSE stream.
	ProviderResponseStream bool
	// ExpectedClientResponseStream is true when expected-client-response is an SSE stream.
	ExpectedClientResponseStream bool

	// Compare is the optional per-case comparator tuning from compare.yaml.
	Compare CompareTuning
	// Replay is the provider replay script from replay.yaml.
	Replay *ReplayScript
}

// CompareTuning is the optional compare.yaml. It carries only what cases.yaml
// cannot: comparator exclusions and matchers that depend on the authored bytes.
type CompareTuning struct {
	// ExcludeExtra adds exclusions to exact-except beyond expectation.allowed_patches.
	// A leading "/" makes an entry a JSON pointer; anything else is a header name.
	ExcludeExtra []string `json:"exclude_extra"`
	// Volatile lists JSON pointers whose value is nondeterministic under semantic
	// mode. Both sides must carry the pointer with the same JSON type, not the same value.
	Volatile []string `json:"volatile"`
	// RejectStatus is the HTTP status a reject case must return.
	RejectStatus int `json:"reject_status"`
	// RejectHeaders are header names and values a reject case must return.
	RejectHeaders map[string]string `json:"reject_headers"`
	// RejectBody is the expected error body, compared as exact-except with Volatile excluded.
	RejectBody map[string]any `json:"reject_body"`
}
