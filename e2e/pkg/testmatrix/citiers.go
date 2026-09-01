package testmatrix

import "strings"

// Cadence is the CI schedule a run belongs to. It selects how much of a profile's
// contract runs: a pull request runs the compact tier, and every other cadence runs
// the profile's own GetTestCases list.
type Cadence string

const (
	// CadencePR is the pull-request gate.
	CadencePR Cadence = "pr"
	// CadenceNightly is the scheduled full run.
	CadenceNightly Cadence = "nightly"
)

// Valid reports whether c is a declared cadence.
func (c Cadence) Valid() bool {
	switch c {
	case CadencePR, CadenceNightly:
		return true
	}
	return false
}

// EnvoyAIGatewayCITier is the subset of BaselineRouterContract that CI runs in
// every cadence. The stress and pressure coverage stays out until that suite is
// stable again; the rest of the contract is still reachable locally through the
// profile's own list.
var EnvoyAIGatewayCITier = []string{
	"chat-completions-request",
	"apiserver-runtime-config-endpoints",
	"domain-classify",
	"semantic-cache",
	"semantic-cache-polarity",
	"pii-detection",
	"jailbreak-detection",
	"decision-priority-selection",
	"plugin-chain-execution",
	"tool-selection",
	"rule-condition-logic",
	"decision-fallback-behavior",
	"plugin-config-variations",
}

// ProtocolConformanceSmoke is the compact all-protocol gate. The single testcase
// it names runs the corpus smoke tier, which covers every ingress protocol the
// router accepts and both buffered and streaming client modes. The full corpus
// runs under ProtocolConformanceContract in the nightly cadence.
var ProtocolConformanceSmoke = []string{
	"protocol-conformance-smoke",
}

// ciTiers is the single source of truth for the testcase subsets CI runs. A
// profile with no entry, or a cadence a profile does not narrow, runs the
// profile's own GetTestCases list. CI derives its -tests argument from this table
// rather than repeating testcase names in workflow YAML.
var ciTiers = map[string]map[Cadence][]string{
	"envoy-ai-gateway": {
		CadencePR:      EnvoyAIGatewayCITier,
		CadenceNightly: EnvoyAIGatewayCITier,
	},
	"protocol-conformance": {
		CadencePR: ProtocolConformanceSmoke,
	},
}

// CITier returns the testcase subset CI runs for a profile at a cadence. A nil
// result means the cadence does not narrow the profile: run its whole list.
func CITier(profile string, cadence Cadence) []string {
	return ciTiers[profile][cadence]
}

// CITierArg renders CITier as the comma-separated -tests argument, or "" when the
// cadence does not narrow the profile.
func CITierArg(profile string, cadence Cadence) string {
	return strings.Join(CITier(profile, cadence), ",")
}

// CITierProfiles returns every profile the table narrows. Tests use it to check
// the table against the live testcase registry.
func CITierProfiles() []string {
	names := make([]string, 0, len(ciTiers))
	for name := range ciTiers {
		names = append(names, name)
	}
	return names
}
