package conformance

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"sigs.k8s.io/yaml"
)

// CasesFile is the inventory filename inside a version root.
const CasesFile = "cases.yaml"

// Conventional per-case artifact filenames. A case directory that exists must
// contain the files its client mode and comparison modes require; see SCHEMA.md.
const (
	fileClientRequest      = "client-request.json"
	fileProviderRequest    = "expected-provider-request.json"
	fileCompareTuning      = "compare.yaml"
	fileReplayScript       = "replay.yaml"
	baseProviderResponse   = "provider-response"
	baseClientResponse     = "expected-client-response"
	extJSON                = ".json"
	extSSE                 = ".sse"
	pointerPrefix          = "/"
	rejectDispatchAttempts = 0
)

// Load reads and validates a versioned fixture tree rooted at dir, for example
// e2e/testcases/testdata/protocol-conformance/v1. It returns every declared case,
// with Fixtures populated for the cases whose directory exists.
//
// Load is strict: an unknown comparison mode, an unknown fidelity action, a
// duplicate case ID, a dangling provenance source, a missing required artifact,
// or an unresolvable replay reference is an error, not a skipped case.
func Load(dir string) (*Inventory, error) {
	raw, err := os.ReadFile(filepath.Join(dir, CasesFile))
	if err != nil {
		return nil, fmt.Errorf("conformance: read inventory: %w", err)
	}

	var inv Inventory
	if err := yaml.UnmarshalStrict(raw, &inv); err != nil {
		return nil, fmt.Errorf("conformance: parse %s: %w", CasesFile, err)
	}
	if inv.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("conformance: unsupported schema_version %q, want %q", inv.SchemaVersion, SchemaVersion)
	}

	var problems []error
	for mode := range inv.ComparisonModes {
		if !mode.Valid() {
			problems = append(problems, fmt.Errorf("comparison_modes declares unknown mode %q", mode))
		}
	}
	for _, action := range inv.FidelityActions {
		if _, err := action.Tier(); err != nil {
			problems = append(problems, fmt.Errorf("fidelity_actions: %w", err))
		}
	}

	seen := make(map[string]int, len(inv.Cases))
	for i := range inv.Cases {
		c := &inv.Cases[i]
		if first, dup := seen[c.ID]; dup {
			problems = append(problems, fmt.Errorf("case %q at index %d duplicates index %d", c.ID, i, first))
			continue
		}
		seen[c.ID] = i

		problems = append(problems, validateCase(c, &inv)...)

		fixtures, err := loadFixtures(dir, c)
		if err != nil {
			problems = append(problems, fmt.Errorf("case %q: %w", c.ID, err))
			continue
		}
		c.Fixtures = fixtures
	}

	sortErrors(problems)
	if err := errors.Join(problems...); err != nil {
		return nil, fmt.Errorf("conformance: invalid fixture tree at %s:\n%w", dir, err)
	}
	return &inv, nil
}

// Case returns the case with the given ID.
func (inv *Inventory) Case(id string) (*Case, bool) {
	for i := range inv.Cases {
		if inv.Cases[i].ID == id {
			return &inv.Cases[i], true
		}
	}
	return nil, false
}

// Tranche returns the cases belonging to one tranche, in declaration order.
func (inv *Inventory) Tranche(name string) []*Case {
	var out []*Case
	for i := range inv.Cases {
		if inv.Cases[i].Tranche == name {
			out = append(out, &inv.Cases[i])
		}
	}
	return out
}

func validateCase(c *Case, inv *Inventory) []error {
	var problems []error
	fail := func(format string, args ...any) {
		problems = append(problems, fmt.Errorf("case %q: "+format, append([]any{c.ID}, args...)...))
	}

	if c.ID == "" {
		problems = append(problems, errors.New("a case has an empty id"))
		return problems
	}
	if c.Contract == "" {
		fail("missing contract")
	}
	if c.Ownership.Primary == "" {
		fail("missing ownership.primary")
	}
	if c.Provenance.Origin == "" {
		fail("missing provenance.origin")
	}
	for _, src := range c.Provenance.Sources {
		if _, ok := inv.NormativeSources[src]; !ok {
			fail("provenance source %q is not declared in normative_sources", src)
		}
	}

	for boundary, mode := range map[string]Mode{
		"expectation.provider_request": c.Expectation.ProviderRequest,
		"expectation.client_response":  c.Expectation.ClientResponse,
	} {
		if !mode.Valid() {
			fail("%s uses unknown comparison mode %q", boundary, mode)
			continue
		}
		if _, declared := inv.ComparisonModes[mode]; !declared {
			fail("%s uses mode %q that comparison_modes does not declare", boundary, mode)
		}
	}

	problems = append(problems, validateFidelity(c)...)
	problems = append(problems, validateRejectShape(c)...)
	return problems
}

// validateFidelity enforces the tier rule from SCHEMA.md: a stateful-or-unsupported
// disposition means the semantics never crossed the boundary, so the case must name
// the exact loss or reject before dispatch. Silent loss is not expressible.
func validateFidelity(c *Case) []error {
	var problems []error
	stateful := make([]string, 0, len(c.Expectation.Fidelity))

	for pointer, action := range c.Expectation.Fidelity {
		tier, err := action.Tier()
		if err != nil {
			problems = append(problems, fmt.Errorf("case %q: fidelity %q: %w", c.ID, pointer, err))
			continue
		}
		if tier == TierStatefulOrUnsupported {
			stateful = append(stateful, pointer)
		}
	}

	if len(stateful) > 0 && !c.Expectation.DeclaresLoss() && c.Expectation.Outcome != "rejected" {
		sort.Strings(stateful)
		problems = append(problems, fmt.Errorf(
			"case %q: fidelity %v is stateful-or-unsupported but the case declares loss %q and outcome %q; name the loss or reject",
			c.ID, stateful, c.Expectation.Loss, c.Expectation.Outcome))
	}
	return problems
}

// validateRejectShape enforces that a rejection is expressed consistently: a reject
// mutation rejects at both boundaries and never dispatches.
func validateRejectShape(c *Case) []error {
	var problems []error
	rejects := []bool{
		c.MutationMode == string(ModeReject),
		c.Expectation.ProviderRequest == ModeReject,
		c.Expectation.ClientResponse == ModeReject,
		c.Expectation.DispatchAttempts == rejectDispatchAttempts,
	}
	first := rejects[0]
	for _, r := range rejects[1:] {
		if r != first {
			problems = append(problems, fmt.Errorf(
				"case %q: inconsistent rejection: mutation_mode=%q provider_request=%q client_response=%q dispatch_attempts=%d",
				c.ID, c.MutationMode, c.Expectation.ProviderRequest, c.Expectation.ClientResponse, c.Expectation.DispatchAttempts))
			break
		}
	}
	return problems
}

// loadFixtures reads a case directory. A case with no directory is contract-only and
// returns (nil, nil); a directory that exists must be complete.
func loadFixtures(root string, c *Case) (*Fixtures, error) {
	dir := filepath.Join(root, c.ID)
	info, err := os.Stat(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("stat case directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", c.ID)
	}

	f := &Fixtures{Dir: dir}
	if f.ClientRequest, err = readRequired(dir, fileClientRequest); err != nil {
		return nil, err
	}

	// A rejection never reaches the provider, so it has no provider-side artifacts.
	if c.Expectation.ProviderRequest != ModeReject {
		if f.ExpectedProviderRequest, err = readRequired(dir, fileProviderRequest); err != nil {
			return nil, err
		}
		if f.ProviderResponse, f.ProviderResponseStream, err = readEitherEncoding(dir, baseProviderResponse); err != nil {
			return nil, err
		}
	}
	if c.Expectation.ClientResponse != ModeReject {
		if f.ExpectedClientResponse, f.ExpectedClientResponseStream, err = readEitherEncoding(dir, baseClientResponse); err != nil {
			return nil, err
		}
	}
	if err := loadCaseScripts(dir, f); err != nil {
		return nil, err
	}
	return f, nil
}

// loadCaseScripts reads the two optional per-case YAML files. Both are validated
// here so a malformed script fails at load rather than mid-run.
func loadCaseScripts(dir string, f *Fixtures) error {
	raw, ok, err := readOptional(dir, fileCompareTuning)
	if err != nil {
		return err
	}
	if ok {
		if err := yaml.UnmarshalStrict(raw, &f.Compare); err != nil {
			return fmt.Errorf("parse %s: %w", fileCompareTuning, err)
		}
	}

	raw, ok, err = readOptional(dir, fileReplayScript)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}

	script, err := parseReplayScript(raw, dir)
	if err != nil {
		return err
	}
	f.Replay = script
	return nil
}

func readRequired(dir, name string) ([]byte, error) {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("missing required artifact %s", name)
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", name, err)
	}
	return raw, nil
}

func readOptional(dir, name string) ([]byte, bool, error) {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read %s: %w", name, err)
	}
	return raw, true, nil
}

// readEitherEncoding reads base.json or base.sse. Exactly one must exist: the
// encoding is the contract, so an ambiguous pair is an authoring error.
func readEitherEncoding(dir, base string) (body []byte, stream bool, err error) {
	buffered, hasBuffered, err := readOptional(dir, base+extJSON)
	if err != nil {
		return nil, false, err
	}
	sse, hasSSE, err := readOptional(dir, base+extSSE)
	if err != nil {
		return nil, false, err
	}

	switch {
	case hasBuffered && hasSSE:
		return nil, false, fmt.Errorf("%s has both %s and %s; declare exactly one encoding", base, extJSON, extSSE)
	case hasBuffered:
		return buffered, false, nil
	case hasSSE:
		return sse, true, nil
	}
	return nil, false, fmt.Errorf("missing required artifact %s%s or %s%s", base, extJSON, base, extSSE)
}

// sortErrors makes loader output stable across map iteration order.
func sortErrors(problems []error) {
	sort.SliceStable(problems, func(i, j int) bool {
		return strings.Compare(problems[i].Error(), problems[j].Error()) < 0
	})
}
