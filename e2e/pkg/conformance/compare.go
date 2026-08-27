package conformance

import (
	"errors"
	"fmt"
	"strings"
)

// This file is the comparator API the caller sees. The four mode implementations
// live in modes.go.

// Boundary names one of the two wire boundaries a case constrains.
type Boundary string

const (
	// BoundaryProviderRequest is the request the router sends to the provider.
	BoundaryProviderRequest Boundary = "provider-request"
	// BoundaryClientResponse is the response the router returns to the client.
	BoundaryClientResponse Boundary = "client-response"
)

// Payload is one observed or expected wire artifact.
//
// Status is 0 when the artifact has no status of its own, such as a captured
// request; a zero expected status means the comparator does not constrain it.
type Payload struct {
	Status  int
	Headers map[string]string
	Body    []byte
	// Stream marks Body as an SSE stream, compared as a parsed event sequence.
	Stream bool
}

// Comparison is a resolved comparison contract for one boundary. Case.Comparison
// builds it from cases.yaml and compare.yaml; callers may build one directly to
// compare artifacts the fixture tree does not own.
type Comparison struct {
	Mode Mode
	// Exclude applies to ModeExactExcept. An entry starting with "/" is an RFC 6901
	// JSON Pointer removed from both bodies; anything else is a header name skipped
	// during header comparison.
	Exclude []string
	// Volatile applies to ModeSemantic and ModeReject. Each entry is a JSON Pointer
	// whose value is nondeterministic: both sides must carry the same JSON type.
	Volatile []string
	// Reject is the expected rejection. It is required by ModeReject and unused otherwise.
	Reject RejectSpec
}

// RejectSpec is the typed rejection a ModeReject boundary must produce.
type RejectSpec struct {
	Status  int
	Headers map[string]string
	Body    map[string]any
}

// Mismatch is one difference between the expected and observed artifact.
type Mismatch struct {
	// Path is a JSON Pointer, a header name, or an SSE event coordinate such as "event[2].data".
	Path   string
	Want   string
	Got    string
	Reason string
}

func (m Mismatch) String() string {
	return fmt.Sprintf("%s: %s (want %s, got %s)", m.Path, m.Reason, m.Want, m.Got)
}

// Result is the outcome of comparing one boundary.
type Result struct {
	Case       string
	Boundary   Boundary
	Mode       Mode
	Mismatches []Mismatch
}

// Pass reports whether the boundary matched its expectation.
func (r Result) Pass() bool { return len(r.Mismatches) == 0 }

// Err returns a reportable error listing every mismatch, or nil when the boundary passed.
func (r Result) Err() error {
	if r.Pass() {
		return nil
	}

	lines := make([]string, 0, len(r.Mismatches)+1)
	lines = append(lines, fmt.Sprintf("case %q boundary %s mode %s: %d mismatch(es)", r.Case, r.Boundary, r.Mode, len(r.Mismatches)))
	for _, m := range r.Mismatches {
		lines = append(lines, "  "+m.String())
	}
	return errors.New(strings.Join(lines, "\n"))
}

// Comparison resolves the comparison contract for one boundary of a loaded case.
//
// Exclusions come from expectation.allowed_patches plus compare.yaml's exclude_extra;
// volatile pointers and the rejection shape come from compare.yaml, which is where a
// fixture author records what the authored bytes cannot express.
func (c *Case) Comparison(boundary Boundary) (Comparison, error) {
	mode := c.Expectation.ClientResponse
	if boundary == BoundaryProviderRequest {
		mode = c.Expectation.ProviderRequest
	}
	if !mode.Valid() {
		return Comparison{}, fmt.Errorf("case %q: boundary %s has unknown mode %q", c.ID, boundary, mode)
	}

	cmp := Comparison{Mode: mode}
	tuning := CompareTuning{}
	if c.Fixtures != nil {
		tuning = c.Fixtures.Compare
	}

	switch mode {
	case ModeExactExcept:
		cmp.Exclude = append(append([]string{}, c.Expectation.AllowedPatches...), tuning.ExcludeExtra...)
	case ModeSemantic:
		cmp.Volatile = tuning.Volatile
	case ModeReject:
		cmp.Volatile = tuning.Volatile
		cmp.Reject = RejectSpec{Status: tuning.RejectStatus, Headers: tuning.RejectHeaders, Body: tuning.RejectBody}
		if cmp.Reject.Status == 0 {
			return Comparison{}, fmt.Errorf("case %q: boundary %s is %q but compare.yaml declares no reject_status", c.ID, boundary, ModeReject)
		}
	case ModeExact:
		if len(c.Expectation.AllowedPatches) > 0 || len(tuning.ExcludeExtra) > 0 {
			return Comparison{}, fmt.Errorf("case %q: boundary %s is %q, which admits no exclusions; use %q", c.ID, boundary, ModeExact, ModeExactExcept)
		}
	}
	return cmp, nil
}

// CompareProviderRequest compares the request the provider fixture observed against
// the case's expected-provider-request artifact.
func (c *Case) CompareProviderRequest(got Payload) (Result, error) {
	return c.compareBoundary(BoundaryProviderRequest, got)
}

// CompareClientResponse compares the response the router returned against the case's
// expected-client-response artifact.
func (c *Case) CompareClientResponse(got Payload) (Result, error) {
	return c.compareBoundary(BoundaryClientResponse, got)
}

func (c *Case) compareBoundary(boundary Boundary, got Payload) (Result, error) {
	cmp, err := c.Comparison(boundary)
	if err != nil {
		return Result{}, err
	}

	want := Payload{}
	if cmp.Mode != ModeReject {
		if !c.Loaded() {
			return Result{}, fmt.Errorf("case %q has no fixture directory; only contract fields are available", c.ID)
		}
		if boundary == BoundaryProviderRequest {
			want = Payload{Body: c.Fixtures.ExpectedProviderRequest}
		} else {
			want = Payload{Body: c.Fixtures.ExpectedClientResponse, Stream: c.Fixtures.ExpectedClientResponseStream}
		}
	}

	result, err := Compare(cmp, want, got)
	if err != nil {
		return Result{}, fmt.Errorf("case %q boundary %s: %w", c.ID, boundary, err)
	}
	result.Case, result.Boundary = c.ID, boundary
	return result, nil
}

// Compare evaluates got against want under cmp.
//
// The returned error reports a malformed comparison or unparsable artifact. A
// failing comparison is not an error: it is a Result with mismatches, so a caller
// can report every difference at once.
func Compare(cmp Comparison, want, got Payload) (Result, error) {
	result := Result{Mode: cmp.Mode}

	switch cmp.Mode {
	case ModeExact:
		result.Mismatches = compareExact(want, got)
	case ModeExactExcept:
		mismatches, err := compareExactExcept(cmp, want, got)
		if err != nil {
			return Result{}, err
		}
		result.Mismatches = mismatches
	case ModeSemantic:
		mismatches, err := compareSemantic(cmp, want, got)
		if err != nil {
			return Result{}, err
		}
		result.Mismatches = mismatches
	case ModeReject:
		mismatches, err := compareReject(cmp, got)
		if err != nil {
			return Result{}, err
		}
		result.Mismatches = mismatches
	default:
		return Result{}, fmt.Errorf("unknown comparison mode %q", cmp.Mode)
	}
	return result, nil
}
