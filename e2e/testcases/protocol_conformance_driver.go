package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// This file owns the per-case protocol-conformance loop and nothing else. The
// fixture tree, the comparators, and the provider fixture belong to DPC-102 and
// DPC-101; the profile and reporting wiring lives in protocol_conformance.go and
// the transports live in protocol_conformance_transport.go.

// clientModeStreaming is the cases.yaml spelling for a client that expects SSE.
const clientModeStreaming = "streaming"

// statusPointer is the ledger key that names the response status line. The ledger
// addresses the whole hop, so not every key is a JSON pointer into a body.
const statusPointer = "/status"

// fidelityLedgerHeader carries the router's observed fidelity ledger as a JSON
// object of pointer to fidelity action.
//
// The router does not emit it yet. DPC-303 introduces the ledger, and this is the
// header the harness reads it from once it does. Until then a case's declared
// ledger is reported as unverified rather than silently treated as satisfied.
const fidelityLedgerHeader = "x-vsr-fidelity-ledger"

// conformanceIngress sends one client request to the router under test.
type conformanceIngress interface {
	Send(ctx context.Context, path string, headers map[string]string, body []byte) (conformanceResponse, error)
}

// conformanceResponse is the router's client-facing answer, captured whole.
type conformanceResponse struct {
	Status  int
	Headers http.Header
	Body    []byte
}

// conformanceProvider is the control surface of the DPC-101 provider fixture. The
// in-process and deployed forms differ only in where the case directory lives, so
// each implementation resolves the directory itself from the case ID.
type conformanceProvider interface {
	// Reset installs the case's replay script and clears the recorded requests.
	Reset(ctx context.Context, c *conformance.Case) error
	// Observed returns the provider requests recorded since the last Reset.
	Observed(ctx context.Context) ([]fixture.ObservedRequest, error)
}

// caseOutcome is one case's verdict. A case is skipped, passed, expected-failed,
// or failed; a skipped case always carries the reason it could not run.
type caseOutcome struct {
	ID       string
	Skipped  bool
	Reason   string
	Failures []string
	// FidelityChecked is true when the router emitted a ledger the case compared
	// against. A case with a declared ledger and no emitted one is not a failure
	// yet, so the count is reported instead of asserted.
	FidelityChecked bool
	// ExpectedFailure is true when the case carries an expected_outcome marker and
	// failed with every signature substring the marker names. Failures still holds
	// what diverged, so the report shows the known gap rather than hiding it.
	ExpectedFailure bool
	// Reference is the marker's reference, carried so the report names the gap.
	Reference string
}

// Passed reports whether the case ran and matched every expectation. An expected
// failure is not a pass: it proved a known gap is still there.
func (o caseOutcome) Passed() bool {
	return !o.Skipped && !o.ExpectedFailure && len(o.Failures) == 0
}

// runConformanceTranche runs every case in order and returns one outcome per case.
// A failing case never stops the run: the point of the corpus is the whole picture.
func runConformanceTranche(
	ctx context.Context,
	cases []*conformance.Case,
	ingress conformanceIngress,
	provider conformanceProvider,
) []caseOutcome {
	outcomes := make([]caseOutcome, 0, len(cases))
	for _, c := range cases {
		outcomes = append(outcomes, runConformanceCase(ctx, c, ingress, provider))
	}
	return outcomes
}

// runConformanceCase drives one case end to end: program the provider fixture, send
// the case's client request to the router, then compare both wire boundaries and the
// fidelity ledger against what the case declares.
func runConformanceCase(
	ctx context.Context,
	c *conformance.Case,
	ingress conformanceIngress,
	provider conformanceProvider,
) caseOutcome {
	if reason := conformanceSkipReason(c); reason != "" {
		return caseOutcome{ID: c.ID, Skipped: true, Reason: reason}
	}

	if err := provider.Reset(ctx, c); err != nil {
		return failedCase(c.ID, "program the provider fixture: %v", err)
	}

	resp, err := ingress.Send(ctx, c.Client.Path, conformanceClientHeaders(c), c.Fixtures.ClientRequest)
	if err != nil {
		return failedCase(c.ID, "send the client request: %v", err)
	}

	observed, err := provider.Observed(ctx)
	if err != nil {
		return failedCase(c.ID, "read the observed provider requests: %v", err)
	}

	outcome := caseOutcome{ID: c.ID, FidelityChecked: resp.Headers.Get(fidelityLedgerHeader) != ""}
	outcome.Failures = append(outcome.Failures, dispatchFailures(c, observed)...)
	outcome.Failures = append(outcome.Failures, providerBoundaryFailures(c, observed)...)
	outcome.Failures = append(outcome.Failures, clientBoundaryFailures(c, resp)...)
	outcome.Failures = append(outcome.Failures, relayFailures(c, resp)...)
	outcome.Failures = append(outcome.Failures, fidelityFailures(c, resp)...)
	return applyExpectedOutcome(c, outcome)
}

// applyExpectedOutcome reads the case's expected_outcome marker over the failures
// the comparators produced.
//
// Three answers, and only one of them is quiet. A marked case that failed with
// every signature substring present is the known gap, reported as an expected
// failure. A marked case that passed is news: the gap may be fixed, so the run
// fails until someone looks. A marked case that failed some other way is also
// news, and the run fails with both the marker and the actual failures, because
// the marker is what a reader would otherwise trust.
func applyExpectedOutcome(c *conformance.Case, outcome caseOutcome) caseOutcome {
	if !c.ExpectsFailure() || outcome.Skipped {
		return outcome
	}
	marker := c.ExpectedOutcome
	outcome.Reason, outcome.Reference = marker.Reason, marker.Reference

	if len(outcome.Failures) == 0 {
		outcome.Failures = []string{fmt.Sprintf(
			"unexpectedly passed: the case is marked an expected failure against %s (%s); the gap may be fixed, so update or remove the marker",
			marker.Reference, marker.Reason)}
		return outcome
	}

	missing := marker.UnmatchedSignature(outcome.Failures)
	if len(missing) == 0 {
		outcome.ExpectedFailure = true
		return outcome
	}
	outcome.Failures = append([]string{fmt.Sprintf(
		"failed, but not with the known failure against %s (%s): signature %q is absent from every failure below",
		marker.Reference, marker.Reason, strings.Join(missing, " | "))}, outcome.Failures...)
	return outcome
}

// conformanceSkipReason names why a case cannot run, or returns "" when it can.
// A contract-only case is a normal state of the corpus, not an error: cases.yaml
// freezes the inventory ahead of the payloads DPC-104 authors.
func conformanceSkipReason(c *conformance.Case) string {
	if !c.Loaded() {
		return "no fixture payloads on disk; the case directory is not authored yet"
	}
	if c.Fixtures.Replay == nil {
		return "the case directory carries no replay.yaml, so the provider fixture has no program to run"
	}
	return ""
}

// conformanceClientHeaders is what the driver sends alongside the client request.
//
// Only a same-protocol case forwards the replay script's expect headers. There the
// header contract is survival: the provider must observe exactly what the client
// sent, so the client has to send it. A cross-protocol case declares provider
// headers the router synthesizes, and a client that pre-sent them would prove
// nothing about the router.
func conformanceClientHeaders(c *conformance.Case) map[string]string {
	headers := map[string]string{"content-type": "application/json"}
	if c.Client.Protocol == c.Provider.Protocol {
		for name, value := range c.Fixtures.Replay.Expect.Headers {
			headers[name] = value
		}
	}
	if c.Client.Mode == clientModeStreaming {
		headers["accept"] = "text/event-stream"
	}
	return headers
}

// dispatchFailures checks how many requests reached the provider and whether the
// fixture accepted them. A refused request answers with StatusExpectationFailed,
// which a body comparison alone would report as a payload difference rather than as
// the misroute it is.
func dispatchFailures(c *conformance.Case, observed []fixture.ObservedRequest) []string {
	var failures []string
	if len(observed) != c.Expectation.DispatchAttempts {
		failures = append(failures, fmt.Sprintf(
			"provider saw %d request(s), want dispatch_attempts %d", len(observed), c.Expectation.DispatchAttempts))
	}
	for i, req := range observed {
		if req.Mismatch != "" {
			failures = append(failures, fmt.Sprintf(
				"provider request %d refused with %d: %s", i, fixture.StatusExpectationFailed, req.Mismatch))
		}
	}
	return failures
}

// providerBoundaryFailures compares the request the provider actually observed.
// The last observed request is the one compared: a retry case declares its attempt
// count separately, and the final attempt is what the provider answered.
func providerBoundaryFailures(c *conformance.Case, observed []fixture.ObservedRequest) []string {
	// A rejection never leaves the router, so this boundary is the absence of a
	// request rather than a payload to compare.
	if c.Expectation.ProviderRequest == conformance.ModeReject {
		if len(observed) > 0 {
			return []string{fmt.Sprintf("case rejects before dispatch but the provider saw %d request(s)", len(observed))}
		}
		return nil
	}
	if len(observed) == 0 {
		return []string{"provider saw no request, so the provider-request boundary cannot be compared"}
	}

	last := observed[len(observed)-1]
	result, err := c.CompareProviderRequest(conformance.Payload{
		Headers: headerMap(last.Headers),
		Body:    last.Body,
	})
	return comparisonFailures(result, err)
}

// clientBoundaryFailures compares the response the router returned to the client.
// The observed encoding comes from the response itself, so a router that answers
// JSON where the case declares a stream fails on the encoding, not on the payload.
func clientBoundaryFailures(c *conformance.Case, resp conformanceResponse) []string {
	result, err := c.CompareClientResponse(conformance.Payload{
		Status:  resp.Status,
		Headers: headerMap(resp.Headers),
		Body:    resp.Body,
		Stream:  isEventStream(resp.Headers),
	})
	return comparisonFailures(result, err)
}

// relayFailures checks the status line and response headers the case declares
// preserved.
//
// The comparators build the expected client payload out of the body artifact alone,
// so an expected status or header has no artifact to come from. The replay script is
// that source: it is where a case states what the provider committed. A ledger entry
// of "preserved" therefore means the client must see exactly that value. Any other
// action names a transformation whose target the corpus does not spell out, so it is
// left to the body comparison.
func relayFailures(c *conformance.Case, resp conformanceResponse) []string {
	committed, ok := committedStatusStep(c.Fixtures.Replay)
	if !ok {
		return nil
	}

	var failures []string
	for pointer, action := range c.Expectation.Fidelity {
		if action != conformance.ActionPreserved {
			continue
		}
		if pointer == statusPointer {
			if resp.Status != committed.Status {
				failures = append(failures, fmt.Sprintf(
					"client status %d, want the provider's committed %d", resp.Status, committed.Status))
			}
			continue
		}
		failures = append(failures, headerRelayFailures(pointer, committed.Headers, resp.Headers)...)
	}

	// The ledger is a map, so a stable order is what makes a failure list readable
	// and comparable between runs.
	sort.Strings(failures)
	return failures
}

// headerRelayFailures reports a preserved response header the router did not relay.
// A ledger pointer is read as a header only when the provider actually committed one
// by that name; every other pointer addresses a body the comparators already own.
func headerRelayFailures(pointer string, committed map[string]string, got http.Header) []string {
	name := strings.TrimPrefix(pointer, "/")
	for declared, want := range committed {
		if !strings.EqualFold(declared, name) {
			continue
		}
		if actual := got.Get(declared); actual != want {
			return []string{fmt.Sprintf(
				"client header %s = %q, want the provider's committed %q", declared, actual, want)}
		}
	}
	return nil
}

// committedStatusStep returns the replay script's status step, which is where the
// provider commits its status line and response headers. The script parser
// guarantees the step is first when it is present.
func committedStatusStep(script *conformance.ReplayScript) (conformance.ReplayStep, bool) {
	if script == nil || len(script.Steps) == 0 || script.Steps[0].Kind != conformance.StepStatus {
		return conformance.ReplayStep{}, false
	}
	return script.Steps[0], true
}

// fidelityFailures compares an emitted fidelity ledger against the case expectation.
// An absent ledger is not a failure while DPC-303 is outstanding; runConformanceCase
// records whether one arrived so a report can show how many cases were verified.
func fidelityFailures(c *conformance.Case, resp conformanceResponse) []string {
	raw := resp.Headers.Get(fidelityLedgerHeader)
	if raw == "" {
		return nil
	}

	var ledger map[string]conformance.FidelityAction
	if err := json.Unmarshal([]byte(raw), &ledger); err != nil {
		return []string{fmt.Sprintf("decode %s: %v", fidelityLedgerHeader, err)}
	}
	result, err := c.CompareFidelity(ledger)
	return comparisonFailures(result, err)
}

// comparisonFailures flattens one comparator result into reportable lines. A
// comparator error is a broken case or an unparsable artifact, which fails the case
// just as a mismatch does but reads differently in the report.
func comparisonFailures(result conformance.Result, err error) []string {
	if err != nil {
		return []string{err.Error()}
	}
	if result.Pass() {
		return nil
	}

	failures := make([]string, 0, len(result.Mismatches))
	for _, mismatch := range result.Mismatches {
		failures = append(failures, string(result.Boundary)+" "+mismatch.String())
	}
	return failures
}

func failedCase(id, format string, args ...any) caseOutcome {
	return caseOutcome{ID: id, Failures: []string{fmt.Sprintf(format, args...)}}
}

// headerMap flattens parsed headers onto the single-valued shape the comparators
// take. A repeated header keeps its first value, which is what a header expectation
// in the corpus names.
func headerMap(header http.Header) map[string]string {
	if len(header) == 0 {
		return nil
	}
	flat := make(map[string]string, len(header))
	for name := range header {
		flat[name] = header.Get(name)
	}
	return flat
}

func isEventStream(header http.Header) bool {
	return strings.Contains(strings.ToLower(header.Get("content-type")), "text/event-stream")
}
