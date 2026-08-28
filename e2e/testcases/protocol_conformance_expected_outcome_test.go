package testcases

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Tests for the expected-outcome marker: the three answers it produces, how a
// report counts them, and the all-skip guard it must not weaken.

// markedCaseForTest is a synthetic marked case, so the expected-outcome tests
// exercise the marker over failure strings the test owns rather than the router.
func markedCaseForTest() *conformance.Case {
	return &conformance.Case{
		ID: "seed-x",
		ExpectedOutcome: &conformance.ExpectedOutcome{
			Status:    conformance.OutcomeFail,
			Reference: "upstream#42",
			Reason:    "the router drops the system blocks",
			Signature: []string{"/system", "field missing"},
		},
	}
}

// TestAKnownFailureIsReportedAsExpected is the quiet answer: the case failed with
// every signature substring present, so the gap is exactly the one the marker
// names and the run does not fail.
func TestAKnownFailureIsReportedAsExpected(t *testing.T) {
	outcome := applyExpectedOutcome(markedCaseForTest(), caseOutcome{
		ID:       "seed-x",
		Failures: []string{"provider-request /system: field missing (want [...], got <absent>)"},
	})

	if !outcome.ExpectedFailure {
		t.Fatalf("the known failure must be reported as expected, got %+v", outcome)
	}
	if outcome.Passed() {
		t.Error("an expected failure must not be counted as a pass")
	}
	if len(outcome.Failures) == 0 {
		t.Error("an expected failure must keep what diverged, so a report can show it")
	}
	if outcome.Reference != "upstream#42" || outcome.Reason == "" {
		t.Errorf("the outcome must carry the marker context, got %+v", outcome)
	}
	if err := conformanceVerdict([]caseOutcome{outcome}); err != nil {
		t.Errorf("an expected failure must not fail the run, got %v", err)
	}
}

// TestAMarkedCaseThatPassesFailsTheRun is the second answer, run against the real
// corpus. seed-02 is marked an expected failure because the router has no native
// Anthropic egress. A passthrough router does have one, so the case passes, and a
// passing marked case must fail the run: the marker may be stale.
func TestAMarkedCaseThatPassesFailsTheRun(t *testing.T) {
	c := findCaseForTest(t, "seed-02-anthropic-native-request-capture")
	if !c.ExpectsFailure() {
		t.Fatal("seed-02 must carry an expected-failure marker")
	}

	server := startFixtureForTest(t)
	router := startRouterForTest(t, fakeRouter{providerURL: server.URL()})

	outcome := runConformanceCase(context.Background(), c, newHTTPConformanceIngress(router, http.DefaultClient), &inProcessConformanceProvider{server: server})
	if outcome.Passed() || outcome.ExpectedFailure {
		t.Fatalf("a marked case that met every expectation must fail the run, got %+v", outcome)
	}
	if !containsSubstring(outcome.Failures, "unexpectedly passed") {
		t.Error("the failure must say the case unexpectedly passed")
	}
	if !containsSubstring(outcome.Failures, c.ExpectedOutcome.Reference) {
		t.Error("the failure must name the marker reference")
	}
	if err := conformanceVerdict([]caseOutcome{outcome}); err == nil {
		t.Error("the run verdict must fail on an unexpectedly passing marked case")
	}
}

// TestAFailureOutsideTheSignatureFailsTheRun is the third answer: a different
// regression must not be absorbed by the marker.
func TestAFailureOutsideTheSignatureFailsTheRun(t *testing.T) {
	outcome := applyExpectedOutcome(markedCaseForTest(), caseOutcome{
		ID:       "seed-x",
		Failures: []string{`provider-request /model: value differs (want "a", got "b")`},
	})

	if outcome.ExpectedFailure || outcome.Passed() {
		t.Fatalf("a different failure must not be absorbed by the marker, got %+v", outcome)
	}
	if !containsSubstring(outcome.Failures, "not with the known failure") {
		t.Error("the failure must say the known signature was absent")
	}
	if !containsSubstring(outcome.Failures, "upstream#42") {
		t.Error("the failure must name the marker reference")
	}
	if !containsSubstring(outcome.Failures, "/model") {
		t.Error("the failure must keep the actual divergence")
	}
	if err := conformanceVerdict([]caseOutcome{outcome}); err == nil {
		t.Error("a wrong-signature failure must fail the run")
	}
}

// TestAPartialSignatureMatchIsNotTheKnownFailure pins the "all substrings" rule.
// One substring alone would let a regression that happens to touch /system hide.
func TestAPartialSignatureMatchIsNotTheKnownFailure(t *testing.T) {
	outcome := applyExpectedOutcome(markedCaseForTest(), caseOutcome{
		ID:       "seed-x",
		Failures: []string{"provider-request /system: unexpected field"},
	})

	if outcome.ExpectedFailure {
		t.Fatalf("a partial signature match must not count as the known failure, got %+v", outcome)
	}
}

// TestExpectedOutcomeLeavesUnmarkedAndSkippedCasesAlone covers the two states the
// marker must never touch.
func TestExpectedOutcomeLeavesUnmarkedAndSkippedCasesAlone(t *testing.T) {
	unmarked := applyExpectedOutcome(&conformance.Case{ID: "seed-y"}, caseOutcome{ID: "seed-y"})
	if !unmarked.Passed() || unmarked.ExpectedFailure {
		t.Errorf("a case with no marker must be reported as it was, got %+v", unmarked)
	}

	skipped := applyExpectedOutcome(markedCaseForTest(), caseOutcome{ID: "seed-x", Skipped: true, Reason: "not authored"})
	if !skipped.Skipped || skipped.ExpectedFailure || len(skipped.Failures) > 0 {
		t.Errorf("a marker must not turn a skip into a failure, got %+v", skipped)
	}
}

// TestConformanceReportCountsExpectedFailuresApart keeps a green run honest: the
// details a report reader sees have to say how many cases proved a known gap
// rather than folding them into the pass count.
func TestConformanceReportCountsExpectedFailuresApart(t *testing.T) {
	var details map[string]interface{}
	opts := pkgtestcases.TestCaseOptions{SetDetails: func(d map[string]interface{}) { details = d }}

	reportConformanceOutcomes([]caseOutcome{
		{ID: "seed-01"},
		{ID: "seed-02", ExpectedFailure: true, Reference: "upstream#42", Reason: "no native egress", Failures: []string{"provider-request /system: field missing"}},
		{ID: "seed-03", Failures: []string{"provider-request /model: value differs"}},
		{ID: "seed-04", Skipped: true, Reason: "not authored"},
	}, opts, "unit")

	for field, want := range map[string]int{
		"cases_total": 4, "cases_passed": 1, "cases_failed": 1,
		"cases_skipped": 1, "cases_expected_failures": 1,
	} {
		if got := details[field]; got != want {
			t.Errorf("%s = %v, want %d", field, got, want)
		}
	}
	expected, ok := details["expected_failures"].(map[string]string)
	if !ok || expected["seed-02"] != "upstream#42: no native egress" {
		t.Errorf("expected_failures = %v, want seed-02 to name its reference and reason", details["expected_failures"])
	}
}

// TestConformanceVerdictStillFailsAnAllSkipRun pins the guard the expected-failure
// path must not weaken: a run that asserted nothing is not green.
func TestConformanceVerdictStillFailsAnAllSkipRun(t *testing.T) {
	err := conformanceVerdict([]caseOutcome{
		{ID: "seed-01", Skipped: true, Reason: "not authored"},
		{ID: "seed-02", Skipped: true, Reason: "not authored"},
	})
	if err == nil || !strings.Contains(err.Error(), "asserted nothing") {
		t.Fatalf("an all-skip run must fail, got %v", err)
	}

	// One expected failure is enough to make the run meaningful: it ran, and it
	// proved the gap its marker names is still exactly that gap.
	err = conformanceVerdict([]caseOutcome{
		{ID: "seed-01", Skipped: true, Reason: "not authored"},
		{ID: "seed-02", ExpectedFailure: true, Failures: []string{"provider-request /system: field missing"}},
	})
	if err != nil {
		t.Fatalf("a run whose only ran case was an expected failure must succeed, got %v", err)
	}
}
