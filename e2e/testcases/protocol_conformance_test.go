package testcases

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// localConformanceTree is the fixture tree as this package's tests see it. The
// testcase itself resolves the same tree from the repository root the runner uses.
const localConformanceTree = "testdata/protocol-conformance/v1"

// TestProtocolConformanceTrancheAgainstTheRealTree runs the promoted tranche through
// the loop with a passthrough router in place of the deployed one. It reports what
// the corpus is: every case either ran or was skipped with a reason, and no case is
// silently dropped.
func TestProtocolConformanceTrancheAgainstTheRealTree(t *testing.T) {
	cases := loadTrancheForTest(t)

	server := startFixtureForTest(t)
	router := startRouterForTest(t, fakeRouter{providerURL: server.URL()})
	provider := &inProcessConformanceProvider{server: server}

	outcomes := runConformanceTranche(context.Background(), cases, newHTTPConformanceIngress(router, http.DefaultClient), provider)
	if len(outcomes) != len(cases) {
		t.Fatalf("got %d outcomes for %d cases; every case must be reported", len(outcomes), len(cases))
	}

	for _, outcome := range outcomes {
		switch {
		case outcome.Skipped:
			if outcome.Reason == "" {
				t.Errorf("case %s is skipped without a reason", outcome.ID)
			}
			t.Logf("skipped %s: %s", outcome.ID, outcome.Reason)
		case outcome.Passed():
			t.Logf("passed %s", outcome.ID)
		default:
			t.Logf("failed %s: %s", outcome.ID, strings.Join(outcome.Failures, "; "))
		}
	}
}

// TestPassthroughRouterSatisfiesASameProtocolCase pins the whole loop against a
// real case: program the fixture, send the client request, read what the provider
// observed, and compare both boundaries. seed-01 is a same-protocol capture whose
// only allowed patches are the model and provider credentials, so a router that
// forwards the request and canonicalizes nothing else satisfies it.
//
// seed-01 rather than seed-02 because seed-01 is the plain identity path; seed-02
// asserts the Anthropic-native capture, which the next test pairs with a marker.
func TestPassthroughRouterSatisfiesASameProtocolCase(t *testing.T) {
	c := findCaseForTest(t, "seed-01-chat-identity-model-patch")
	if !c.Loaded() {
		t.Fatal("seed-01 payloads must be authored; they are part of the committed seed tranche")
	}

	server := startFixtureForTest(t)
	router := startRouterForTest(t, fakeRouter{providerURL: server.URL(), mutate: canonicalizeChatMaxTokens})

	outcome := runConformanceCase(context.Background(), c, newHTTPConformanceIngress(router, http.DefaultClient), &inProcessConformanceProvider{server: server})
	if outcome.Skipped {
		t.Fatalf("seed-01 was skipped: %s", outcome.Reason)
	}
	if !outcome.Passed() {
		t.Fatalf("a passthrough router must satisfy a same-protocol capture, got %v", outcome.Failures)
	}
}

// TestPreStreamErrorRelaysStatusAndHeader covers seed-05. The comparators compare
// bodies only, so the status line and the Retry-After header the case declares
// preserved are asserted by the runner against what the provider committed.
func TestPreStreamErrorRelaysStatusAndHeader(t *testing.T) {
	c := findCaseForTest(t, "seed-05-chat-prestream-rate-limit")
	if !c.Loaded() {
		t.Fatal("seed-05 payloads must be authored; they are part of the committed seed tranche")
	}

	server := startFixtureForTest(t)
	router := startRouterForTest(t, fakeRouter{providerURL: server.URL(), mutate: canonicalizeChatMaxTokens})

	outcome := runConformanceCase(context.Background(), c, newHTTPConformanceIngress(router, http.DefaultClient), &inProcessConformanceProvider{server: server})
	if outcome.Skipped {
		t.Fatalf("seed-05 was skipped: %s", outcome.Reason)
	}

	// The fixture relays the committed 429 and Retry-After, so neither relay check
	// may fire. seed-05 declares no allowed patches, so the model a real router
	// selects is the one remaining difference a passthrough cannot produce.
	if containsSubstring(outcome.Failures, "client status") {
		t.Fatalf("the relayed status must satisfy the case, got %v", outcome.Failures)
	}
	if containsSubstring(outcome.Failures, "retry-after") {
		t.Fatalf("the relayed Retry-After must satisfy the case, got %v", outcome.Failures)
	}
	if len(outcome.Failures) != 1 || !containsSubstring(outcome.Failures, "/model") {
		t.Fatalf("the unpatched model must be the only difference, got %v", outcome.Failures)
	}
}

// TestRelayFailuresNameTheStatusAndHeaderThatWereNotRelayed is the negative half:
// a router that swallows the provider's 429 and its Retry-After must fail.
func TestRelayFailuresNameTheStatusAndHeaderThatWereNotRelayed(t *testing.T) {
	c := findCaseForTest(t, "seed-05-chat-prestream-rate-limit")
	if !c.Loaded() {
		t.Fatal("seed-05 payloads must be authored; they are part of the committed seed tranche")
	}

	failures := relayFailures(c, conformanceResponse{Status: http.StatusOK, Headers: http.Header{}})
	if len(failures) != 2 {
		t.Fatalf("both the status and the header must be reported, got %v", failures)
	}
	if !containsSubstring(failures, "429") {
		t.Fatalf("the failure must name the committed status, got %v", failures)
	}
	if !containsSubstring(failures, "retry-after") {
		t.Fatalf("the failure must name the committed header, got %v", failures)
	}
}

// TestMidStreamTruncationDispatchesExactlyOnce covers seed-06. The provider commits
// a stream, writes an unterminated prefix at an unaligned chunk size, then drops the
// connection. The client must still see the bytes that arrived, and the router must
// not dispatch a second time.
func TestMidStreamTruncationDispatchesExactlyOnce(t *testing.T) {
	c := findCaseForTest(t, "seed-06-anthropic-openrouter-midstream-truncation")
	if !c.Loaded() {
		t.Fatal("seed-06 payloads must be authored; they are part of the committed seed tranche")
	}

	server := startFixtureForTest(t)
	// seed-06 is cross-protocol: an Anthropic client over a Chat provider, so the
	// fake router has to rewrite the path the way a translating router would.
	router := startRouterForTest(t, fakeRouter{providerURL: server.URL(), providerPath: "/v1/chat/completions"})

	outcome := runConformanceCase(context.Background(), c, newHTTPConformanceIngress(router, http.DefaultClient), &inProcessConformanceProvider{server: server})
	if outcome.Skipped {
		t.Fatalf("seed-06 was skipped: %s", outcome.Reason)
	}

	// A transport fault must not be reported as a failure to reach the router: the
	// truncated bytes are the evidence the case asserts on.
	if containsSubstring(outcome.Failures, "send the client request") {
		t.Fatalf("a provider disconnect must leave the received bytes readable, got %v", outcome.Failures)
	}
	if containsSubstring(outcome.Failures, "dispatch_attempts") {
		t.Fatalf("the truncated stream must not be retried, got %v", outcome.Failures)
	}
	if got := len(server.Observed()); got != 1 {
		t.Fatalf("provider saw %d dispatch(es), want exactly 1", got)
	}
}

// TestConformanceCaseSkipsWhenPayloadsAreAbsent covers the contract-only state every
// case is in before its directory is authored.
func TestConformanceCaseSkipsWhenPayloadsAreAbsent(t *testing.T) {
	c := &conformance.Case{ID: "seed-99-unauthored"}
	outcome := runConformanceCase(context.Background(), c, nil, nil)

	if !outcome.Skipped {
		t.Fatal("a case with no fixtures must be skipped, not run")
	}
	if !strings.Contains(outcome.Reason, "not authored") {
		t.Fatalf("skip reason must name the missing payloads, got %q", outcome.Reason)
	}
	if outcome.Passed() {
		t.Fatal("a skipped case must not be counted as passed")
	}
}

// TestConformanceCaseSkipsWithoutAReplayScript covers a directory the loader accepted
// but that carries no provider program.
func TestConformanceCaseSkipsWithoutAReplayScript(t *testing.T) {
	c := &conformance.Case{ID: "seed-98-no-script", Fixtures: &conformance.Fixtures{Dir: t.TempDir()}}
	outcome := runConformanceCase(context.Background(), c, nil, nil)

	if !outcome.Skipped || !strings.Contains(outcome.Reason, "replay.yaml") {
		t.Fatalf("expected a skip naming replay.yaml, got skipped=%v reason=%q", outcome.Skipped, outcome.Reason)
	}
}

func TestDispatchFailuresReportsAttemptCountAndRefusals(t *testing.T) {
	c := &conformance.Case{ID: "seed-x", Expectation: conformance.Expectation{DispatchAttempts: 1}}

	if failures := dispatchFailures(c, nil); len(failures) != 1 || !strings.Contains(failures[0], "want dispatch_attempts 1") {
		t.Fatalf("a missing dispatch must be reported, got %v", failures)
	}

	refused := []fixture.ObservedRequest{{Mismatch: "path \"/v1/responses\", want \"/v1/messages\""}}
	failures := dispatchFailures(c, refused)
	if len(failures) != 1 || !strings.Contains(failures[0], "597") {
		t.Fatalf("a refused request must be reported as a misroute, got %v", failures)
	}
}

func TestProviderBoundaryRequiresZeroDispatchForARejectCase(t *testing.T) {
	c := &conformance.Case{
		ID:          "seed-y",
		Expectation: conformance.Expectation{ProviderRequest: conformance.ModeReject},
	}

	if failures := providerBoundaryFailures(c, nil); len(failures) != 0 {
		t.Fatalf("a reject case with no dispatch must pass its provider boundary, got %v", failures)
	}
	failures := providerBoundaryFailures(c, []fixture.ObservedRequest{{Path: "/v1/chat/completions"}})
	if len(failures) != 1 || !strings.Contains(failures[0], "rejects before dispatch") {
		t.Fatalf("a reject case that dispatched must fail, got %v", failures)
	}
}

func TestFidelityFailuresComparesAnEmittedLedger(t *testing.T) {
	c := &conformance.Case{
		ID: "seed-z",
		Expectation: conformance.Expectation{
			Fidelity: map[string]conformance.FidelityAction{"/model": conformance.ActionPatched},
		},
	}

	if failures := fidelityFailures(c, conformanceResponse{Headers: http.Header{}}); failures != nil {
		t.Fatalf("an absent ledger is not a failure while the router does not emit one, got %v", failures)
	}

	emitted := http.Header{}
	emitted.Set(fidelityLedgerHeader, `{"/model":"omitted"}`)
	failures := fidelityFailures(c, conformanceResponse{Headers: emitted})
	if len(failures) != 1 || !strings.Contains(failures[0], "tier differs") {
		t.Fatalf("a ledger that drops a patched field must fail on its tier, got %v", failures)
	}
}

func TestConformanceClientHeadersOnlyForwardSameProtocolExpectations(t *testing.T) {
	script := &conformance.ReplayScript{
		Expect: conformance.ExpectSpec{Headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	same := &conformance.Case{
		Client:   conformance.ClientSpec{Protocol: "anthropic-messages", Mode: "buffered"},
		Provider: conformance.RouteSpec{Protocol: "anthropic-messages"},
		Fixtures: &conformance.Fixtures{Replay: script},
	}
	if got := conformanceClientHeaders(same)["anthropic-version"]; got != "2023-06-01" {
		t.Fatalf("a same-protocol case must send the declared header, got %q", got)
	}

	crossed := &conformance.Case{
		Client:   conformance.ClientSpec{Protocol: "openai-responses", Mode: clientModeStreaming},
		Provider: conformance.RouteSpec{Protocol: "anthropic-messages"},
		Fixtures: &conformance.Fixtures{Replay: script},
	}
	headers := conformanceClientHeaders(crossed)
	if _, ok := headers["anthropic-version"]; ok {
		t.Fatal("a cross-protocol case must not pre-send a header the router is meant to synthesize")
	}
	if headers["accept"] != "text/event-stream" {
		t.Fatalf("a streaming client must ask for a stream, got %q", headers["accept"])
	}
}

func TestConformanceVerdictIgnoresSkipsAndNamesFailures(t *testing.T) {
	outcomes := []caseOutcome{
		{ID: "seed-01", Skipped: true, Reason: "not authored"},
		{ID: "seed-02"},
	}
	if err := conformanceVerdict(outcomes); err != nil {
		t.Fatalf("a run of skips and passes must succeed, got %v", err)
	}

	outcomes = append(outcomes, caseOutcome{ID: "seed-03", Failures: []string{"provider-request /model: differs"}})
	err := conformanceVerdict(outcomes)
	if err == nil || !strings.Contains(err.Error(), "seed-03") || !strings.Contains(err.Error(), "/model") {
		t.Fatalf("the verdict must name the failing case and its mismatch, got %v", err)
	}
}
