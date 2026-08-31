package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// This is the corpus record mode.
//
// The comparators tell you that a golden diverged; they never tell you what the
// router actually sent. Re-authoring a golden by reading per-pointer mismatch
// fragments is guesswork, so the only honest way to refresh the corpus after a
// router change is to capture both wire boundaries again.
//
// It lives beside the driver on purpose. Recording has to send exactly what the
// driver sends, or the refreshed golden encodes the recorder's behavior instead
// of the router's. Reusing conformanceClientHeaders, httpConformanceIngress and
// remoteConformanceProvider is what makes the two agree by construction.
//
// The recorder is inert unless both URLs are set, so an ordinary unit run and CI
// skip it.
//
//	CONFORMANCE_RECORD_ROUTER_URL=http://127.0.0.1:18080 \
//	CONFORMANCE_RECORD_FIXTURE_URL=http://127.0.0.1:18199 \
//	go test ./testcases -run TestRecordProtocolConformanceGoldens -v
//
// Optional:
//
//	CONFORMANCE_RECORD_CASES     comma-separated case IDs; default is the promoted tranche
//	CONFORMANCE_RECORD_CASE_ROOT fixture tree path inside the fixture process
//	CONFORMANCE_RECORD_TREE      corpus tree, relative to this package
//	CONFORMANCE_RECORD_OUT       diagnostics directory
//	CONFORMANCE_RECORD_DRY_RUN   capture and report, write nothing
const (
	recordRouterURLEnv = "CONFORMANCE_RECORD_ROUTER_URL"
	recordFixtureURL   = "CONFORMANCE_RECORD_FIXTURE_URL"
	recordCasesEnv     = "CONFORMANCE_RECORD_CASES"
	recordCaseRootEnv  = "CONFORMANCE_RECORD_CASE_ROOT"
	recordTreeEnv      = "CONFORMANCE_RECORD_TREE"
	recordOutEnv       = "CONFORMANCE_RECORD_OUT"
	recordDryRunEnv    = "CONFORMANCE_RECORD_DRY_RUN"

	// recordTreeDefault is the corpus relative to this package, which is the
	// working directory `go test` uses.
	recordTreeDefault = "testdata/protocol-conformance/v1"
	recordOutDefault  = "/tmp/conformance-record"
)

// recordedCase is the per-case diagnostic written next to the refreshed goldens.
// The goldens themselves carry no provenance, so this file is where a reviewer
// looks to see the path the router chose and whether the fixture accepted it.
type recordedCase struct {
	ID             string                    `json:"id"`
	ClientPath     string                    `json:"client_path"`
	ClientHeaders  map[string]string         `json:"client_headers"`
	ResponseStatus int                       `json:"response_status"`
	ResponseHeader http.Header               `json:"response_headers"`
	Observed       []fixture.ObservedRequest `json:"observed"`
	Wrote          []string                  `json:"wrote,omitempty"`
	Note           string                    `json:"note,omitempty"`
}

func TestRecordProtocolConformanceGoldens(t *testing.T) {
	routerURL, fixtureURL := os.Getenv(recordRouterURLEnv), os.Getenv(recordFixtureURL)
	if routerURL == "" || fixtureURL == "" {
		t.Skipf("record mode is off; set %s and %s to capture goldens", recordRouterURLEnv, recordFixtureURL)
	}

	tree := envOr(recordTreeEnv, recordTreeDefault)
	inventory, err := conformance.Load(tree)
	if err != nil {
		t.Fatalf("load the corpus at %s: %v", tree, err)
	}

	cases, err := recordSelection(inventory, os.Getenv(recordCasesEnv))
	if err != nil {
		t.Fatal(err)
	}

	outDir := envOr(recordOutEnv, recordOutDefault)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("create the diagnostics directory %s: %v", outDir, err)
	}

	client := &http.Client{Timeout: conformanceRequestTimeout}
	ingress := newHTTPConformanceIngress(routerURL, client)
	provider := newRemoteConformanceProvider(fixtureURL, envOr(recordCaseRootEnv, conformanceFixtureCaseRoot), client)
	dryRun := os.Getenv(recordDryRunEnv) != ""

	records := make([]recordedCase, 0, len(cases))
	for _, c := range cases {
		c := c
		t.Run(c.ID, func(t *testing.T) {
			record := recordOneCase(t, c, ingress, provider, dryRun)
			records = append(records, record)
		})
	}

	writeJSONFile(t, filepath.Join(outDir, "record.json"), records)
	t.Logf("diagnostics written to %s/record.json", outDir)
}

// recordOneCase drives one case the way the driver does, then promotes what the
// router actually produced into that case's goldens.
//
// A capture is only usable when the fixture observed exactly one request and
// accepted it. Zero requests means the router never reached the provider, and a
// refused request means the fixture answered 597, so the client response is a
// refusal notice rather than a translated provider reply. Writing a golden from
// either would freeze the wrong bytes, so both cases report and write nothing.
func recordOneCase(
	t *testing.T,
	c *conformance.Case,
	ingress conformanceIngress,
	provider conformanceProvider,
	dryRun bool,
) recordedCase {
	t.Helper()

	if reason := conformanceSkipReason(c); reason != "" {
		t.Skipf("cannot record: %s", reason)
	}

	// A marked case's goldens describe the behavior the Router should reach, not
	// the behavior it has. Recording over them turns the assertion into a
	// tautology: the golden would match what the gap produces, the case would
	// pass, and the marker would then fail the run for passing unexpectedly. So
	// capture the evidence and report it, but never promote it. Removing the
	// marker is the deliberate act that makes a case recordable again.
	if c.ExpectsFailure() {
		dryRun = true
	}

	ctx, cancel := context.WithTimeout(context.Background(), conformanceRequestTimeout)
	defer cancel()

	if err := provider.Reset(ctx, c); err != nil {
		t.Fatalf("program the provider fixture: %v", err)
	}

	headers := conformanceClientHeaders(c)
	resp, err := ingress.Send(ctx, c.Client.Path, headers, c.Fixtures.ClientRequest)
	if err != nil {
		t.Fatalf("send the client request: %v", err)
	}

	observed, err := provider.Observed(ctx)
	if err != nil {
		t.Fatalf("read the observed provider requests: %v", err)
	}

	record := recordedCase{
		ID:             c.ID,
		ClientPath:     c.Client.Path,
		ClientHeaders:  headers,
		ResponseStatus: resp.Status,
		ResponseHeader: resp.Headers,
		Observed:       observed,
	}

	t.Logf("client %s -> %d (%d bytes); provider saw %d request(s)",
		c.Client.Path, resp.Status, len(resp.Body), len(observed))

	if note := recordUsable(observed); note != "" {
		record.Note = note
		t.Errorf("capture is not usable, wrote nothing: %s", note)
		return record
	}
	last := observed[len(observed)-1]
	t.Logf("provider request: %s %s (%d bytes)", last.Method, last.Path, len(last.Body))

	if dryRun {
		record.Note = "nothing written"
		if c.ExpectsFailure() {
			record.Note = fmt.Sprintf(
				"nothing written: the case is marked an expected failure against %s, so its goldens are a contract to reach, not a capture to refresh",
				c.ExpectedOutcome.Reference)
		}
		t.Log(record.Note)
		return record
	}

	record.Wrote = append(record.Wrote,
		writeGolden(t, c.Fixtures.Dir, "expected-provider-request", last.Body, false),
		writeGolden(t, c.Fixtures.Dir, "expected-client-response", resp.Body, c.Fixtures.ExpectedClientResponseStream),
	)
	return record
}

// recordUsable names why a capture cannot become a golden, or returns "" when it
// can.
func recordUsable(observed []fixture.ObservedRequest) string {
	if len(observed) == 0 {
		return "the fixture observed no request, so the router never dispatched"
	}
	last := observed[len(observed)-1]
	if last.Mismatch != "" {
		return fmt.Sprintf("the fixture refused the request (%s %s): %s", last.Method, last.Path, last.Mismatch)
	}
	return ""
}

// writeGolden writes one golden. A stream keeps its bytes exactly as they arrived,
// because chunk boundaries and blank-line framing are part of what the SSE
// comparator asserts. A JSON body is re-indented instead: the comparators decode
// before comparing, so formatting carries no contract, and an indented golden is
// the difference between a reviewable diff and one long line.
func writeGolden(t *testing.T, dir, base string, body []byte, stream bool) string {
	t.Helper()

	if stream {
		return writeFile(t, filepath.Join(dir, base+".sse"), body)
	}
	var indented bytes.Buffer
	if err := json.Indent(&indented, body, "", "  "); err != nil {
		t.Fatalf("%s.json is not JSON, so it cannot be written as a JSON golden: %v", base, err)
	}
	return writeFile(t, filepath.Join(dir, base+".json"), append(indented.Bytes(), '\n'))
}

func writeFile(t *testing.T, path string, payload []byte) string {
	t.Helper()
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	t.Logf("wrote %s (%d bytes)", path, len(payload))
	return path
}

// recordSelection resolves the cases to record. An explicit list is checked
// against the whole inventory so a typo fails loudly rather than recording
// nothing.
func recordSelection(inv *conformance.Inventory, ids string) ([]*conformance.Case, error) {
	if strings.TrimSpace(ids) == "" {
		return inv.Tranche(conformanceTranche), nil
	}
	selected := make([]*conformance.Case, 0)
	for _, id := range strings.Split(ids, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		c, ok := inv.Case(id)
		if !ok {
			return nil, fmt.Errorf("%s names %q, which is not in the inventory", recordCasesEnv, id)
		}
		selected = append(selected, c)
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("%s selected no cases", recordCasesEnv)
	}
	return selected, nil
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("encode %s: %v", path, err)
	}
	if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
