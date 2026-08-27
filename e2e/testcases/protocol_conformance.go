package testcases

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	// protocolConformanceTestCase is the registered name of the full first-slice
	// run. It is what nightly CI runs.
	protocolConformanceTestCase = "protocol-conformance-first-six"
	// protocolConformanceSmokeTestCase is the registered name of the compact
	// all-protocol subset. It is what the PR gate runs.
	protocolConformanceSmokeTestCase = "protocol-conformance-smoke"
)

const (
	// conformanceTreeV1 is the versioned fixture tree, relative to the repository
	// root the runner starts from.
	conformanceTreeV1 = "e2e/testcases/testdata/protocol-conformance/v1"
	// conformanceTranche is the promoted tranche. The deferred import-* tranche is
	// contract-only until its own plan slice lands, so this testcase never runs it.
	conformanceTranche = conformance.PromotedTranche
	// conformanceSmokeSelection labels the smoke_tier subset in the test report.
	conformanceSmokeSelection = "smoke"
)

// The provider fixture the profile deploys beside the router. The case root is the
// fixture tree baked into that image, which is where the fixture resolves a replay
// script's file references.
const (
	conformanceFixtureNamespace = "conformance-fixture-system"
	conformanceFixtureService   = "conformance-fixture"
	conformanceFixturePort      = "8080"
	conformanceFixtureCaseRoot  = "/fixtures/protocol-conformance/v1"
)

// conformanceRequestTimeout bounds one case. It has to outlast a scripted delay
// followed by a provider disconnect, which is the slowest shape in the corpus.
const conformanceRequestTimeout = 120 * time.Second

func init() {
	pkgtestcases.Register(protocolConformanceTestCase, pkgtestcases.TestCase{
		Description: "Run the promoted protocol-conformance corpus against both wire boundaries of the router",
		Tags:        []string{"conformance", "protocol", "functional"},
		Fn:          runConformanceSelection(conformanceTranche, promotedTrancheCases),
	})
	pkgtestcases.Register(protocolConformanceSmokeTestCase, pkgtestcases.TestCase{
		Description: "Run the compact all-protocol protocol-conformance smoke tier against both wire boundaries of the router",
		Tags:        []string{"conformance", "protocol", "functional", "smoke"},
		Fn:          runConformanceSelection(conformanceSmokeSelection, (*conformance.Inventory).Smoke),
	})
}

// promotedTrancheCases selects the whole promoted tranche.
func promotedTrancheCases(inv *conformance.Inventory) []*conformance.Case {
	return inv.Tranche(conformanceTranche)
}

// runConformanceSelection binds one corpus selection to a registered testcase.
// The selection is named so the report says which subset ran, and the two
// registered names are the only tier knob CI needs: the PR gate asks for the
// smoke name, nightly asks for the full one.
func runConformanceSelection(
	selection string,
	pick func(*conformance.Inventory) []*conformance.Case,
) func(context.Context, *kubernetes.Clientset, pkgtestcases.TestCaseOptions) error {
	return func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
		return testProtocolConformance(ctx, client, opts, selection, pick)
	}
}

// testProtocolConformance runs every promoted conformance case through the router.
//
// Per case: the provider fixture is programmed with the case's replay script, the
// case's client request is sent to the router ingress, and the request the provider
// observed plus the response the router returned are compared against what the case
// declares. A case whose payloads are not authored yet is reported as skipped with
// its reason; it is neither failed nor dropped from the report.
func testProtocolConformance(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	selection string,
	pick func(*conformance.Inventory) []*conformance.Case,
) error {
	inventory, err := conformance.Load(conformanceTreeV1)
	if err != nil {
		return err
	}
	cases := pick(inventory)
	if len(cases) == 0 {
		return fmt.Errorf("conformance selection %q declares no cases", selection)
	}

	routerSession, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open the router ingress: %w", err)
	}
	defer routerSession.Close()

	fixtureSession, err := fixtures.OpenServiceSession(ctx, client, conformanceFixtureOptions(opts))
	if err != nil {
		return fmt.Errorf("open the provider fixture control endpoint: %w", err)
	}
	defer fixtureSession.Close()

	httpClient := &http.Client{Timeout: conformanceRequestTimeout}
	outcomes := runConformanceTranche(
		ctx,
		cases,
		newHTTPConformanceIngress(routerSession.BaseURL(), httpClient),
		newRemoteConformanceProvider(fixtureSession.BaseURL(), conformanceFixtureCaseRoot, httpClient),
	)

	reportConformanceOutcomes(outcomes, opts, selection)
	return conformanceVerdict(outcomes)
}

// conformanceFixtureOptions points a service session at the provider fixture rather
// than at the profile's ingress service.
func conformanceFixtureOptions(opts pkgtestcases.TestCaseOptions) pkgtestcases.TestCaseOptions {
	fixtureOpts := opts
	fixtureOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Name:        conformanceFixtureService,
		Namespace:   conformanceFixtureNamespace,
		ServicePort: conformanceFixturePort,
	}
	return fixtureOpts
}

// reportConformanceOutcomes publishes the per-case picture. Skips carry their reason
// and failures carry every mismatch, so a report reader never has to guess whether a
// case passed, was not authored, or genuinely diverged.
func reportConformanceOutcomes(outcomes []caseOutcome, opts pkgtestcases.TestCaseOptions, selection string) {
	skipped := map[string]string{}
	failures := map[string][]string{}
	passed, ledgers := 0, 0

	for _, outcome := range outcomes {
		switch {
		case outcome.Skipped:
			skipped[outcome.ID] = outcome.Reason
		case outcome.Passed():
			passed++
		default:
			failures[outcome.ID] = outcome.Failures
		}
		if outcome.FidelityChecked {
			ledgers++
		}
	}

	if opts.Verbose {
		printConformanceOutcomes(outcomes)
	}
	if opts.SetDetails == nil {
		return
	}
	opts.SetDetails(map[string]interface{}{
		"selection":                selection,
		"cases_total":              len(outcomes),
		"cases_passed":             passed,
		"cases_failed":             len(failures),
		"cases_skipped":            len(skipped),
		"skipped":                  skipped,
		"failures":                 failures,
		"fidelity_ledgers_checked": ledgers,
	})
}

func printConformanceOutcomes(outcomes []caseOutcome) {
	for _, outcome := range outcomes {
		switch {
		case outcome.Skipped:
			fmt.Printf("[Conformance] ⏭️  %s skipped: %s\n", outcome.ID, outcome.Reason)
		case outcome.Passed():
			fmt.Printf("[Conformance] ✅ %s\n", outcome.ID)
		default:
			fmt.Printf("[Conformance] ❌ %s\n", outcome.ID)
			for _, failure := range outcome.Failures {
				fmt.Printf("[Conformance]      %s\n", failure)
			}
		}
	}
}

// conformanceVerdict fails the testcase when any case that ran diverged. Skipped
// cases never fail the run individually — an unauthored case has nothing to
// assert — but a run where every case skipped asserted nothing and must fail
// rather than report green.
func conformanceVerdict(outcomes []caseOutcome) error {
	var failed []string
	lines := []string{}
	ran := 0

	for _, outcome := range outcomes {
		if outcome.Skipped {
			continue
		}
		ran++
		if outcome.Passed() {
			continue
		}
		failed = append(failed, outcome.ID)
		lines = append(lines, fmt.Sprintf("case %s:", outcome.ID))
		for _, failure := range outcome.Failures {
			lines = append(lines, "  "+failure)
		}
	}
	if ran == 0 && len(outcomes) > 0 {
		return fmt.Errorf("all %d conformance case(s) skipped; the run asserted nothing", len(outcomes))
	}
	if len(failed) == 0 {
		return nil
	}

	sort.Strings(failed)
	return fmt.Errorf("%d conformance case(s) failed (%s):\n%s",
		len(failed), strings.Join(failed, ", "), strings.Join(lines, "\n"))
}
