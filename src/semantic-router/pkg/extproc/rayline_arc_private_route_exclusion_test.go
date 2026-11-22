package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A private turn is served on its arm's private route (an Azure ZDR pin, say),
// which is a route of its own. A provider 429 there excludes that route for
// private turns and leaves the arm's standard route offered to standard
// turns; a standard turn's 429 does the converse. Control: with the fallback
// off neither is excluded, which is what let a private arc-sage episode retry
// into the same Azure rate limit (D-sage-fallback).
func TestRaylineARCPrivateRouteRateLimitExcludesOnlyThatRoute(t *testing.T) {
	for _, tc := range []struct {
		enabled bool
		private bool
	}{
		{true, true},
		{true, false},
		{false, true},
	} {
		resetCellExclusions(t)
		fixture, _, _, _ := fallbackFixture(t, tc.enabled)
		scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
		scorer.privateRoutes = make([]string, len(scorer.routes))
		for arm, route := range scorer.routes {
			scorer.privateRoutes[arm] = route + "\x00azure-zdr"
		}
		ctx := &RequestContext{
			RequestID: "req-private-429", VSRSelectedDecision: fixture.decision,
			VSRRaylineARC: &selection.RaylineARCTrace{
				SelectedArm: 1, WorkerProviderModel: "off", WorkerRoute: scorer.routeFor(1, tc.private),
			},
		}
		recordTurnFailure(ctx, turnFailureRateLimited, false)

		now := time.Now()
		_, privateOut := scorer.cellExclusions(now, true)[1]
		_, standardOut := scorer.cellExclusions(now, false)[1]
		if privateOut != (tc.enabled && tc.private) || standardOut != (tc.enabled && !tc.private) {
			t.Fatalf("enabled=%v private=%v: private route excluded=%v, standard route excluded=%v",
				tc.enabled, tc.private, privateOut, standardOut)
		}
	}
}
