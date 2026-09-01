//go:build vsr_next_bucket_b

// Parked until Bucket B re-seats the ARC dispatch hooks on upstream's
// prepareProviderDispatch / applyDispatchDecision seam. Build with
// -tags vsr_next_bucket_b once those symbols exist again.

package extproc

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// dispatchFailureRouterWithUnresolvableBackend builds the smallest router whose
// backend resolution fails: the model prefers an endpoint that names a provider
// profile which does not exist, so ResolveAddress returns an error rather than
// simply reporting "not found".
func dispatchFailureRouterWithUnresolvableBackend() *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {PreferredEndpoints: []string{"endpoint-a"}},
				},
				VLLMEndpoints: []config.VLLMEndpoint{
					{Name: "endpoint-a", ProviderProfileName: "missing-profile"},
				},
			},
		},
	}
}

// TestSelectionDispatchFailureClassesAreNotDoublePrefixed pins the failure
// class each caller hands to selectionDispatchFailureResponseFor. That helper
// owns the "dispatch_" prefix for the ARC metric, so a caller passing an
// already-prefixed class silently produces a label such as
// "dispatch_dispatch_mapping" — a second time series that no dashboard or
// alert queries, making the failure invisible rather than merely misnamed.
func TestSelectionDispatchFailureClassesAreNotDoublePrefixed(t *testing.T) {
	router := dispatchFailureRouterWithUnresolvableBackend()
	ctx := &RequestContext{
		RequestID: "req-dispatch-mapping",
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			&recordingSelectionTransaction{},
		),
	}

	correctBefore := testutil.ToFloat64(
		metrics.RaylineARCSelectionFailures.WithLabelValues("dispatch_mapping"),
	)
	doubledBefore := testutil.ToFloat64(
		metrics.RaylineARCSelectionFailures.WithLabelValues("dispatch_dispatch_mapping"),
	)

	_, _, _, response, err := router.resolveAutoRoutingTarget(ctx, "model-a")
	if err != nil {
		t.Fatalf("an armed selection must fail closed with a response, not an error: %v", err)
	}
	if response == nil {
		t.Fatal("expected an ARC dispatch failure response")
	}

	if got := testutil.ToFloat64(
		metrics.RaylineARCSelectionFailures.WithLabelValues("dispatch_mapping"),
	) - correctBefore; got != 1 {
		t.Fatalf("dispatch_mapping delta = %v, want 1", got)
	}
	if got := testutil.ToFloat64(
		metrics.RaylineARCSelectionFailures.WithLabelValues("dispatch_dispatch_mapping"),
	) - doubledBefore; got != 0 {
		t.Fatalf("dispatch_dispatch_mapping delta = %v, want 0 (class was double-prefixed)", got)
	}
}
