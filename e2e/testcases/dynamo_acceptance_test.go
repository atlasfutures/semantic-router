package testcases

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// newRefusingServer answers every request with HTTP 500.
func newRefusingServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "refused", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	return server
}

// refusingCluster is a real client-go clientset pointed at an API server that
// refuses. Nothing is stubbed; every call a case makes through it fails.
func refusingCluster(t *testing.T) *kubernetes.Clientset {
	t.Helper()
	client, err := kubernetes.NewForConfig(&rest.Config{Host: newRefusingServer(t).URL})
	if err != nil {
		t.Fatalf("clientset against a refusing API server: %v", err)
	}
	return client
}

// refusingBackendPort is a refusing backend's local port, the form the dynamo
// cases pass around after a port-forward.
func refusingBackendPort(t *testing.T) string {
	t.Helper()
	url := newRefusingServer(t).URL
	_, port, err := net.SplitHostPort(strings.TrimPrefix(url, "http://"))
	if err != nil {
		t.Fatalf("test server URL %q: %v", url, err)
	}
	return port
}

// TestDynamoGPUUtilizationFailsWhenTheClusterRefusesEveryRequest is the ask.
//
// testDynamoGPUUtilization (dynamo_gpu_utilization.go:23),
// testDynamoDynamicBatching (dynamo_dynamic_batching.go:26) and
// testDynamoPerformanceComparison (dynamo_performance_comparison.go:25) state
// no acceptance condition. Every terminal statement in the three bodies is
// return nil: gpu_utilization at :39, :60 and :146, dynamic_batching at :124,
// performance_comparison at :169. What happened is printed and handed to
// SetDetails, so a run in which nothing succeeded is reported green. This
// directory's AGENTS.md asks the opposite: "Each testcase must assert an
// externally visible contract with an explicit pass/fail condition."
//
// Two siblings do assert, so the house style is already here:
// dynamo_optimized_inference.go:107 requires a 50% success rate and
// dynamo_category_classification.go:199 requires 70%.
//
// Only the GPU case runs end to end without a cluster: the other two reach
// their requests through setupServiceConnection (common.go:18), which
// port-forwards into a live one. They are covered below by sendBatchedRequest,
// the call whose outcome dynamic_batching counts.
func TestDynamoGPUUtilizationFailsWhenTheClusterRefusesEveryRequest(t *testing.T) {
	if err := testDynamoGPUUtilization(
		context.Background(),
		refusingCluster(t),
		pkgtestcases.TestCaseOptions{},
	); err == nil {
		t.Error("testDynamoGPUUtilization returned nil after the cluster refused every request")
	}
}

// TestDynamoCasesReturnNilOnTotalFailureToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
//
// Its second half records that the outcome the batching case needs is already
// in its hands: every request against a refusing backend reports Success false,
// and dynamo_dynamic_batching.go never turns successCount into an error.
func TestDynamoCasesReturnNilOnTotalFailureToday(t *testing.T) {
	ctx := context.Background()

	if err := testDynamoGPUUtilization(ctx, refusingCluster(t), pkgtestcases.TestCaseOptions{}); err != nil {
		t.Fatalf("the recorded behaviour has changed: %v", err)
	}

	port := refusingBackendPort(t)
	for requestID := 0; requestID < 5; requestID++ {
		if result := sendBatchedRequest(ctx, port, "What is 2+2?", requestID, false); result.Success {
			t.Errorf("request %d succeeded against a refusing backend", requestID)
		}
	}
}
