package testcases

import (
	"context"
	"fmt"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("mcp-http-classification", pkgtestcases.TestCase{
		Description: "Test MCP classification via HTTP transport",
		Tags:        []string{"mcp", "classification", "http"},
		Fn:          testMCPHTTPClassification,
	})
}

func testMCPHTTPClassification(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing MCP HTTP transport classification")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Critical: always clean up port forwarding

	// Load test cases
	testCases, err := loadMCPTestCases("e2e/testcases/testdata/mcp/mcp_http_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Execute tests and collect results
	var results []MCPTestResult
	for _, testCase := range testCases {
		resp, err := executeMCPRequest(ctx, localPort, testCase.Query, opts.Verbose)
		if err != nil {
			results = append(results, MCPTestResult{
				Description:      testCase.Description,
				Query:            testCase.Query,
				ExpectedCategory: testCase.ExpectedCategory,
				Success:          false,
				Error:            err.Error(),
			})
			continue
		}
		defer resp.Body.Close()

		result := validateMCPResponse(resp, testCase, opts.Verbose)
		results = append(results, result)
	}

	// Calculate accuracy
	totalTests := len(results)
	successfulTests, accuracy := calculateAccuracy(results)

	// Report statistics
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":      totalTests,
			"successful_tests": successfulTests,
			"accuracy_rate":    fmt.Sprintf("%.2f%%", accuracy),
			"failed_tests":     totalTests - successfulTests,
		})
	}

	// Print results
	printMCPTestResults("MCP HTTP CLASSIFICATION", results, totalTests, successfulTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] MCP HTTP classification test completed: %d/%d successful (%.2f%% accuracy)\n",
			successfulTests, totalTests, accuracy)
	}

	// No acceptance contract covers this testcase, so the floor lives here.
	// The previous bar passed on a single successful case out of N.
	if err := requireAccuracyFloor("mcp HTTP classification", successfulTests, totalTests,
		minUncalibratedRoutingAccuracy); err != nil {
		return err
	}

	return nil
}
