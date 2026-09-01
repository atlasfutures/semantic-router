package testcases

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// setupServiceConnection sets up port forwarding to the configured service
// and returns the local port to use for HTTP requests and a cleanup function
func setupServiceConnection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, func(), error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return "", nil, err
	}
	return session.LocalPort(), session.Close, nil
}

// setupRouterAPIConnection sets up port forwarding to the semantic-router API service
// This is used for accessing /api/v1/feedback and /api/v1/ratings endpoints
// which are not exposed through the Envoy Gateway
func setupRouterAPIConnection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, func(), error) {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return "", nil, err
	}
	return session.LocalPort(), session.Close, nil
}

// minUncalibratedRoutingAccuracy is the provisional floor for routing and
// classification testcases that no acceptance contract covers.
//
// It matches the most lenient calibrated floor in
// pkg/testcases/acceptance_contracts.go (domain classification and plugin chain
// both sit at 60%). It is set by analogy rather than from a measured pass rate,
// because these testcases previously recorded none: they accepted a single
// correct case out of N. Raise each call site to its own calibrated constant as
// real pass rates get recorded.
const minUncalibratedRoutingAccuracy = 60.0

// minMCPFallbackAccuracy is deliberately below minUncalibratedRoutingAccuracy.
// The fallback testcase exercises graceful degradation rather than perfect
// classification, which its original comment stated but never encoded as a
// number. This keeps that tolerance while removing the "one success passes" bar.
const minMCPFallbackAccuracy = 40.0

// requireAccuracyFloor turns a counted accuracy into a testcase verdict.
//
// It replaces the "0% only" bars these testcases used to carry, where a single
// correct case out of N satisfied the whole suite. The floors passed in are the
// calibrated values from pkg/testcases/acceptance_contracts.go; applying them
// here as well means they hold on every profile, not only on the one profile
// that declares an acceptance contract.
//
// A run that executed no cases fails: an emptied or unreadable case list must
// not read as a pass.
func requireAccuracyFloor(label string, correct, total int, minimum float64) error {
	if total == 0 {
		return fmt.Errorf("%s: no cases were executed; nothing was asserted", label)
	}

	actual := float64(correct) / float64(total) * 100
	if actual < minimum {
		return fmt.Errorf("%s: %d/%d correct (%.2f%%) below minimum %.2f%%",
			label, correct, total, actual, minimum)
	}

	return nil
}

// Random content generation for stress tests

var (
	// Question templates for variety
	questionTemplates = []string{
		"What is %d + %d?",
		"Calculate %d * %d",
		"What is the result of %d - %d?",
		"Solve: %d divided by %d",
		"What is %d%% of %d?",
		"If I have %d apples and buy %d more, how many do I have?",
		"What is the square root of %d?",
		"What is %d to the power of %d?",
		"How many days are in %d weeks?",
		"What is the average of %d and %d?",
	}

	topics = []string{
		"Explain the concept of machine learning",
		"What is the capital of France?",
		"How does photosynthesis work?",
		"What are the benefits of exercise?",
		"Describe the water cycle",
		"What is quantum computing?",
		"How do vaccines work?",
		"What causes climate change?",
		"Explain blockchain technology",
		"What is artificial intelligence?",
		"How does the internet work?",
		"What is the theory of relativity?",
		"Describe the solar system",
		"What is DNA?",
		"How do airplanes fly?",
	}

	tasks = []string{
		"Write a short poem about nature",
		"Summarize the main points of renewable energy",
		"List 5 programming languages",
		"Describe a typical day in your life",
		"Explain how to make a sandwich",
		"Give me 3 tips for better sleep",
		"What are the colors of the rainbow?",
		"Name 5 countries in Europe",
		"Describe your favorite hobby",
		"What are the main food groups?",
	}
)

// generateRandomContent generates random request content for stress testing
func generateRandomContent(requestID int) string {
	// Use requestID as seed for reproducibility within a test run
	r := rand.New(rand.NewSource(time.Now().UnixNano() + int64(requestID)))

	contentType := r.Intn(3)

	switch contentType {
	case 0:
		// Math question
		template := questionTemplates[r.Intn(len(questionTemplates))]
		num1 := r.Intn(100) + 1
		num2 := r.Intn(100) + 1
		return fmt.Sprintf("Request #%d: "+template, requestID, num1, num2)
	case 1:
		// General topic
		topic := topics[r.Intn(len(topics))]
		return fmt.Sprintf("Request #%d: %s", requestID, topic)
	default:
		// Task
		task := tasks[r.Intn(len(tasks))]
		return fmt.Sprintf("Request #%d: %s", requestID, task)
	}
}

// formatResponseHeaders formats HTTP response headers for logging
func formatResponseHeaders(headers map[string][]string) string {
	if len(headers) == 0 {
		return "  (no headers)"
	}

	var sb strings.Builder
	for key, values := range headers {
		for _, value := range values {
			sb.WriteString(fmt.Sprintf("  %s: %s\n", key, value))
		}
	}
	return sb.String()
}

// truncateString truncates a string to maxLen characters, adding "..." if truncated
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
