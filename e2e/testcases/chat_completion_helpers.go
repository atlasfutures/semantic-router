package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

const localChatCompletionsPath = "/v1/chat/completions"

type localChatCompletionResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

// sendLocalChatCompletion sends a chat-completion request to the router. It
// always sets the x-vsr-debug request header: the v0.4 contract demotes the
// intermediate decision/classification and matched-signal response headers off
// the default surface (#2205), and every routing/classification test using this
// helper asserts those demoted headers, so the debug surface is always required.
func sendLocalChatCompletion(
	ctx context.Context,
	localPort string,
	model string,
	prompt string,
	timeout time.Duration,
) (*localChatCompletionResponse, error) {
	requestBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s%s", localPort, localChatCompletionsPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return &localChatCompletionResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       bodyBytes,
	}, nil
}

// assertChatCompletionSucceeded requires that a body served with 200 is an
// actual completion rather than an error envelope.
//
// A 200 is not on its own evidence that the turn succeeded. Under FULL_DUPLEX
// the router can end the downstream response itself, so a refusal is delivered
// as an error body under the status the upstream had already sent. A case that
// asserts the status and never reads the body counts that turn as healthy.
//
// This is the weak form of the check, for cases that assert routing or
// telemetry rather than content and so cannot name the text they expect.
// assertChatCompletionBody is the strong form, for cases that can.
//
// subject names the request in the failure, so a case making several calls
// says which one returned the envelope.
func assertChatCompletionSucceeded(body []byte, subject string) error {
	var envelope struct {
		Error   json.RawMessage   `json:"error"`
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return fmt.Errorf("%s: served 200 but the body is not JSON: %w: %s",
			subject, err, truncateString(string(body), 500))
	}
	if len(envelope.Error) > 0 && string(envelope.Error) != "null" {
		return fmt.Errorf("%s: served 200 but the body is an error envelope: %s",
			subject, truncateString(string(body), 500))
	}
	if len(envelope.Choices) == 0 {
		return fmt.Errorf("%s: served 200 but the body carries no choices, so it is not a completion: %s",
			subject, truncateString(string(body), 500))
	}
	return nil
}

// assertResponseAPISucceeded is the Response API form of the same check. It is
// needed for a reason the chat form is not: a Response API error envelope
// decodes cleanly into the success struct and leaves every field zero, so a
// decode that returns no error proves nothing on its own.
func assertResponseAPISucceeded(response *fixtures.ResponseAPIResponse, rawBody []byte, subject string) error {
	if response == nil || response.Object != "response" {
		return fmt.Errorf("%s: served 200 but the body is not a response object: %s",
			subject, truncateString(string(rawBody), 500))
	}
	if len(response.Output) == 0 {
		return fmt.Errorf("%s: served 200 but the response carries no output: %s",
			subject, truncateString(string(rawBody), 500))
	}
	return nil
}

func formatUnexpectedChatCompletionStatus(response *localChatCompletionResponse) string {
	var errorMsg strings.Builder
	errorMsg.WriteString(fmt.Sprintf("Unexpected status code: %d\n", response.StatusCode))
	errorMsg.WriteString(fmt.Sprintf("Response body: %s\n", string(response.Body)))
	errorMsg.WriteString("Response headers:\n")
	errorMsg.WriteString(formatResponseHeaders(response.Headers))
	return errorMsg.String()
}

func logUnexpectedChatCompletionStatus(
	verbose bool,
	response *localChatCompletionResponse,
	subject string,
	detailLines ...string,
) {
	if !verbose {
		return
	}

	fmt.Printf("[Test] ✗ HTTP %d Error for %s\n", response.StatusCode, subject)
	for _, detail := range detailLines {
		fmt.Printf("  %s\n", detail)
	}
	fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(response.Headers))
	fmt.Printf("  Response Body: %s\n", string(response.Body))
}
