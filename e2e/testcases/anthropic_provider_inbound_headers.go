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

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-provider-inbound-headers", pkgtestcases.TestCase{
		Description: "Anthropic protocol headers a client sends survive the hop to the provider",
		Tags:        []string{"anthropic", "protocol-codec", "headers"},
		Fn:          testAnthropicProviderInboundHeaders,
	})
}

// testAnthropicProviderInboundHeaders asserts on the request headers a provider
// received, which nothing else does today.
//
// Several tests assert the response headers a client sees, and
// anthropic-chat-cache-control asserts the request body a provider saw. The
// request headers a provider saw are recorded by the shim and never checked, so a
// change to the header pass-through policy could drop anthropic-version or
// anthropic-beta on the way upstream while every existing test still passed.
// Anthropic rejects a request with no version header, so that regression would
// only ever be found against a real provider.
//
// Same-protocol only, deliberately. A Messages client sends these headers and the
// Router forwards them, which is the contract under test. A Chat or Responses
// client has no Anthropic header to forward and the Router synthesizes none, so a
// cross-protocol request reaches the provider with no version header at all. That
// is a real gap, but closing it is a behaviour change rather than a test, and a
// backend can already supply the header through extra_headers on its provider
// profile.
func testAnthropicProviderInboundHeaders(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "anthropic-backend-system",
		Name:        "anthropic-backend-qwen",
		ServicePort: "8080",
	}
	backendSession, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backendSession.Close()

	sessionID := fmt.Sprintf("provider-inbound-headers-%d", time.Now().UnixNano())

	// anthropic-beta rides along so the assertion cannot be satisfied by one
	// well-known header being special-cased somewhere.
	sent := map[string]string{
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "prompt-caching-2024-07-31",
	}
	request := map[string]any{
		"model":      "MoM",
		"max_tokens": 32,
		"messages": []any{
			map[string]any{"role": "user", "content": "Reply with one short word."},
		},
	}
	if err := sendAnthropicMessagesWithHeaders(ctx, session, request, sessionID, sent); err != nil {
		return err
	}

	recorded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return fmt.Errorf("read the provider's recorded request: %w", err)
	}
	observed, err := providerInboundHeaders(recorded)
	if err != nil {
		return err
	}
	for name, want := range sent {
		if got := observed[name]; got != want {
			return fmt.Errorf(
				"provider received %s = %q, want %q; the client sent it and the hop must not drop or rewrite it",
				name, got, want)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"provider_inbound_headers_asserted": len(sent)})
	}
	return nil
}

// providerInboundHeaders pulls the recorded request headers out of the shim's
// debug payload. The shim already stores them beside the body; only a reader was
// missing. Names are folded to lower case because HTTP/2 lower-cases them anyway
// and the recorder preserves whatever case arrived.
func providerInboundHeaders(payload []byte) (map[string]string, error) {
	var recorded struct {
		Headers map[string]string `json:"headers"`
	}
	if err := json.Unmarshal(payload, &recorded); err != nil {
		return nil, fmt.Errorf("decode the provider's recorded request: %w", err)
	}
	if len(recorded.Headers) == 0 {
		return nil, fmt.Errorf(
			"the provider recorded no request headers: %s", truncateString(string(payload), 500))
	}
	lowered := make(map[string]string, len(recorded.Headers))
	for name, value := range recorded.Headers {
		lowered[strings.ToLower(name)] = value
	}
	return lowered, nil
}

func sendAnthropicMessagesWithHeaders(
	ctx context.Context,
	session *fixtures.ServiceSession,
	request map[string]any,
	sessionID string,
	headers map[string]string,
) error {
	encoded, err := json.Marshal(request)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, session.BaseURL()+"/v1/messages", bytes.NewReader(encoded),
	)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-test-session-id", sessionID)
	for name, value := range headers {
		req.Header.Set(name, value)
	}

	resp, err := session.HTTPClient(60 * time.Second).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return readErr
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(
			"Messages request returned HTTP %d: %s", resp.StatusCode, truncateString(string(body), 500))
	}
	return nil
}
