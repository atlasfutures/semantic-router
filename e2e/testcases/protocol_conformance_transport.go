package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// This file holds the deployed transports the protocol-conformance driver runs
// against: an HTTP ingress pointed at the router, and a provider fixture reached
// over its control endpoints. Both are thin; the case logic is in the driver.

// httpConformanceIngress posts a case's client request to the router ingress.
type httpConformanceIngress struct {
	baseURL string
	client  *http.Client
}

func newHTTPConformanceIngress(baseURL string, client *http.Client) *httpConformanceIngress {
	return &httpConformanceIngress{baseURL: baseURL, client: client}
}

// Send posts body to the case's client path and reads the whole response. A
// streaming case is read to completion, including a stream the provider truncated,
// because a truncation case asserts on exactly what did arrive.
func (i *httpConformanceIngress) Send(
	ctx context.Context,
	requestPath string,
	headers map[string]string,
	body []byte,
) (conformanceResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, i.baseURL+requestPath, bytes.NewReader(body))
	if err != nil {
		return conformanceResponse{}, fmt.Errorf("build request for %s: %w", requestPath, err)
	}
	for name, value := range headers {
		req.Header.Set(name, value)
	}

	resp, err := i.client.Do(req)
	if err != nil {
		return conformanceResponse{}, fmt.Errorf("post %s: %w", requestPath, err)
	}
	defer func() { _ = resp.Body.Close() }()

	// A provider disconnect surfaces here as a read error on a response the router
	// already committed. The bytes that did arrive are the evidence, so they are
	// returned rather than discarded with the error.
	received, readErr := io.ReadAll(resp.Body)
	if readErr != nil && len(received) == 0 {
		return conformanceResponse{}, fmt.Errorf("read response from %s: %w", requestPath, readErr)
	}
	return conformanceResponse{Status: resp.StatusCode, Headers: resp.Header, Body: received}, nil
}

// remoteConformanceProvider drives a provider fixture that runs beside the router
// rather than in this process. caseRoot is the fixture tree path inside that
// container, which is where the fixture resolves a script's file references.
type remoteConformanceProvider struct {
	baseURL  string
	caseRoot string
	client   *http.Client
}

func newRemoteConformanceProvider(baseURL, caseRoot string, client *http.Client) *remoteConformanceProvider {
	return &remoteConformanceProvider{baseURL: baseURL, caseRoot: caseRoot, client: client}
}

// Reset posts the case's raw replay.yaml to the fixture control endpoint. The bytes
// are sent verbatim so the fixture applies the same validation the loader did.
func (p *remoteConformanceProvider) Reset(ctx context.Context, c *conformance.Case) error {
	raw, err := os.ReadFile(filepath.Join(c.Fixtures.Dir, conformance.ReplayFile))
	if err != nil {
		return fmt.Errorf("read %s for case %q: %w", conformance.ReplayFile, c.ID, err)
	}

	target := fmt.Sprintf("%s%s?dir=%s", p.baseURL, fixture.PathReset, url.QueryEscape(path.Join(p.caseRoot, c.ID)))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("build reset request for case %q: %w", c.ID, err)
	}
	req.Header.Set("content-type", "application/yaml")

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("reset the provider fixture for case %q: %w", c.ID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNoContent {
		detail, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("reset the provider fixture for case %q: status %d: %s", c.ID, resp.StatusCode, bytes.TrimSpace(detail))
	}
	return nil
}

// Observed reads the requests the fixture recorded since the last Reset.
func (p *remoteConformanceProvider) Observed(ctx context.Context) ([]fixture.ObservedRequest, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+fixture.PathObserved, nil)
	if err != nil {
		return nil, fmt.Errorf("build observed request: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("read the fixture observed endpoint: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("read the fixture observed endpoint: status %d", resp.StatusCode)
	}

	var observed []fixture.ObservedRequest
	if err := json.NewDecoder(resp.Body).Decode(&observed); err != nil {
		return nil, fmt.Errorf("decode the fixture observed endpoint: %w", err)
	}
	return observed, nil
}
