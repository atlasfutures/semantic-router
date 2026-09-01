package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// Shared scaffolding for the protocol-conformance loop tests: a real provider
// fixture, a fake router in front of it, and the corpus accessors.

// inProcessConformanceProvider drives a fixture running in this test process, where
// the case directory is the one the loader read.
type inProcessConformanceProvider struct {
	server *fixture.Server
}

func (p *inProcessConformanceProvider) Reset(_ context.Context, c *conformance.Case) error {
	p.server.Reset(c.Fixtures.Replay, c.Fixtures.Dir)
	return nil
}

func (p *inProcessConformanceProvider) Observed(context.Context) ([]fixture.ObservedRequest, error) {
	return p.server.Observed(), nil
}

func loadTrancheForTest(t *testing.T) []*conformance.Case {
	t.Helper()
	inventory, err := conformance.Load(localConformanceTree)
	if err != nil {
		t.Fatalf("load the conformance tree: %v", err)
	}
	cases := inventory.Tranche(conformanceTranche)
	if len(cases) == 0 {
		t.Fatalf("tranche %q declares no cases", conformanceTranche)
	}
	return cases
}

func findCaseForTest(t *testing.T, id string) *conformance.Case {
	t.Helper()
	for _, c := range loadTrancheForTest(t) {
		if c.ID == id {
			return c
		}
	}
	t.Fatalf("case %q is not in tranche %q", id, conformanceTranche)
	return nil
}

func startFixtureForTest(t *testing.T) *fixture.Server {
	t.Helper()
	server, err := fixture.Start("127.0.0.1:0")
	if err != nil {
		t.Fatalf("start the provider fixture: %v", err)
	}
	t.Cleanup(func() {
		http.DefaultClient.CloseIdleConnections()
		_ = server.Shutdown(context.Background())
	})
	return server
}

// fakeRouter is the stand-in for the deployed router. providerPath rewrites the
// outbound path, which is what a cross-protocol case's router does; mutate rewrites
// the JSON body. The zero value is a passthrough, which is the identity a
// same-protocol capture expects.
type fakeRouter struct {
	providerURL  string
	providerPath string
	mutate       func(body map[string]any)
}

// startRouterForTest runs one fakeRouter and returns its base URL.
func startRouterForTest(t *testing.T, router fakeRouter) string {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := forwardedBody(r, router.mutate)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		outbound := router.providerPath
		if outbound == "" {
			outbound = r.URL.Path
		}
		req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, router.providerURL+outbound, bytes.NewReader(body))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		req.Header = r.Header.Clone()

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer func() { _ = resp.Body.Close() }()

		for name, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(name, value)
			}
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}))
	t.Cleanup(server.Close)
	return server.URL
}

// forwardedBody returns the request body the fake router sends on. Passthrough keeps
// the original bytes, so a same-protocol case still proves byte-level survival.
func forwardedBody(r *http.Request, mutate func(body map[string]any)) ([]byte, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if mutate == nil {
		return raw, nil
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, err
	}
	mutate(decoded)
	return json.Marshal(decoded)
}

// canonicalizeChatMaxTokens is the one rewrite a same-protocol OpenAI Chat hop
// still performs. The codec accepts either spelling of the output-token ceiling
// and emits only the canonical one, so a fake router that forwarded max_tokens
// untouched would model byte-level passthrough -- an identity the Chat contract
// stopped having when the codec took ownership of the wire format.
func canonicalizeChatMaxTokens(body map[string]any) {
	value, ok := body["max_tokens"]
	if !ok {
		return
	}
	delete(body, "max_tokens")
	body["max_completion_tokens"] = value
}

func containsSubstring(lines []string, want string) bool {
	for _, line := range lines {
		if strings.Contains(line, want) {
			return true
		}
	}
	return false
}
