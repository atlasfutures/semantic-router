package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The declared limit, in the router's token estimate. Small enough that a
// few kilobytes of prose cross it while one line does not.
const contextLimitE2ETokens = 2048

// armedARCRouterWithContextLimit builds the real router from a config file
// (NewServer plus Start, as a deployment does), on the fixture artifact and
// an encoder that answers its probe, with max_context_tokens declared on the
// decision, and returns it armed.
func armedARCRouterWithContextLimit(t *testing.T) (*OpenAIRouter, *hangingARCEncoder) {
	t.Helper()
	t.Setenv(fixtureAPIKeyEnv, "arc-fixture-key")
	encoder := newHangingARCEncoder(t)
	provider := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			http.Error(writer, "no upstream call is expected", http.StatusNotImplemented)
		},
	))
	t.Cleanup(provider.Close)
	artifactDir := writeARCArtifact(t, provider.URL)
	configPath := writeARCRouterConfig(t, artifactDir, encoder.URL(), provider.URL,
		encoderBudget{totalTimeoutSeconds: 90, retryInitialSeconds: 1, retryMaxSeconds: 2})
	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	declared := strings.Replace(string(raw), "readiness_wait_seconds: -1",
		"readiness_wait_seconds: -1\n          max_context_tokens: "+fmt.Sprint(contextLimitE2ETokens), 1)
	if declared == string(raw) {
		t.Fatal("the router config template no longer has the line the limit is declared after")
	}
	if err := os.WriteFile(configPath, []byte(declared), 0o600); err != nil {
		t.Fatal(err)
	}

	port := freeTCPPort(t)
	var built atomic.Pointer[Server]
	serveErr := make(chan error, 1)
	go func() {
		server, err := NewServer(configPath, port, false, "", nil)
		if err != nil {
			serveErr <- fmt.Errorf("NewServer: %w", err)
			return
		}
		built.Store(server)
		serveErr <- server.Start()
	}()
	t.Cleanup(func() {
		if server := built.Load(); server != nil {
			server.Stop()
		}
	})
	awaitTCPAccept(t, port, raylineARCPortListenBudget, serveErr)
	encoder.answer()
	awaitARCArmed(t, 30*time.Second)
	return built.Load().GetRouter(), encoder
}

// The limit is a behaviour-visible API contract, so it is asserted through
// the surfaces a caller reaches, on a router built the way a deployment
// builds it: GET /v1/models at the ExtProc request headers publishes the
// declared limit on the alias beside its candidates, and a turn over the
// limit sent through the request headers and body is refused with the
// replayable 503 and the public failure class, before the encoder is asked
// anything. A turn within the limit is not refused on it.
func TestARCContextLimitIsPublishedAndEnforcedThroughTheRouterSurfaces(t *testing.T) {
	router, encoder := armedARCRouterWithContextLimit(t)
	alias := config.DefaultVSRAutoModelName

	// GET /v1/models through the request-headers surface.
	listing, err := router.handleModelsRequestHeaders("GET", "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	immediate := listing.GetImmediateResponse()
	if immediate == nil || int(immediate.GetStatus().GetCode()) != 200 {
		t.Fatalf("GET /v1/models = %+v, want an immediate 200", listing)
	}
	var models struct {
		Data []struct {
			ID      string `json:"id"`
			Routing struct {
				Candidates       []json.RawMessage `json:"candidates"`
				MaxContextTokens *int              `json:"max_context_tokens"`
			} `json:"routing"`
		} `json:"data"`
	}
	if err := json.Unmarshal(immediate.GetBody(), &models); err != nil {
		t.Fatalf("listing is not JSON: %s", immediate.GetBody())
	}
	published := false
	for _, model := range models.Data {
		if model.ID != alias {
			continue
		}
		published = true
		if model.Routing.MaxContextTokens == nil || *model.Routing.MaxContextTokens != contextLimitE2ETokens {
			t.Fatalf("%s routing.max_context_tokens = %v, want %d", alias, model.Routing.MaxContextTokens, contextLimitE2ETokens)
		}
		if len(model.Routing.Candidates) != 2 {
			t.Fatalf("%s candidates = %d, want the two fixture arms beside the limit", alias, len(model.Routing.Candidates))
		}
	}
	if !published {
		t.Fatalf("%s is not listed in %s", alias, immediate.GetBody())
	}

	// A turn over the limit, through the request headers and body.
	probes := encoder.attempts.Load()
	long := strings.Repeat("the quick brown fox jumps over the lazy dog ", 400)
	ctx := newRecipeRequestContext(alias)
	if _, err := router.handleRequestHeaders(recipeRequestHeaders(alias, "context-limit-over"), ctx); err != nil {
		t.Fatal(err)
	}
	response, err := router.handleRequestBody(recipeRequestBody(alias, `{"role":"user","content":"`+long+`"}`), ctx)
	if err != nil {
		t.Fatal(err)
	}
	finalizeSelectionProcessTerminal(ctx)
	refusal := response.GetImmediateResponse()
	if refusal == nil {
		t.Fatalf("a turn over the limit was not refused: %+v", response)
	}
	if got := int(refusal.GetStatus().GetCode()); got != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503, the status the gateway replays", got)
	}
	class, retryAfter := "", ""
	for _, header := range refusal.GetHeaders().GetSetHeaders() {
		switch strings.ToLower(header.GetHeader().GetKey()) {
		case headers.VSRFailureClass:
			class = string(header.GetHeader().GetRawValue())
		case "retry-after":
			retryAfter = string(header.GetHeader().GetRawValue())
		}
	}
	if class != selectionFailureUnavailable {
		t.Fatalf("%s = %q, want %s", headers.VSRFailureClass, class, selectionFailureUnavailable)
	}
	if retryAfter != "" {
		t.Fatalf("retry-after = %q on a refusal waiting cannot help", retryAfter)
	}
	if got := encoder.attempts.Load(); got != probes {
		t.Fatalf("encoder requests = %d after the refusal, want %d: a refused turn must not reach the encoder", got, probes)
	}
	if ctx.VSRContextTokenCount <= contextLimitE2ETokens {
		t.Fatalf("estimate = %d is not over the %d limit; the body is too short for this test", ctx.VSRContextTokenCount, contextLimitE2ETokens)
	}

	// A turn within the limit is not refused on it.
	ctx = newRecipeRequestContext(alias)
	if _, err := router.handleRequestHeaders(recipeRequestHeaders(alias, "context-limit-under"), ctx); err != nil {
		t.Fatal(err)
	}
	response, err = router.handleRequestBody(recipeRequestBody(alias, `{"role":"user","content":"go"}`), ctx)
	if err != nil {
		t.Fatal(err)
	}
	finalizeSelectionProcessTerminal(ctx)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("a turn within the limit was refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
}
