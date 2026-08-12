package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylineremote"
)

type remoteProtocolFixture struct {
	mu       sync.Mutex
	calls    map[string]int
	prepare  map[string]any
	settle   map[string]any
	auth     []string
	decision string
}

func (fixture *remoteProtocolFixture) handler(
	t *testing.T,
) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		fixture.mu.Lock()
		fixture.calls[request.URL.Path]++
		fixture.auth = append(
			fixture.auth,
			request.Header.Get("authorization"),
		)
		fixture.mu.Unlock()
		if !fixture.serveCatalog(t, writer, request) &&
			!fixture.serveTransaction(t, writer, request) {
			writer.WriteHeader(http.StatusNotFound)
		}
	})
}

func (fixture *remoteProtocolFixture) serveCatalog(
	t *testing.T,
	writer http.ResponseWriter,
	request *http.Request,
) bool {
	switch request.URL.Path {
	case "/v1/route/capabilities":
		writeRemoteFixtureJSON(t, writer, map[string]any{
			"schema_version": "rayline-router.selection-capabilities.v1",
			"transaction_schema_version": raylineremote.
				TransactionSchemaVersion,
			"bundle_version": "bundle-immutable-v1",
			"protocols": []string{
				raylineremote.OpenAIChatProtocol,
			},
			"operations": []string{
				"prepare",
				"renew",
				"commit",
				"abort",
				"settle",
			},
			"workers": []string{
				"outside-worker",
				"worker-b",
				"worker-a",
			},
			"lease_seconds":   30,
			"pending_journal": "bounded_in_process_single_replica",
		})
	case "/v1/workers":
		writeRemoteFixtureJSON(t, writer, []map[string]any{
			remoteFixtureWorker(
				"outside-worker",
				"provider/outside",
				"disabled",
			),
			remoteFixtureWorker("worker-b", "provider/b", "enabled"),
			remoteFixtureWorker("worker-a", "provider/a", "disabled"),
		})
	default:
		return false
	}
	return true
}

func (fixture *remoteProtocolFixture) serveTransaction(
	t *testing.T,
	writer http.ResponseWriter,
	request *http.Request,
) bool {
	switch request.URL.Path {
	case "/v1/route/prepare":
		payload := decodeRemoteFixtureRequest(t, request)
		decisionID, _ := payload["decision_id"].(string)
		fixture.mu.Lock()
		fixture.prepare = payload
		fixture.decision = decisionID
		fixture.mu.Unlock()
		writeRemoteFixtureJSON(t, writer, map[string]any{
			"schema_version":   raylineremote.TransactionSchemaVersion,
			"decision_id":      decisionID,
			"receipt":          "receipt_fixture_opaque",
			"state":            "prepared",
			"selected_worker":  "worker-b",
			"policy":           "fixture-policy",
			"route_call_index": 4,
			"bundle_version":   "bundle-immutable-v1",
			"lease_expires_at": time.Now().
				Add(30 * time.Second).
				Format(time.RFC3339Nano),
		})
	case "/v1/route/renew":
		payload := decodeRemoteFixtureRequest(t, request)
		writeRemoteFixtureJSON(t, writer, remoteFixtureState(
			payload,
			"prepared",
			time.Now().Add(30*time.Second).Format(time.RFC3339Nano),
		))
	case "/v1/route/commit":
		payload := decodeRemoteFixtureRequest(t, request)
		writeRemoteFixtureJSON(
			t,
			writer,
			remoteFixtureState(payload, "committed", ""),
		)
	case "/v1/route/abort":
		payload := decodeRemoteFixtureRequest(t, request)
		writeRemoteFixtureJSON(
			t,
			writer,
			remoteFixtureState(payload, "aborted", ""),
		)
	case "/v1/route/settle":
		payload := decodeRemoteFixtureRequest(t, request)
		fixture.mu.Lock()
		fixture.settle = payload
		fixture.mu.Unlock()
		writeRemoteFixtureJSON(
			t,
			writer,
			remoteFixtureState(payload, "settled", ""),
		)
	default:
		return false
	}
	return true
}

func TestRaylineRemoteSelectorRunsAuthoritativeTransaction(
	t *testing.T,
) {
	fixture := &remoteProtocolFixture{
		calls: make(map[string]int),
	}
	server := httptest.NewServer(fixture.handler(t))
	defer server.Close()
	t.Setenv("RAYLINE_REMOTE_TEST_API_KEY", "fixture-api-key")
	t.Setenv(
		"RAYLINE_REMOTE_TEST_HMAC_KEY",
		"0123456789abcdef0123456789abcdef",
	)
	cfg, decision := remoteSelectorConfig(server.URL)
	selector, readinessFailure := createRaylineRemoteSelector(cfg)
	if readinessFailure != "" || selector == nil {
		t.Fatalf(
			"selector readiness failure=%q selector=%#v",
			readinessFailure,
			selector,
		)
	}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineRemote, selector)
	router := &OpenAIRouter{
		Config:        cfg,
		ModelSelector: registry,
	}
	requestContext := &RequestContext{RequestID: "request-fixture"}
	selectionContext := &selection.SelectionContext{
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
		RaylineRemote: &selection.RaylineRemoteSelectionContext{
			DecisionID:   "rt_fixture-decision",
			RawEpisodeID: "raw-private-episode",
			RequestBody: []byte(`{
				"model":"auto",
				"messages":[{"role":"user","content":"prompt-canary"}],
				"tools":[{"type":"function","function":{"name":"tool-canary"}}]
			}`),
		},
	}
	selected, method, err := router.selectModelFromCandidates(
		selectionContext,
		decision.Algorithm,
		requestContext,
	)
	if err != nil {
		t.Fatal(err)
	}
	if selected == nil ||
		selected.Model != "logical-b" ||
		method != string(selection.MethodRaylineRemote) {
		t.Fatalf("selection=%#v method=%q", selected, method)
	}
	if requestContext.SelectionTransaction == nil ||
		requestContext.VSRRaylineRemote == nil ||
		requestContext.VSRRaylineRemote.SelectedIndex != 1 {
		t.Fatalf("request binding = %#v", requestContext)
	}
	if err := selectionDispatchAllowed(requestContext); err != nil {
		t.Fatal(err)
	}
	requestContext.UpstreamStatusCode = 200
	if err := finalizeSelectionResponseHeaders(
		requestContext,
		true,
	); err != nil {
		t.Fatal(err)
	}
	inputTokens := 11
	outputTokens := 7
	cost := 0.000025
	finalizeSelectionSettlement(
		requestContext,
		selectionActualOutcome{
			OutcomeClass: "success",
			StatusCode:   200,
			InputTokens:  &inputTokens,
			OutputTokens: &outputTokens,
			CostUSD:      &cost,
		},
	)
	finalizeSelectionProcessTerminal(requestContext)

	assertRemoteProtocolFixture(t, fixture, inputTokens, outputTokens)
}

//nolint:cyclop // This assertion audits the complete privacy and lifecycle contract.
func assertRemoteProtocolFixture(
	t *testing.T,
	fixture *remoteProtocolFixture,
	inputTokens int,
	outputTokens int,
) {
	t.Helper()
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if fixture.calls["/v1/route/prepare"] != 1 ||
		fixture.calls["/v1/route/renew"] != 1 ||
		fixture.calls["/v1/route/commit"] != 1 ||
		fixture.calls["/v1/route/settle"] != 1 ||
		fixture.calls["/v1/route/abort"] != 0 {
		t.Fatalf("protocol calls = %#v", fixture.calls)
	}
	for _, authorization := range fixture.auth {
		if authorization != "Bearer fixture-api-key" {
			t.Fatalf("authorization = %q", authorization)
		}
	}
	if fixture.prepare["episode_key"] == "raw-private-episode" ||
		!strings.HasPrefix(
			fixture.prepare["episode_key"].(string),
			"hmac-sha256:",
		) {
		t.Fatalf(
			"episode identity was not protected: %#v",
			fixture.prepare["episode_key"],
		)
	}
	candidates, _ := fixture.prepare["candidates"].([]any)
	if len(candidates) != 2 ||
		candidates[0] != "worker-a" ||
		candidates[1] != "worker-b" {
		t.Fatalf("candidate mask = %#v", candidates)
	}
	policyRequest, _ := fixture.prepare["request"].(map[string]any)
	if _, leaked := policyRequest["model"]; leaked {
		t.Fatalf("client model leaked into policy request: %#v", policyRequest)
	}
	outcome, _ := fixture.settle["outcome"].(map[string]any)
	if outcome["input_tokens"] != float64(inputTokens) ||
		outcome["output_tokens"] != float64(outputTokens) {
		t.Fatalf("settled outcome = %#v", outcome)
	}
}

func remoteSelectorConfig(
	baseURL string,
) (*config.RouterConfig, *config.Decision) {
	cacheWrite := 1.0
	reasoningOff := false
	reasoningOn := true
	decision := config.Decision{
		Name: "remote-decision",
		ModelRefs: []config.ModelRef{
			{
				Model: "logical-a",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: &reasoningOff,
				},
			},
			{
				Model: "logical-b",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: &reasoningOn,
				},
			},
		},
		Algorithm: &config.AlgorithmConfig{
			Type:    config.RaylineRemoteAlgorithmType,
			OnError: "fail_closed",
			RaylineRemote: &config.RaylineRemoteAlgorithmConfig{
				BaseURL: baseURL,
				// The fixture serves plaintext on loopback.
				AllowInsecureTransport: true,
				BundleVersion:          "bundle-immutable-v1",
				APIKeyEnv:              "RAYLINE_REMOTE_TEST_API_KEY",
				EpisodeIDHeader:        "x-rayline-episode-id",
				EpisodeHMACKeyEnv:      "RAYLINE_REMOTE_TEST_HMAC_KEY",
				DecisionIDHeader:       "x-rayline-route-id",
				ConnectTimeoutMS:       100,
				RequestTimeoutMS:       1_000,
				LeaseTTLSeconds:        30,
				MaxRetries:             0,
				Workers: []config.RaylineRemoteWorkerConfig{
					{ID: "worker-a", Model: "logical-a"},
					{ID: "worker-b", Model: "logical-b"},
				},
			},
		},
		Adaptations: config.DecisionAdaptationsConfig{
			Mode: config.DecisionAdaptationModeBypass,
		},
	}
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{decision},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"logical-a": {
					PreferredEndpoints: []string{"provider-a"},
					APIFormat:          config.APIFormatOpenAI,
					ExternalModelIDs: map[string]string{
						"mock": "provider/a",
					},
					Pricing: config.ModelPricing{
						Currency:         "USD",
						PromptPer1M:      1,
						CachedInputPer1M: 1,
						CacheWritePer1M:  &cacheWrite,
						CompletionPer1M:  2,
					},
				},
				"logical-b": {
					PreferredEndpoints: []string{"provider-b"},
					APIFormat:          config.APIFormatOpenAI,
					ExternalModelIDs: map[string]string{
						"mock": "provider/b",
					},
					Pricing: config.ModelPricing{
						Currency:         "USD",
						PromptPer1M:      1,
						CachedInputPer1M: 1,
						CacheWritePer1M:  &cacheWrite,
						CompletionPer1M:  2,
					},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "provider-a", Type: "mock"},
				{Name: "provider-b", Type: "mock"},
			},
		},
	}
	return cfg, &cfg.Decisions[0]
}

func remoteFixtureWorker(
	id string,
	model string,
	thinkingMode string,
) map[string]any {
	return map[string]any{
		"id":                         id,
		"backend":                    "mock",
		"model":                      model,
		"thinking_mode":              thinkingMode,
		"openrouter_allow_fallbacks": false,
		"capability_tags":            []string{"text"},
		"pricing_snapshot_version":   "fixture-prices-v1",
		"per_token_prices": map[string]any{
			"input":       0.000001,
			"cache_read":  0.000001,
			"cache_write": 0.000001,
			"output":      0.000002,
		},
	}
}

func decodeRemoteFixtureRequest(
	t *testing.T,
	request *http.Request,
) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	return payload
}

func remoteFixtureState(
	payload map[string]any,
	state string,
	leaseExpiresAt string,
) map[string]any {
	result := map[string]any{
		"schema_version":   raylineremote.TransactionSchemaVersion,
		"decision_id":      payload["decision_id"],
		"receipt":          payload["receipt"],
		"state":            state,
		"route_call_index": 4,
	}
	if leaseExpiresAt != "" {
		result["lease_expires_at"] = leaseExpiresAt
	}
	return result
}

func writeRemoteFixtureJSON(
	t *testing.T,
	writer http.ResponseWriter,
	value any,
) {
	t.Helper()
	writer.Header().Set("content-type", "application/json")
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		t.Error(err)
	}
}
