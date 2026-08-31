package raylineremote

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

//nolint:cyclop // The table covers the bounded wire-input validation contract.
func TestChatRequestAndOpaqueIdentifiers(t *testing.T) {
	rawEpisode := "private-episode-canary"
	hmacKey := []byte("0123456789abcdef0123456789abcdef")
	episodeKey, err := DeriveEpisodeKey(rawEpisode, hmacKey)
	if err != nil {
		t.Fatal(err)
	}
	if !episodeKeyPattern.MatchString(episodeKey) ||
		strings.Contains(episodeKey, rawEpisode) {
		t.Fatalf("unexpected episode key %q", episodeKey)
	}
	otherKey, err := DeriveEpisodeKey(
		rawEpisode,
		[]byte("abcdef0123456789abcdef0123456789"),
	)
	if err != nil || otherKey == episodeKey {
		t.Fatal("episode HMAC did not depend on the secret key")
	}

	decisionID, err := MintDecisionID()
	if err != nil {
		t.Fatal(err)
	}
	if !decisionIDPattern.MatchString(decisionID) {
		t.Fatalf("invalid minted decision ID %q", decisionID)
	}

	body := []byte(`{
		"model":"rayline/worker-a",
		"messages":[{"role":"user","content":"prompt-canary"}],
		"tools":[{"type":"function","function":{"name":"tool-canary"}}],
		"authorization":"must-not-cross"
	}`)
	request, err := ChatRequestFromBody(body)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	content := string(encoded)
	for _, forbidden := range []string{
		`"model"`,
		"rayline/worker-a",
		"authorization",
		"must-not-cross",
	} {
		if strings.Contains(content, forbidden) {
			t.Fatalf("policy request exposed %q: %s", forbidden, content)
		}
	}
	for _, required := range []string{
		OpenAIChatProtocol,
		"prompt-canary",
		"tool-canary",
	} {
		if !strings.Contains(content, required) {
			t.Fatalf("policy request omitted %q: %s", required, content)
		}
	}
}

func TestChatRequestRejectsMalformedShapes(t *testing.T) {
	tests := [][]byte{
		nil,
		[]byte(`not-json`),
		[]byte(`[]`),
		[]byte(`{"messages":[]}`),
		[]byte(`{"messages":["not-an-object"]}`),
		[]byte(`{"messages":[{"role":"user"}],"tools":["bad"]}`),
	}
	for index, body := range tests {
		if _, err := ChatRequestFromBody(body); err == nil ||
			!IsFailureClass(err, FailureRequest) {
			t.Fatalf("case %d error = %v", index, err)
		}
	}
	if _, err := DeriveEpisodeKey(
		"episode",
		[]byte("short"),
	); err == nil {
		t.Fatal("short HMAC secret was accepted")
	}
}

//nolint:cyclop // One scenario verifies every state transition and idempotent replay.
func TestClientPrepareAndTransactionLifecycle(t *testing.T) {
	fixture := newProtocolFixture(t)
	defer fixture.server.Close()
	client := fixture.client(t)
	if err := client.CheckReadiness(context.Background()); err != nil {
		t.Fatal(err)
	}
	transaction, result, err := client.Prepare(
		context.Background(),
		validPrepareInput(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedWorker != "worker-b" ||
		result.RouteCallIndex != 3 ||
		transaction.SelectedWorker() != "worker-b" ||
		transaction.RouteCallIndex() != 3 {
		t.Fatalf(
			"unexpected selection: result=%#v worker=%q index=%d",
			result,
			transaction.SelectedWorker(),
			transaction.RouteCallIndex(),
		)
	}
	if err := transaction.ValidateDispatch(
		context.Background(),
	); err != nil {
		t.Fatal(err)
	}
	if err := transaction.CommitOnHeaders(
		context.Background(),
		http.StatusOK,
	); err != nil {
		t.Fatal(err)
	}
	inputTokens := 21
	outputTokens := 8
	latency := 12.5
	cost := 0.003
	outcome := ActualOutcome{
		OutcomeClass: OutcomeSuccess,
		StatusCode:   http.StatusOK,
		InputTokens:  &inputTokens,
		OutputTokens: &outputTokens,
		LatencyMS:    &latency,
		CostUSD:      &cost,
	}
	if err := transaction.Settle(
		context.Background(),
		outcome,
	); err != nil {
		t.Fatal(err)
	}
	// Local idempotency prevents duplicate lifecycle calls from crossing the
	// service boundary.
	if err := transaction.CommitOnHeaders(
		context.Background(),
		http.StatusOK,
	); err != nil {
		t.Fatal(err)
	}
	if err := transaction.Settle(
		context.Background(),
		outcome,
	); err != nil {
		t.Fatal(err)
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	for operation, want := range map[string]int{
		"capabilities": 1,
		"workers":      1,
		"prepare":      1,
		"renew":        1,
		"commit":       1,
		"settle":       1,
	} {
		if fixture.calls[operation] != want {
			t.Errorf(
				"%s calls = %d, want %d",
				operation,
				fixture.calls[operation],
				want,
			)
		}
	}
	if strings.Contains(
		string(fixture.lifecycleBodies["settle"]),
		`"input_tokens":0`,
	) {
		t.Fatal("settlement fabricated zero-valued input evidence")
	}
}

func TestClientAbortIsIdempotentAndDoesNotCommit(t *testing.T) {
	fixture := newProtocolFixture(t)
	defer fixture.server.Close()
	transaction, _, err := fixture.client(t).Prepare(
		context.Background(),
		validPrepareInput(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := transaction.Abort(
		context.Background(),
		AbortProviderNetwork,
	); err != nil {
		t.Fatal(err)
	}
	if err := transaction.Abort(
		context.Background(),
		AbortProviderNetwork,
	); err != nil {
		t.Fatal(err)
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if fixture.calls["abort"] != 1 ||
		fixture.calls["commit"] != 0 {
		t.Fatalf("calls = %#v", fixture.calls)
	}
}

func TestClientRejectsPrepareResponseDrift(t *testing.T) {
	mutations := []struct {
		name   string
		mutate func(map[string]any)
		class  FailureClass
	}{
		{
			name: "wrong decision",
			mutate: func(response map[string]any) {
				response["decision_id"] = "rt_other"
			},
			class: FailureContract,
		},
		{
			name: "wrong worker",
			mutate: func(response map[string]any) {
				response["selected_worker"] = "worker-c"
			},
			class: FailureContract,
		},
		{
			name: "wrong bundle",
			mutate: func(response map[string]any) {
				response["bundle_version"] = "bundle-other"
			},
			class: FailureContract,
		},
		{
			name: "bad receipt",
			mutate: func(response map[string]any) {
				response["receipt"] = "receipt with spaces"
			},
			class: FailureContract,
		},
		{
			name: "expired lease",
			mutate: func(response map[string]any) {
				response["lease_expires_at"] = "2020-01-01T00:00:00Z"
			},
			class: FailureLease,
		},
		{
			name: "unknown field",
			mutate: func(response map[string]any) {
				response["provider_headers"] = map[string]any{
					"authorization": "secret",
				}
			},
			class: FailureDecode,
		},
	}
	for _, test := range mutations {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(
				writer http.ResponseWriter,
				request *http.Request,
			) {
				response := validPrepareResponse()
				test.mutate(response)
				writeJSON(t, writer, response)
			}))
			defer server.Close()
			client, err := NewClient(validClientConfig(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			_, _, err = client.Prepare(
				context.Background(),
				validPrepareInput(t),
			)
			if !IsFailureClass(err, test.class) {
				t.Fatalf("error = %v, want class %s", err, test.class)
			}
		})
	}
}

func TestClientBoundsErrorsRetriesAndTimeouts(t *testing.T) {
	t.Run("bounded status", testClientBoundedStatus)
	t.Run("retry safe status", testClientRetrySafeStatus)
	t.Run("timeout", testClientTimeout)
	t.Run("response limit", testClientResponseLimit)
}

func testClientBoundedStatus(t *testing.T) {
	privateBody := "private-prompt private-receipt private-api-key"
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		_ *http.Request,
	) {
		writer.WriteHeader(http.StatusConflict)
		_, _ = writer.Write([]byte(
			`{"error":{"code":"episode_busy","message":"` +
				privateBody + `"}}`,
		))
	}))
	defer server.Close()
	client, err := NewClient(validClientConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = client.Prepare(context.Background(), validPrepareInput(t))
	if err == nil ||
		!strings.Contains(err.Error(), "episode_busy") ||
		strings.Contains(err.Error(), privateBody) {
		t.Fatalf("unbounded error = %v", err)
	}
}

func testClientRetrySafeStatus(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		_ *http.Request,
	) {
		if calls.Add(1) == 1 {
			writer.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		writeJSON(t, writer, validPrepareResponse())
	}))
	defer server.Close()
	config := validClientConfig(server.URL)
	config.MaxRetries = 1
	client, err := NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := client.Prepare(
		context.Background(),
		validPrepareInput(t),
	); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func testClientTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		_ http.ResponseWriter,
		_ *http.Request,
	) {
		time.Sleep(100 * time.Millisecond)
	}))
	defer server.Close()
	config := validClientConfig(server.URL)
	config.ConnectTimeout = time.Millisecond
	config.RequestTimeout = 15 * time.Millisecond
	client, err := NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = client.Prepare(context.Background(), validPrepareInput(t))
	if !IsFailureClass(err, FailureTimeout) {
		t.Fatalf("error = %v, want timeout", err)
	}
}

func testClientResponseLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		_ *http.Request,
	) {
		writer.Header().Set("content-type", "application/json")
		_, _ = writer.Write([]byte(
			`{"padding":"` + strings.Repeat("x", maxWireBytes) + `"}`,
		))
	}))
	defer server.Close()
	client, err := NewClient(validClientConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = client.Prepare(context.Background(), validPrepareInput(t))
	if !IsFailureClass(err, FailureDecode) {
		t.Fatalf("error = %v, want decode", err)
	}
}

func TestClientRetriesPreResponseTransportFailure(t *testing.T) {
	var calls atomic.Int32
	transport := roundTripFunc(func(
		request *http.Request,
	) (*http.Response, error) {
		if calls.Add(1) == 1 {
			return nil, errors.New("private network detail")
		}
		recorder := httptest.NewRecorder()
		writeJSON(t, recorder, validPrepareResponse())
		return recorder.Result(), nil
	})
	config := validClientConfig("http://rayline.test")
	config.MaxRetries = 1
	client, err := newClient(
		config,
		&http.Client{Transport: transport},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := client.Prepare(
		context.Background(),
		validPrepareInput(t),
	); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

type protocolFixture struct {
	t               *testing.T
	server          *httptest.Server
	mu              sync.Mutex
	calls           map[string]int
	lifecycleBodies map[string][]byte
}

func newProtocolFixture(t *testing.T) *protocolFixture {
	t.Helper()
	fixture := &protocolFixture{
		t:               t,
		calls:           make(map[string]int),
		lifecycleBodies: make(map[string][]byte),
	}
	fixture.server = httptest.NewServer(http.HandlerFunc(fixture.serveHTTP))
	return fixture
}

func (fixture *protocolFixture) client(t *testing.T) *Client {
	t.Helper()
	client, err := NewClient(validClientConfig(fixture.server.URL))
	if err != nil {
		t.Fatal(err)
	}
	return client
}

//nolint:cyclop // The in-memory server models the complete transaction protocol.
func (fixture *protocolFixture) serveHTTP(
	writer http.ResponseWriter,
	request *http.Request,
) {
	fixture.t.Helper()
	if request.Header.Get("authorization") != "Bearer test-api-key" {
		fixture.t.Errorf("missing independent service authorization")
		writer.WriteHeader(http.StatusUnauthorized)
		return
	}
	operation := strings.TrimPrefix(
		request.URL.Path,
		"/v1/route/",
	)
	if request.URL.Path == "/v1/workers" {
		operation = "workers"
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		fixture.t.Error(err)
	}
	fixture.mu.Lock()
	fixture.calls[operation]++
	fixture.lifecycleBodies[operation] = append([]byte(nil), body...)
	fixture.mu.Unlock()

	switch operation {
	case "capabilities":
		writeJSON(fixture.t, writer, map[string]any{
			"schema_version":             CapabilitiesSchemaVersion,
			"transaction_schema_version": TransactionSchemaVersion,
			"bundle_version":             "bundle-test-v1",
			"protocols":                  []string{OpenAIChatProtocol},
			"operations": []string{
				"prepare",
				"renew",
				"commit",
				"abort",
				"settle",
			},
			"workers":         []string{"worker-a", "worker-b"},
			"lease_seconds":   30,
			"pending_journal": pendingJournalMVP,
		})
	case "workers":
		writeJSON(fixture.t, writer, []map[string]any{
			validWorker("worker-a", "provider-model-a"),
			validWorker("worker-b", "provider-model-b"),
		})
	case "prepare":
		if strings.Contains(string(body), `"model"`) {
			fixture.t.Error("client-controlled model crossed policy boundary")
		}
		writeJSON(fixture.t, writer, validPrepareResponse())
	case "renew":
		writeJSON(fixture.t, writer, validStateResponse(
			"prepared",
			time.Now().Add(30*time.Second).Format(time.RFC3339Nano),
		))
	case "commit":
		writeJSON(fixture.t, writer, validStateResponse("committed", ""))
	case "abort":
		writeJSON(fixture.t, writer, validStateResponse("aborted", ""))
	case "settle":
		writeJSON(fixture.t, writer, validStateResponse("settled", ""))
	default:
		writer.WriteHeader(http.StatusNotFound)
	}
}

// The client is the last seam before prompt bodies leave the process, so it
// refuses a plaintext endpoint on its own rather than trusting the caller to
// have validated the config.
func TestNewClientRefusesPlaintextBaseURLWithoutOptIn(t *testing.T) {
	config := validClientConfig("http://rayline-router:8000")
	config.AllowInsecureTransport = false
	client, err := NewClient(config)
	if client != nil {
		t.Fatal("plaintext base URL produced a usable client")
	}
	var failure *Failure
	if !errors.As(err, &failure) ||
		failure.Code != "insecure_base_url" {
		t.Fatalf("error = %v, want failure code insecure_base_url", err)
	}
}

func TestNewClientAcceptsTLSBaseURLWithoutOptIn(t *testing.T) {
	config := validClientConfig("https://rayline-router:8443")
	config.AllowInsecureTransport = false
	if _, err := NewClient(config); err != nil {
		t.Fatalf("https base URL rejected: %v", err)
	}
}

func validClientConfig(baseURL string) ClientConfig {
	return ClientConfig{
		BaseURL: baseURL,
		// httptest serves plaintext on loopback, which is exactly the
		// hermetic case the opt-in exists for.
		AllowInsecureTransport: true,
		BundleVersion:          "bundle-test-v1",
		APIKey:                 "test-api-key",
		ConnectTimeout:         100 * time.Millisecond,
		RequestTimeout:         time.Second,
		LeaseTTL:               30 * time.Second,
		MaxRetries:             1,
		Workers: []WorkerContract{
			{
				ID:           "worker-a",
				Model:        "provider-model-a",
				Backend:      "mock",
				ThinkingMode: "disabled",
				Prices: &WorkerPrices{
					Input:      0.000001,
					CacheRead:  0.0000001,
					CacheWrite: 0.000001,
					Output:     0.000002,
				},
			},
			{
				ID:           "worker-b",
				Model:        "provider-model-b",
				Backend:      "mock",
				ThinkingMode: "enabled",
				Prices: &WorkerPrices{
					Input:      0.000001,
					CacheRead:  0.0000001,
					CacheWrite: 0.000001,
					Output:     0.000002,
				},
			},
		},
	}
}

func validPrepareInput(t *testing.T) PrepareInput {
	t.Helper()
	request, err := ChatRequestFromBody([]byte(`{
		"model":"auto",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","function":{"name":"search"}}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	episodeKey, err := DeriveEpisodeKey(
		"episode-private",
		[]byte("0123456789abcdef0123456789abcdef"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return PrepareInput{
		DecisionID: "rt_test-decision",
		EpisodeKey: episodeKey,
		Candidates: []string{"worker-a", "worker-b"},
		Request:    request,
	}
}

func validPrepareResponse() map[string]any {
	return map[string]any{
		"schema_version":   TransactionSchemaVersion,
		"decision_id":      "rt_test-decision",
		"receipt":          "rsel_valid_receipt",
		"state":            "prepared",
		"selected_worker":  "worker-b",
		"policy":           "test-policy",
		"route_call_index": 3,
		"bundle_version":   "bundle-test-v1",
		"lease_expires_at": time.Now().
			Add(30 * time.Second).
			Format(time.RFC3339Nano),
	}
}

func validStateResponse(
	state string,
	leaseExpiresAt string,
) map[string]any {
	response := map[string]any{
		"schema_version":   TransactionSchemaVersion,
		"decision_id":      "rt_test-decision",
		"receipt":          "rsel_valid_receipt",
		"state":            state,
		"route_call_index": 3,
	}
	if leaseExpiresAt != "" {
		response["lease_expires_at"] = leaseExpiresAt
	}
	return response
}

func validWorker(id string, model string) map[string]any {
	thinkingMode := map[string]string{
		"worker-a": "disabled",
		"worker-b": "enabled",
	}[id]
	if thinkingMode == "" {
		thinkingMode = "disabled"
	}
	return map[string]any{
		"id":                         id,
		"backend":                    "mock",
		"model":                      model,
		"thinking_mode":              thinkingMode,
		"openrouter_allow_fallbacks": false,
		"capability_tags":            []string{"text"},
		"pricing_snapshot_version":   "prices-v1",
		"per_token_prices": map[string]any{
			"input":       0.000001,
			"cache_read":  0.0000001,
			"cache_write": 0.000001,
			"output":      0.000002,
		},
	}
}

func writeJSON(
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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(
	request *http.Request,
) (*http.Response, error) {
	return function(request)
}
