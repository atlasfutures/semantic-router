//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func awaitTCPAccept(t *testing.T, port int, budget time.Duration, serveErr <-chan error) {
	t.Helper()
	address := net.JoinHostPort("127.0.0.1", fmt.Sprint(port))
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", address, time.Second)
		if err == nil {
			_ = conn.Close()
			return
		}
		select {
		case err := <-serveErr:
			t.Fatalf("the router stopped before it listened: %v", err)
		default:
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("the gRPC port %d did not accept a connection within %s", port, budget)
}

func awaitARCArmed(t *testing.T, budget time.Duration) {
	t.Helper()
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		if arcComponentReadyGauge() == 1 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("ARC never armed within %s after the encoder started answering", budget)
}

func arcComponentReadyGauge() float64 {
	return testutil.ToFloat64(
		metrics.RaylineARCComponentReady.WithLabelValues("artifact_head_encoder"),
	)
}

func freeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve a port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("release the reserved port: %v", err)
	}
	return port
}

// hangingARCEncoder accepts the connection and then holds the request open,
// which is the failure this test is about: a refused connection fails fast and
// never had the chance to block startup.
type hangingARCEncoder struct {
	server   *httptest.Server
	released chan struct{}
	release  sync.Once
	attempts atomic.Int64
}

func newHangingARCEncoder(t *testing.T) *hangingARCEncoder {
	t.Helper()
	encoder := &hangingARCEncoder{released: make(chan struct{})}
	encoder.server = httptest.NewServer(http.HandlerFunc(encoder.serve))
	// Cleanups run last-registered-first, so the encoder is released before it
	// is closed. Without that, a failing test hangs to its own timeout instead
	// of reporting: Close waits for in-flight requests, and the router keeps
	// probing, so a held-open handler is replaced by the next one forever.
	t.Cleanup(encoder.server.Close)
	t.Cleanup(encoder.answer)
	return encoder
}

func (encoder *hangingARCEncoder) URL() string { return encoder.server.URL }

func (encoder *hangingARCEncoder) answer() {
	encoder.release.Do(func() { close(encoder.released) })
}

func (encoder *hangingARCEncoder) awaitAttempts(t *testing.T, want int64, budget time.Duration) {
	t.Helper()
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		if encoder.attempts.Load() >= want {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf(
		"the encoder saw %d probe attempts within %s, want %d: the router did not re-probe",
		encoder.attempts.Load(), budget, want,
	)
}

func (encoder *hangingARCEncoder) serve(writer http.ResponseWriter, request *http.Request) {
	encoder.attempts.Add(1)
	select {
	case <-encoder.released:
	case <-request.Context().Done():
		return
	}
	embedding := make([]float32, fixtureHistoryDimension)
	embedding[fixtureRoutingAxis] = 1
	response := map[string]any{
		"request_id": "arc-port-fixture-probe",
		"created_at": time.Now().Unix(),
		"data": map[string]any{
			"embedding":            embedding,
			"serialized_tokens":    120,
			"full_history_tokens":  120,
			"truncated_tokens":     0,
			"cached_prefix_tokens": 0,
			"serializer_version":   fixtureSerializer,
			"model":                fixtureEncoderModel,
			"model_revision":       fixtureEncoderRevision,
			"tokenizer_revision":   fixtureEncoderRevision,
			"tokenizer_sha256":     fixtureTokenizerSHA256,
			"eos_token_id":         fixtureEOSTokenID,
			"engine_build_id":      fixtureEngineBuildID,
			"io_plugin_version":    fixtureIOPluginVersion,
			"pooling_capabilities": []string{"chunked_causal_mean"},
		},
	}
	writer.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(writer).Encode(response)
}

// encoderBudget is the part of the config the two tests disagree about. A long
// budget exposes a router that waits for the encoder before it listens; a short
// one lets a hung probe die quickly so the re-probe can be observed.
type encoderBudget struct {
	totalTimeoutSeconds int
	retryInitialSeconds int
	retryMaxSeconds     int
}

func writeARCRouterConfig(
	t *testing.T,
	artifactDir, encoderURL, providerURL string,
	budget encoderBudget,
) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	config := strings.NewReplacer(
		"{{ARTIFACT_DIR}}", artifactDir,
		"{{ARTIFACT_REVISION}}", fixtureArtifactID,
		"{{ENCODER_URL}}", encoderURL,
		"{{ENCODER_MODEL}}", fixtureEncoderModel,
		"{{ENCODER_REVISION}}", fixtureEncoderRevision,
		"{{ENGINE_BUILD_ID}}", fixtureEngineBuildID,
		"{{IO_PLUGIN_VERSION}}", fixtureIOPluginVersion,
		"{{SERIALIZER}}", fixtureSerializer,
		"{{PROVIDER_URL}}", providerURL,
		"{{API_KEY_ENV}}", fixtureAPIKeyEnv,
		"{{WORKER_A}}", fixtureWorkerA,
		"{{WORKER_B}}", fixtureWorkerB,
		"{{TOTAL_TIMEOUT_SECONDS}}", fmt.Sprint(budget.totalTimeoutSeconds),
		"{{RETRY_INITIAL_SECONDS}}", fmt.Sprint(budget.retryInitialSeconds),
		"{{RETRY_MAX_SECONDS}}", fmt.Sprint(budget.retryMaxSeconds),
	).Replace(arcRouterConfigTemplate)
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatalf("write router config: %v", err)
	}
	return path
}

// One decision, the two fixture arms, and everything the router does not need
// switched off explicitly rather than left to a default, so no config change
// elsewhere can pull a model download into this test.
const arcRouterConfigTemplate = `version: v0.3

providers:
  defaults:
    default_model: {{WORKER_A}}
  models:
    - name: {{WORKER_A}}
      provider_model_id: synthetic/{{WORKER_A}}
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.5
        cache_write_per_1m: 1.5
        completion_per_1m: 2
      backend_refs:
        - name: synthetic-a
          base_url: {{PROVIDER_URL}}
          provider: openai
          type: openai
          api_key_env: {{API_KEY_ENV}}
    - name: {{WORKER_B}}
      provider_model_id: synthetic/{{WORKER_B}}
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 2
        cached_input_per_1m: 1
        cache_write_per_1m: 2.5
        completion_per_1m: 4
      backend_refs:
        - name: synthetic-b
          base_url: {{PROVIDER_URL}}
          provider: openai
          type: openai
          api_key_env: {{API_KEY_ENV}}

routing:
  modelCards:
    - name: {{WORKER_A}}
      modality: text
    - name: {{WORKER_B}}
      modality: text
  decisions:
    - name: rayline-arc-startup
      description: Synthetic ARC route for the startup port test
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: {{WORKER_A}}
          use_reasoning: false
        - model: {{WORKER_B}}
          use_reasoning: true
      adaptations:
        mode: bypass
      plugins:
        - type: router_replay
          configuration:
            enabled: false
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          artifact_dir: {{ARTIFACT_DIR}}
          artifact_revision: {{ARTIFACT_REVISION}}
          encoder:
            base_url: {{ENCODER_URL}}
            replicas: []
            membership: {}
            failover: {}
            model: {{ENCODER_MODEL}}
            model_revision: {{ENCODER_REVISION}}
            expected_build_id: {{ENGINE_BUILD_ID}}
            expected_io_plugin_version: {{IO_PLUGIN_VERSION}}
            serializer_version: {{SERIALIZER}}
            serving_rung: B
            required_pooling_capabilities:
              - chunked_causal_mean
            connect_timeout_seconds: 1
            # Each probe attempt is bounded only by this, and the next attempt
            # cannot start until this one ends, so it sets both how visible a
            # blocking construction path is and how soon a re-probe can run.
            total_timeout_seconds: {{TOTAL_TIMEOUT_SECONDS}}
            max_retries: 0
            max_inflight_encoder_calls: 4
            probe_retry_initial_seconds: {{RETRY_INITIAL_SECONDS}}
            probe_retry_max_seconds: {{RETRY_MAX_SECONDS}}
          episode:
            id_header: x-rayline-session
            close_header: ""
            backend: memory
            key_prefix: "vsr:rayline-arc-startup:"
            acquire_timeout_seconds: 2
            lease_ttl_seconds: 3
            idle_ttl_seconds: 900
            max_in_memory_episodes: 128
            development_mode: true

global:
  stores:
    semantic_cache:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: ""
        qwen3_model_path: ""
        gemma_model_path: ""
        bert_model_path: ""
        multimodal_model_path: ""
    modules:
      prompt_guard:
        enabled: false
        model_ref: ""
        model_id: ""
        jailbreak_mapping_path: ""
        use_mmbert_32k: false
      classifier:
        domain:
          model_ref: ""
          model_id: ""
          category_mapping_path: ""
          use_mmbert_32k: false
        pii:
          model_ref: ""
          model_id: ""
          pii_mapping_path: ""
          use_mmbert_32k: false
      feedback_detector:
        enabled: false
        model_ref: ""
        model_id: ""
        use_mmbert_32k: false
`
