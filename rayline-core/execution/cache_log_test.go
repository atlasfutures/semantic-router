package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

func TestCacheCompletionLogRequiresSuccessfulExecution(t *testing.T) {
	for _, outcome := range []string{"complete", "upstream failure", "truncated stream", "refusal"} {
		t.Run(outcome, func(t *testing.T) {
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
			defer slog.SetDefault(previous)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch outcome {
				case "upstream failure":
					writeJSON(w, 503, map[string]any{"error": map[string]string{"message": "unavailable", "type": "server_error"}})
				case "truncated stream":
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n")
				case "refusal":
					_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","refusal":"cannot answer"},"finish_reason":"stop"}]}`)
				default:
					_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21,"prompt_tokens_details":{"cached_tokens":12}}}`)
				}
			}))
			defer upstream.Close()
			server, endpoint := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), &decisionCapture{})
			now := time.Unix(1700000000, 0).UTC()
			server.now = func() time.Time { return now }
			body := fmt.Sprintf(`{"model":"arc-1.0","messages":[{"role":"user","content":"private cache log sentinel"}],"stream":%v}`, outcome == "truncated stream")
			executeHTTP(t, endpoint.URL, "/v1/chat/completions", body, "cache-log")
			state, err := server.states.acquire(t.Context(), "cache-log", "main")
			if err != nil {
				t.Fatal(err)
			}
			defer state.release()
			var completed []map[string]any
			decoder := json.NewDecoder(bytes.NewReader(logs.Bytes()))
			for decoder.More() {
				var event map[string]any
				if err := decoder.Decode(&event); err != nil {
					t.Fatal(err)
				}
				if event["msg"] == "execution completed" {
					completed = append(completed, event)
				}
			}
			if strings.Contains(logs.String(), "private cache log sentinel") || strings.Contains(logs.String(), testCoreToken) {
				t.Fatal("request material entered cache log")
			}
			if outcome != "complete" {
				if len(completed) != 0 || state.completed != 0 || len(state.workers) != 0 {
					t.Fatalf("failed execution advanced completion evidence: %s", logs.Bytes())
				}
				return
			}
			if len(completed) != 1 || state.completed != 1 || len(state.workers) != 1 {
				t.Fatal("successful execution omitted committed evidence")
			}
			cache := completed[0]["cache"].(map[string]any)
			call := cache["current_call"].(map[string]any)
			if cache["completed_deployment_records"] != float64(1) || call["completed_at"] != now.Format(time.RFC3339) || call["prompt_tokens"] != float64(20) || call["cache_read_tokens"] != float64(12) || call["usage_provenance"] != "provider_reported" || call["cache_write_tokens"] != float64(0) || call["ttl_s"] != nil || call["ttl_source"] != nil || call["current_prefix"] != nil || call["residency"] != nil {
				t.Fatalf("cache log invented or lost evidence: %+v", cache)
			}
		})
	}
}

func TestCacheLogRetainsWorkerClocksAndQualifiedPrefixWithoutWarmthClaim(t *testing.T) {
	now := time.Unix(1700000000, 0).UTC()
	state := &branchState{completed: 3, workers: map[string]WorkerEvidence{
		"local": {DeploymentID: "local", Call: cachecontext.CallCache{ActionID: "local-action", CompletedAt: now.Add(-time.Minute)}},
		"cloud": {DeploymentID: "cloud", Call: cachecontext.CallCache{ActionID: "cloud-action", CompletedAt: now, TTLSeconds: 300, TTLSource: "anthropic cache_control ephemeral 5m"}},
	}}
	prefix, overlap := 110, 90
	evidence := ExecutionContext{Workers: []WorkerEvidence{{DeploymentID: "local", PrefixTokens: &prefix, OverlapTokens: &overlap, TokenUnits: "qualified-runtime", PrefixProvenance: "provider_reported"}}}
	log := completionCacheEvidence(state, "cloud", evidence, nil)
	if log.CompletedDeploymentRecords != 2 || log.CompletedTurns != 3 || log.CurrentCall.TTLSeconds == nil || *log.CurrentCall.TTLSeconds != 300 || log.CurrentCall.CurrentPrefix != nil || log.CurrentCall.Residency != nil {
		t.Fatalf("current call evidence changed: %+v", log)
	}
	local := log.Workers[1]
	if local.Deployment != "local" || !local.CompletedAt.Equal(now.Add(-time.Minute)) || local.CurrentPrefix == nil || local.CurrentPrefix.Tokens != prefix || local.CurrentPrefix.OverlapTokens == nil || *local.CurrentPrefix.OverlapTokens != overlap || local.CurrentPrefix.Provenance != "provider_reported" || local.Residency != nil || local.TTLSeconds != nil || local.UsageProvenance != nil {
		t.Fatalf("retained worker evidence changed: %+v", local)
	}
	if state.workers["local"].PrefixTokens != nil || !state.workers["local"].Call.CompletedAt.Equal(now.Add(-time.Minute)) {
		t.Fatal("logging mutated the execution ledger")
	}
}
