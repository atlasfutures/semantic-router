package execution

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Only the external llama.cpp HTTP API is simulated. The real counter,
// qualification, codec, renderer and token-prefix comparison run unchanged.
type llamaEndpointFixture struct {
	mu             sync.Mutex
	template       string
	prefill        bool
	rewriteHistory bool
	counted        []string
}

func llamaCounterFixture(t *testing.T) (Deployment, *llamaEndpointFixture) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	model := []byte("external-runtime-model-fixture")
	if err := os.WriteFile(path, model, 0600); err != nil {
		t.Fatal(err)
	}
	fixture := &llamaEndpointFixture{template: "external-runtime-template-fixture"}
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fixture.mu.Lock()
		defer fixture.mu.Unlock()
		switch r.URL.Path {
		case "/props":
			writeJSON(w, 200, map[string]any{"model_path": path, "model_alias": "qwen-fixture", "build_info": "b11840-36e36e63a", "chat_template": fixture.template, "endpoint_props": false, "modalities": map[string]bool{"vision": false, "audio": false}})
		case "/apply-template":
			var body struct {
				Messages []struct {
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
				Tools      json.RawMessage `json:"tools"`
				Generation *bool           `json:"add_generation_prompt"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			prompt := string(body.Tools)
			for _, message := range body.Messages {
				prompt += "<" + message.Role + ">" + string(message.Content) + "</end>"
			}
			if body.Generation == nil || *body.Generation || fixture.prefill {
				prompt += "<assistant>"
			}
			if fixture.rewriteHistory && body.Generation != nil && !*body.Generation && len(body.Tools) > 0 {
				prompt = "changed-history-template:" + prompt
			}
			writeJSON(w, 200, map[string]string{"prompt": prompt})
		case "/tokenize":
			var body struct {
				Content      string `json:"content"`
				AddSpecial   bool   `json:"add_special"`
				ParseSpecial bool   `json:"parse_special"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if !body.AddSpecial || !body.ParseSpecial {
				t.Error("counter did not match serving special-token flags")
			}
			fixture.counted = append(fixture.counted, body.Content)
			// A deterministic test vocabulary supplied by the external API,
			// including a BOS token, not an estimate used by the executor.
			tokens := []int{1}
			for _, character := range body.Content {
				tokens = append(tokens, int(character)+100)
			}
			writeJSON(w, 200, map[string]any{"tokens": tokens})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(endpoint.Close)
	deployment := testConfig(t, endpoint.URL+"/v1/chat/completions", false).Deployments[0]
	deployment.PromptCounter = &LlamaCPPConfig{ModelPath: path, ModelSHA256: fingerprint(model), BuildInfo: "b11840-36e36e63a", ChatTemplateSHA256: fingerprint([]byte(fixture.template))}
	return deployment, fixture
}

func TestLlamaCounterCountsFullPreparedPromptAndExactHistoryPrefix(t *testing.T) {
	deployment, fixture := llamaCounterFixture(t)
	counter, err := newLlamaCPPCounter(t.Context(), deployment)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"qwen-fixture","tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":"system rules"},{"role":"user","content":"first; private append"},{"role":"assistant","content":"answer"},{"role":"user","content":"next; private append"}]}`)
	prompt := cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "e1", Body: body}
	measured, err := cachecontext.MeasurePrompt(t.Context(), counter, prompt)
	if err != nil {
		t.Fatal(err)
	}
	prefix := measured.Prefix(measured)
	if prefix == nil || prefix.Tokens == 0 || prefix.OverlapTokens == nil || *prefix.OverlapTokens != prefix.Tokens || prefix.Provenance != "provider_reported" {
		t.Fatalf("serving counter evidence = %+v", prefix)
	}
	fixture.mu.Lock()
	if len(fixture.counted) != 2 || !strings.Contains(fixture.counted[0], "read_file") || !strings.Contains(fixture.counted[0], "system rules") || !strings.Contains(fixture.counted[0], "next; private append") || strings.Contains(fixture.counted[1], "next; private append") {
		t.Fatalf("counter did not count complete provider material and carried history: %v", fixture.counted)
	}
	fixture.rewriteHistory = true
	fixture.mu.Unlock()
	changed, err := cachecontext.MeasurePrompt(t.Context(), counter, prompt)
	if err != nil {
		t.Fatal(err)
	}
	if changed.Prefix(measured) != nil {
		t.Fatal("template rewrite was mislabeled as a measured carried-prefix length")
	}
}

func TestLlamaCounterRefusesQualificationDriftAndUnsupportedInput(t *testing.T) {
	for _, name := range []string{"template drift", "model file changed", "prefill enabled", "model hash mismatch", "multimodal", "cancelled hash"} {
		t.Run(name, func(t *testing.T) {
			deployment, fixture := llamaCounterFixture(t)
			ctx := t.Context()
			if name == "model hash mismatch" {
				deployment.PromptCounter.ModelSHA256 = strings.Repeat("0", 64)
			}
			if name == "prefill enabled" {
				fixture.prefill = true
			}
			if name == "cancelled hash" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			counter, err := newLlamaCPPCounter(ctx, deployment)
			if name == "model hash mismatch" || name == "prefill enabled" || name == "cancelled hash" {
				if err == nil {
					t.Fatal("unqualified runtime admitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			body := []byte(`{"model":"qwen-fixture","messages":[{"role":"user","content":"question"}]}`)
			switch name {
			case "template drift":
				fixture.mu.Lock()
				fixture.template = "different template"
				fixture.mu.Unlock()
			case "model file changed":
				if err := os.WriteFile(deployment.PromptCounter.ModelPath, []byte("changed-model-file"), 0600); err != nil {
					t.Fatal(err)
				}
			case "multimodal":
				body = []byte(`{"model":"qwen-fixture","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`)
			}
			if _, err := cachecontext.MeasurePrompt(ctx, counter, cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "e1", Body: body}); err == nil {
				t.Fatal("runtime drift or uncountable input produced evidence")
			}
		})
	}
}

func TestConfiguredCounterReusesPreparedMeasurementWithoutAdvancingState(t *testing.T) {
	deployment, fixture := llamaCounterFixture(t)
	config := testConfig(t, deployment.URL, true)
	deployment.RegistryProvider = "openrouter"
	config.Deployments[0] = deployment
	server, _ := executorWithHTTPDecisions(t, config, &decisionCapture{})
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	state, err := server.states.acquire(t.Context(), "counter-reuse", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	selected := server.actions[config.Actions[0].ID]
	prepare := func(body []byte) *preparedDispatch {
		t.Helper()
		request, envelope, _, err := server.engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := server.prepare(request, envelope, selected, state)
		if err != nil {
			t.Fatal(err)
		}
		return prepared
	}
	first := prepare([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"first"}]}`))
	measured, err := server.measurePreparedPrompt(t.Context(), selected, first.body, state, nil)
	if err != nil {
		t.Fatal(err)
	}
	completedAt := time.Now()
	// Seed a committed call with the same checkpoint the successful execution
	// branch stores. The next consult must only stage another prompt.
	state.placements[first.key] = first.placer.State()
	state.workers[deployment.ID] = WorkerEvidence{DeploymentID: deployment.ID, Call: cachecontext.CallCache{ActionID: selected.action.ID, CompletedAt: completedAt}, prompt: measured}
	before, _ := json.Marshal(state.placements)
	nextBody := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"first"},{"role":"assistant","content":"answer"},{"role":"user","content":"next"}]}`)
	evidence, measurements, err := server.measureExecutionContext(t.Context(), nextBody, llmprotocol.OpenAIChatV1, []string{"user", "assistant", "user"}, state)
	if err != nil {
		t.Fatal(err)
	}
	if len(evidence.Workers) != 1 || evidence.Workers[0].PrefixTokens == nil || evidence.Workers[0].OverlapTokens == nil || evidence.Workers[0].PrefixProvenance != "provider_reported" || evidence.Workers[0].Residency != nil {
		t.Fatalf("qualified prefix evidence missing or overclaimed: %+v", evidence.Workers)
	}
	next := prepare(nextBody)
	reused, err := server.measurePreparedPrompt(t.Context(), selected, next.body, state, measurements)
	if err != nil || reused != measurements[deployment.ID] {
		t.Fatalf("same prepared prompt was not reused: %v", err)
	}
	after, _ := json.Marshal(state.placements)
	if string(before) != string(after) || state.completed != 0 || state.workers[deployment.ID].prompt != measured || !state.workers[deployment.ID].Call.CompletedAt.Equal(completedAt) {
		t.Fatal("prefix consult committed hidden history or completion evidence")
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if len(fixture.counted) != 3 || !strings.Contains(fixture.counted[0], "Until the next steering instruction") {
		t.Fatalf("expected first/full/history counts including real private append, got %d", len(fixture.counted))
	}
}
