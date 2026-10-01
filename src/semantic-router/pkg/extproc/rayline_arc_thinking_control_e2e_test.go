//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The shared v5 fixture's actions: Opus native-only, and GLM at the neutral
// level and at "up", all at the default base.
const (
	v5OpusAction = "0527fe7631783f956fc850185d3130926ed252b9e49e005eba68a8f884152321"
	v5GLMNone    = "9d2b81063cac1733ef6859b8fc061ba5192366aa0966c7ca5e055110041aa332"
	v5GLMUp      = "b18e89bdb29cfb16b9cf71e46207ebde0ad463e5256f4b2c837280d84f13d7d7"
	v5UpText     = "Until the next steering instruction, reason more thoroughly before acting."
	v5Neutral    = "Until the next steering instruction, use your normal judgement."
	v5Golden     = "../selection/raylinearc/thinkingcontrol/testdata/golden"
)

// v5Router serves the v5 fixture with GLM on glmFormat (openai or anthropic)
// and Opus on OpenRouter's Messages wire, and returns the fake service.
func v5Router(t *testing.T, glmFormat string) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	manifest, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "package.json")
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	packageSHA := hex.EncodeToString(sum[:])
	fake := newFakePolicyService(t, policyTestAlias, packageSHA, []string{v5OpusAction, v5GLMNone, v5GLMUp})
	rendered := strings.NewReplacer(
		"{{POLICY_URL}}", fake.URL(), "{{ALIAS}}", policyTestAlias, "{{PACKAGE}}", packageSHA,
		"{{MANIFEST}}", manifestPath, "{{GLM_FORMAT}}", glmFormat,
		"{{OPUS}}", v5OpusAction, "{{GLM_NONE}}", v5GLMNone, "{{GLM_UP}}", v5GLMUp,
	).Replace(v5ConfigTemplate)
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	return router, fake
}

// v5Turn dispatches one client Messages body, commits the turn as a 200, and
// returns the provider-bound bytes and the request context.
func v5Turn(t *testing.T, router *OpenAIRouter, fake *fakePolicyService, action, episode, client string) ([]byte, *RequestContext) {
	t.Helper()
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: fmt.Sprintf("v5-%s-%d", episode, time.Now().UnixNano()),
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"}, {Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"}, {Key: "x-rayline-session", Value: episode},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("no provider body was written")
	}
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	finalizeSelectionProcessTerminal(ctx)
	return mutation.GetBody(), ctx
}

// A v5 action dispatched on Messages renders exactly what pathfinder's
// renderer renders: a captured Claude Code body, steered "up" on GLM at the
// default base, reaches the provider as the golden corpus's bytes.
func TestV5DispatchMatchesTheGoldenCorpusOnMessages(t *testing.T) {
	router, fake := v5Router(t, "anthropic")
	dir := filepath.Join(v5Golden, "messages", "claude_code_body_default")
	client, err := os.ReadFile(filepath.Join(dir, "client-00.json"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(dir, "provider-00.json"))
	if err != nil {
		t.Fatal(err)
	}
	// The capture names a placeholder model; the route is the auto alias.
	client = bytes.Replace(client, []byte(`"model":"MODEL"`), []byte(`"model":"auto"`), 1)
	got, ctx := v5Turn(t, router, fake, v5GLMUp, "v5-golden", string(client))
	if !bytes.Equal(got, want) {
		t.Fatalf("provider body differs from the golden:\n got %s\nwant %s", got, want)
	}
	trace := ctx.RaylineARCThinking
	if trace == nil || trace.Lever != thinkingControlLever || trace.LevelInForce != "up" ||
		trace.InstructionState != "steered" || trace.Written != "instruction" {
		t.Fatalf("trace = %+v", trace)
	}
}

// On Chat the steer is a text part on the tail user message; a return to
// none writes the neutral marker at the new tail and replays the steer at
// its anchor; the client's thinking fields never reach the provider.
func TestV5DispatchPlacesAndReplaysOnChat(t *testing.T) {
	router, fake := v5Router(t, "openai")
	first := `{"model":"auto","max_tokens":1024,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},` +
		`"messages":[{"role":"user","content":"fix the failing test"}]}`
	body, _ := v5Turn(t, router, fake, v5GLMUp, "v5-chat", first)
	var provider struct {
		Model           string            `json:"model"`
		Messages        []json.RawMessage `json:"messages"`
		Reasoning       json.RawMessage   `json:"reasoning"`
		ReasoningEffort json.RawMessage   `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(body, &provider); err != nil {
		t.Fatal(err)
	}
	if provider.Model != "z-ai/glm-5.3-flash" || provider.Reasoning != nil || provider.ReasoningEffort != nil {
		t.Fatalf("model %q reasoning %s effort %s", provider.Model, provider.Reasoning, provider.ReasoningEffort)
	}
	tail := string(provider.Messages[len(provider.Messages)-1])
	if !strings.Contains(tail, `{"type":"text","text":"`+v5UpText) {
		t.Fatalf("the steer is not the tail's last part: %s", tail)
	}

	second := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":"Looking."},{"role":"user","content":"go on"}]}`
	body, ctx := v5Turn(t, router, fake, v5GLMNone, "v5-chat", second)
	if err := json.Unmarshal(body, &provider); err != nil {
		t.Fatal(err)
	}
	if len(provider.Messages) < 3 {
		t.Fatalf("messages = %d", len(provider.Messages))
	}
	anchor := string(provider.Messages[len(provider.Messages)-3])
	if !strings.Contains(anchor, v5UpText) {
		t.Fatalf("the steer was not replayed at its anchor: %s", anchor)
	}
	if tail := string(provider.Messages[len(provider.Messages)-1]); !strings.Contains(tail, v5Neutral) {
		t.Fatalf("the neutral marker is not on the new tail: %s", tail)
	}
	if trace := ctx.RaylineARCThinking; trace == nil || trace.InstructionState != "neutral_marker" || trace.Written != "neutral_marker" {
		t.Fatalf("trace = %+v", ctx.RaylineARCThinking)
	}
}

// A native-only action renders no instruction, and its base owns the
// thinking fields: a default base sends none of the client's.
func TestV5NativeOnlyActionStripsTheClientsThinking(t *testing.T) {
	router, fake := v5Router(t, "openai")
	client := `{"model":"auto","max_tokens":1024,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},` +
		`"messages":[{"role":"user","content":"hello"}]}`
	body, ctx := v5Turn(t, router, fake, v5OpusAction, "v5-native", client)
	var provider map[string]json.RawMessage
	if err := json.Unmarshal(body, &provider); err != nil {
		t.Fatal(err)
	}
	if _, present := provider["thinking"]; present {
		t.Fatalf("the client's thinking travelled: %s", body)
	}
	if strings.Contains(string(provider["output_config"]), "effort") || strings.Contains(string(body), "steering instruction") {
		t.Fatalf("the client's effort or a steer travelled: %s", body)
	}
	if trace := ctx.RaylineARCThinking; trace == nil || trace.InstructionState != "" || trace.Written != "" {
		t.Fatalf("trace = %+v", ctx.RaylineARCThinking)
	}
}

const v5ConfigTemplate = `version: v0.3

providers:
  defaults:
    model: glm
  models:
    - name: glm
      provider_model_id: z-ai/glm-5.3-flash
      api_format: {{GLM_FORMAT}}
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openrouter-glm
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: opus
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openrouter-opus
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: glm
      modality: text
    - name: opus
      modality: text
  decisions:
    - name: rayline-arc-v5-e2e
      description: Policy-service ARC route serving a v5 package
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: glm
          use_reasoning: true
        - model: opus
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
          policy_service:
            base_url: {{POLICY_URL}}
            total_timeout_seconds: 5
            package_alias: {{ALIAS}}
            package_sha256: {{PACKAGE}}
            package_manifest: {{MANIFEST}}
            allow_experimental_controls: true
            model_schedule: task_turn_compaction_v1
            bindings:
              - action_id: {{OPUS}}
                worker: opus
              - action_id: {{GLM_NONE}}
                worker: glm
              - action_id: {{GLM_UP}}
                worker: glm
          episode:
            id_header: x-rayline-session
            backend: memory
            key_prefix: "vsr:rayline-arc-v5-e2e:"
            acquire_timeout_seconds: 2
            lease_ttl_seconds: 3
            idle_ttl_seconds: 900
            max_in_memory_episodes: 128
            development_mode: true

global:
  services:
    response_api:
      enabled: true
      store_backend: memory
  stores:
    semantic_cache:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: ""
`
