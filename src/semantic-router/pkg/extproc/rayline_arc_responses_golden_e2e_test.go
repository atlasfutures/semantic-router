//go:build !windows && cgo

package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// Pathfinder's Responses golden corpus at 09c42c1e respects admission: every
// case is either no_lever, whose provider bytes are the client's apart from
// the model, or refused with the registry's message. Through the router the
// first must come out byte for byte and the second must be refused where VSR
// refuses it, at load, with the recorded message. The placement corpus is
// the renderer's oracle only and is not read here.
const responsesGoldenDir = "../selection/raylinearc/thinkingcontrol/testdata/golden/responses"

// responsesNativeDefaultControl is the registry's native-default control: no
// budget, no instruction.
const responsesNativeDefaultControl = "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0"

type responsesGoldenCase struct {
	Model     string `json:"model"`
	Provider  string `json:"provider"`
	WireModel string `json:"wire_model"`
	Admission string `json:"admission"`
	Calls     []struct {
		ClientBody   string  `json:"client_body"`
		ControlID    *string `json:"control_id"`
		ExpectedBody string  `json:"expected_body"`
	} `json:"calls"`
	Refusal *struct {
		Error string `json:"error"`
	} `json:"refusal"`
}

func TestResponsesGoldenCorpusThroughTheRouter(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	entries, err := os.ReadDir(responsesGoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		if entry.IsDir() {
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(responsesGoldenDir, name)
			var c responsesGoldenCase
			raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &c); err != nil {
				t.Fatal(err)
			}
			switch {
			case c.Admission == "no_lever" && c.Refusal == nil:
				assertResponsesNoLeverThroughTheRouter(t, dir, c)
			case c.Refusal != nil:
				assertResponsesRefusalAtLoad(t, c)
			default:
				t.Fatalf("case %s is admitted with a lever; add its router-path oracle", name)
			}
		})
	}
}

// assertResponsesNoLeverThroughTheRouter routes each client body to a model
// alias that serves the case's wire model on Responses, so the route changes
// the model, and requires the case's provider bytes.
func assertResponsesNoLeverThroughTheRouter(t *testing.T, dir string, c responsesGoldenCase) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	config := strings.NewReplacer("{{WIRE_MODEL}}", c.WireModel).Replace(responsesNoLeverConfigTemplate)
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	for index, call := range c.Calls {
		client, err := os.ReadFile(filepath.Join(dir, call.ClientBody))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join(dir, call.ExpectedBody))
		if err != nil {
			t.Fatal(err)
		}
		recorded := `"model":"` + c.WireModel + `"`
		if strings.Count(string(client), recorded) != 1 {
			t.Fatalf("call %d no longer names %s", index, recorded)
		}
		routed := strings.Replace(string(client), recorded, `"model":"codex-arm"`, 1)
		if got := dispatchResponsesClientBody(t, router, fmt.Sprintf("golden-noop-%d", index), routed); got != string(want) {
			t.Fatalf("call %d provider body =\n%s\nwant\n%s", index, got, want)
		}
	}
}

// assertResponsesRefusalAtLoad binds every control the case's calls name to
// a worker on the case's (model, provider, responses) cell and requires the
// router to refuse the configuration with the recorded message.
func assertResponsesRefusalAtLoad(t *testing.T, c responsesGoldenCase) {
	t.Helper()
	controls := registryControls(t)
	var manifest map[string]any
	fixture, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(fixture, &manifest); err != nil {
		t.Fatal(err)
	}
	var actions []any
	var bindings strings.Builder
	seen := map[string]bool{}
	for _, call := range c.Calls {
		if call.ControlID == nil || seen[*call.ControlID] {
			continue
		}
		seen[*call.ControlID] = true
		control, ok := controls[*call.ControlID]
		if !ok {
			t.Fatalf("control %s is not in the registry", *call.ControlID)
		}
		var level any
		if instruction, ok := control.(map[string]any)["instruction"].(map[string]any); ok {
			level = instruction["level"]
		}
		actionID := fmt.Sprintf("%064x", len(actions)+1)
		actions = append(actions, map[string]any{
			"action_id": actionID, "model": "trained-arm", "control": control,
			"control_id": *call.ControlID, "level": level, "trained_arm_ids": []any{},
		})
		fmt.Fprintf(&bindings, "              - action_id: %q\n                worker: arm\n", actionID)
	}
	if len(actions) == 0 {
		t.Fatal("the refused case binds no control")
	}
	// ARC needs a second worker, and every worker must serve an action: the
	// spare serves the native-default control, which every Responses cell
	// admits, bound last so the case's own refusal is the one reported.
	spareID := fmt.Sprintf("%064x", len(actions)+1)
	actions = append(actions, map[string]any{
		"action_id": spareID, "model": "trained-spare", "control": controls[responsesNativeDefaultControl],
		"control_id": responsesNativeDefaultControl, "level": nil, "trained_arm_ids": []any{},
	})
	fmt.Fprintf(&bindings, "              - action_id: %q\n                worker: spare\n", spareID)
	manifest["actions"] = actions
	decision := manifest["decision"].(map[string]any)
	first := actions[0].(map[string]any)["action_id"]
	decision["fallback_action_id"] = first
	decision["reference_action_ids"] = map[string]any{"arm": first}
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "package.json")
	if err := os.WriteFile(manifestPath, body, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	baseURL := map[string]string{"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}[c.Provider]
	if baseURL == "" {
		t.Fatalf("no backend for provider %q", c.Provider)
	}
	config := strings.NewReplacer(
		"{{MODEL}}", c.Model, "{{PROVIDER}}", c.Provider, "{{BASE_URL}}", baseURL,
		"{{PACKAGE}}", hex.EncodeToString(sum[:]), "{{MANIFEST}}", manifestPath, "{{BINDINGS}}", bindings.String(),
	).Replace(responsesRefusalConfigTemplate)
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewOpenAIRouter(path); err == nil || !strings.Contains(err.Error(), c.Refusal.Error) {
		t.Fatalf("load error = %v, want the recorded refusal %q", err, c.Refusal.Error)
	}
}

// registryControls is the compiled registry's control objects by id, as the
// manifest states them.
func registryControls(t *testing.T) map[string]any {
	t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/thinkingcontrol/thinking_controls.compiled.json")
	if err != nil {
		t.Fatal(err)
	}
	var registry struct {
		Controls map[string]any `json:"controls"`
	}
	if err := json.Unmarshal(raw, &registry); err != nil {
		t.Fatal(err)
	}
	return registry.Controls
}

// dispatchResponsesClientBody sends a Responses client body through the
// request phases and returns the provider-bound bytes.
func dispatchResponsesClientBody(t *testing.T, router *OpenAIRouter, requestID, body string) string {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: requestID, StartTime: time.Now(), TraceContext: context.Background()}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/responses"},
			{Key: "content-type", Value: "application/json"},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	return string(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody())
}

const responsesNoLeverConfigTemplate = `version: v0.3

providers:
  defaults:
    model: codex-arm
  models:
    - name: codex-arm
      provider_model_id: {{WIRE_MODEL}}
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openai-codex
          base_url: https://api.openai.com/v1
          provider: openai
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: codex-arm
      modality: text

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

const responsesRefusalConfigTemplate = `version: v0.3

providers:
  defaults:
    model: arm
  models:
    - name: arm
      provider_model_id: {{MODEL}}
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: arm-backend
          base_url: {{BASE_URL}}
          provider: {{PROVIDER}}
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: spare
      provider_model_id: {{MODEL}}
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: spare-backend
          base_url: {{BASE_URL}}
          provider: {{PROVIDER}}
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: arm
      modality: text
    - name: spare
      modality: text
  decisions:
    - name: rayline-arc-responses-golden
      description: Policy-service ARC route binding a golden case's controls
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: arm
          use_reasoning: true
        - model: spare
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
            base_url: http://127.0.0.1:1
            total_timeout_seconds: 5
            package_alias: rayline/responses-golden
            package_sha256: {{PACKAGE}}
            package_manifest: {{MANIFEST}}
            allow_experimental_controls: true
            model_schedule: task_turn_compaction_v1
            trained_models:
              arm: trained-arm
              spare: trained-spare
            bindings:
{{BINDINGS}}          episode:
            id_header: x-rayline-session
            backend: memory
            key_prefix: "vsr:rayline-arc-responses-golden:"
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
