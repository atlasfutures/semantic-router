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
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Pathfinder's Responses golden corpus respects admission: every case is
// no_lever, whose provider bytes are the client's apart from the model;
// refused with the registry's message; or admitted with a lever. Through the
// router the first and the last must come out byte for byte, call by call,
// and the second must be refused where VSR refuses it, at load, with the
// recorded message. The placement corpus is the renderer's oracle only and
// is not read here.
const responsesGoldenDir = "../selection/raylinearc/thinkingcontrol/testdata/golden/responses"

// responsesNativeDefaultControl is the registry's native-default control: no
// budget, no instruction.
const responsesNativeDefaultControl = "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0"

type responsesGoldenCase struct {
	Format    string `json:"format"`
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
				assertResponsesLeverThroughTheRouter(t, dir, c)
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
		// The client names its own model, which need not be the wire model
		// (noop_fidelity sends gpt-6-sol); the route replaces it.
		named := clientModelMember.FindAllString(string(client), -1)
		if len(named) != 1 {
			t.Fatalf("call %d names %d models, want 1", index, len(named))
		}
		routed := strings.Replace(string(client), named[0], `"model":"codex-arm"`, 1)
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
	path, _, _ := responsesGoldenPolicyConfig(t, c, "http://127.0.0.1:1")
	if _, err := NewOpenAIRouter(path); err == nil || !strings.Contains(err.Error(), c.Refusal.Error) {
		t.Fatalf("load error = %v, want the recorded refusal %q", err, c.Refusal.Error)
	}
}

// assertResponsesLeverThroughTheRouter serves the case's controls from a
// policy-service package on the case's (model, provider, responses) cell, has
// the policy choose each call's control, and requires each call's provider
// bytes. The calls are one episode, each committed as a 200, so the ledger
// carries from one to the next as the corpus records it.
func assertResponsesLeverThroughTheRouter(t *testing.T, dir string, c responsesGoldenCase) {
	t.Helper()
	path, packageSHA, actionFor := responsesGoldenPolicyConfig(t, c, "{{POLICY_URL}}")
	var catalog []string
	for _, action := range actionFor {
		catalog = append(catalog, action)
	}
	sort.Strings(catalog)
	fake := newFakePolicyService(t, responsesGoldenAlias, packageSHA, catalog)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, []byte(strings.ReplaceAll(string(raw), "{{POLICY_URL}}", fake.URL())), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	episode := "golden-" + filepath.Base(dir)
	for index, call := range c.Calls {
		if call.ControlID == nil {
			t.Fatalf("call %d names no control", index)
		}
		client, err := os.ReadFile(filepath.Join(dir, call.ClientBody))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join(dir, call.ExpectedBody))
		if err != nil {
			t.Fatal(err)
		}
		// The client names its own model (codex sends gpt-6-sol); the policy
		// route chooses the worker, so the client asks for auto.
		named := clientModelMember.FindAllString(string(client), -1)
		if len(named) != 1 {
			t.Fatalf("call %d names %d models, want 1", index, len(named))
		}
		routed := strings.Replace(string(client), named[0], `"model":"auto"`, 1)
		action := actionFor[*call.ControlID]
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
		got := dispatchLeverTurn(t, router, goldenClientPath(t, c.Format), fmt.Sprintf("%s-%d", episode, index), episode, routed)
		if got != string(want) {
			t.Fatalf("call %d provider body =\n%s\nwant\n%s", index, got, want)
		}
	}
}

// codecOwnedGoldenCases are the corpus cases whose router-path bytes are the
// Messages codec's, not placement's, each for its stated reason; the fold on
// them is pinned by TestGoldenParity on the renderer.
var codecOwnedGoldenCases = map[string]string{
	"inband_system_fold_developer": "Anthropic Messages has no developer role, so the router's Messages codec " +
		"re-encodes a body carrying one (member order, and a tool_result's string content becomes a block list); " +
		"a Messages client never sends this role",
}

const responsesGoldenAlias = "rayline/responses-golden"

// goldenAPIFormat is the model card api_format that serves a corpus format.
func goldenAPIFormat(t *testing.T, format string) string {
	t.Helper()
	switch format {
	case "responses":
		return "responses"
	case "messages":
		return "anthropic"
	case "chat":
		return "openai"
	}
	t.Fatalf("no api_format for corpus format %q", format)
	return ""
}

// goldenClientPath is the client endpoint a corpus format's bodies are sent to.
func goldenClientPath(t *testing.T, format string) string {
	t.Helper()
	switch format {
	case "responses":
		return "/v1/responses"
	case "messages":
		return "/v1/messages"
	}
	t.Fatalf("no client path for corpus format %q", format)
	return ""
}

var clientModelMember = regexp.MustCompile(`"model":"[^"]*"`)

// responsesSpareControl is a native-only control the case's cell admits, for
// the spare worker every package needs: the registry's native default where
// the cell admits it, else the cell's own native-only control (a cell whose
// only base is high admits no default).
func responsesSpareControl(t *testing.T, c responsesGoldenCase, controls map[string]any) string {
	t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/thinkingcontrol/thinking_controls.compiled.json")
	if err != nil {
		t.Fatal(err)
	}
	var registry struct {
		Cells []struct {
			Model    string   `json:"model"`
			Provider string   `json:"provider"`
			Format   string   `json:"format"`
			Controls []string `json:"controls"`
		} `json:"cells"`
	}
	if err := json.Unmarshal(raw, &registry); err != nil {
		t.Fatal(err)
	}
	for _, cell := range registry.Cells {
		if cell.Model != c.Model || cell.Provider != c.Provider || cell.Format != c.Format {
			continue
		}
		for _, id := range cell.Controls {
			if id == responsesNativeDefaultControl {
				return id
			}
		}
		for _, id := range cell.Controls {
			if control, ok := controls[id].(map[string]any); ok && control["instruction"] == nil {
				return id
			}
		}
	}
	return responsesNativeDefaultControl
}

// responsesGoldenPolicyConfig writes a policy-service package that binds
// every control the case's calls name to a worker on the case's (model,
// provider, responses) cell, and a config serving it from policyURL. It
// returns the config path, the package's sha256 and each control's action.
func responsesGoldenPolicyConfig(t *testing.T, c responsesGoldenCase, policyURL string) (string, string, map[string]string) {
	t.Helper()
	controls := registryControls(t)
	var manifest map[string]any
	fixture, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fixture, &manifest); err != nil {
		t.Fatal(err)
	}
	var actions []any
	var bindings strings.Builder
	actionFor := map[string]string{}
	for _, call := range c.Calls {
		if call.ControlID == nil || actionFor[*call.ControlID] != "" {
			continue
		}
		control, ok := controls[*call.ControlID]
		if !ok {
			t.Fatalf("control %s is not in the registry", *call.ControlID)
		}
		var level any
		if instruction, ok := control.(map[string]any)["instruction"].(map[string]any); ok {
			level = instruction["level"]
		}
		actionID := fmt.Sprintf("%064x", len(actions)+1)
		actionFor[*call.ControlID] = actionID
		actions = append(actions, map[string]any{
			"action_id": actionID, "model": "trained-arm", "control": control,
			"control_id": *call.ControlID, "level": level, "trained_arm_ids": []any{},
		})
		fmt.Fprintf(&bindings, "              - action_id: %q\n                worker: arm\n", actionID)
	}
	if len(actions) == 0 {
		t.Fatal("the case binds no control")
	}
	// ARC needs a second worker, and every worker must serve an action: the
	// spare serves a native-only control its cell admits, bound last so the
	// case's own refusal is the one reported.
	spareID := fmt.Sprintf("%064x", len(actions)+1)
	spare := responsesSpareControl(t, c, controls)
	actions = append(actions, map[string]any{
		"action_id": spareID, "model": "trained-spare", "control": controls[spare],
		"control_id": spare, "level": nil, "trained_arm_ids": []any{},
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
	packageSHA := hex.EncodeToString(sum[:])
	config := strings.NewReplacer(
		"{{MODEL}}", c.Model, "{{PROVIDER}}", c.Provider, "{{BASE_URL}}", baseURL, "{{POLICY_URL}}", policyURL,
		"{{API_FORMAT}}", goldenAPIFormat(t, c.Format),
		"{{PACKAGE}}", packageSHA, "{{MANIFEST}}", manifestPath, "{{BINDINGS}}", bindings.String(),
	).Replace(responsesPolicyConfigTemplate)
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, packageSHA, actionFor
}

// dispatchLeverTurn sends one client body of an episode
// through the request phases, commits it as a 200, and returns the
// provider-bound bytes.
func dispatchLeverTurn(t *testing.T, router *OpenAIRouter, path, requestID, episode, body string) string {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: requestID, StartTime: time.Now(), TraceContext: context.Background()}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: path},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
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
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("no provider body was written")
	}
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
	return string(mutation.GetBody())
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
			{Key: ":method", Value: "POST"}, {Key: ":path", Value: "/v1/responses"},
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

const responsesPolicyConfigTemplate = `version: v0.3

providers:
  defaults:
    model: arm
  models:
    - name: arm
      provider_model_id: {{MODEL}}
      api_format: {{API_FORMAT}}
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
      api_format: {{API_FORMAT}}
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
      capabilities: [tool_result_images]
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
            base_url: {{POLICY_URL}}
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

const messagesGoldenDir = "../selection/raylinearc/thinkingcontrol/testdata/golden/messages"

// The Messages corpus's in-band system fold cases (pathfinder FOLD_RULES,
// fold_before_unit_v1), through the router: every admitted case must reach
// the provider as the corpus's bytes, call by call in one committed episode,
// and every refused case must be refused at load with its recorded message.
func TestMessagesFoldGoldenCorpusThroughTheRouter(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	entries, err := os.ReadDir(messagesGoldenDir)
	if err != nil {
		t.Fatal(err)
	}
	ran := 0
	for _, entry := range entries {
		if !entry.IsDir() || !strings.Contains(entry.Name(), "fold") {
			continue
		}
		name := entry.Name()
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(messagesGoldenDir, name)
			raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
			if err != nil {
				t.Fatal(err)
			}
			var c responsesGoldenCase
			if err := json.Unmarshal(raw, &c); err != nil {
				t.Fatal(err)
			}
			if reason, owned := codecOwnedGoldenCases[name]; owned {
				t.Skip(reason)
			}
			if c.Refusal != nil {
				// A fold refusal is placement's, at the call that breaks the
				// rule, not the package's at load; TestGoldenParity pins its
				// message on the renderer.
				t.Skipf("placement refusal %q is pinned by TestGoldenParity", c.Refusal.Error)
			}
			assertResponsesLeverThroughTheRouter(t, dir, c)
		})
		ran++
	}
	if ran == 0 {
		t.Fatal("the Messages corpus has no fold cases")
	}
}
