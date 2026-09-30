//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

// The policy-service mode end to end, in process: the router is built from a
// config file, arms against a fake policy service, and runs a client's
// Anthropic Messages request through the real ext_proc request path. What is
// asserted is the provider-bound body -- the bytes that would leave for the
// provider -- for each kind of action the package can choose.
func TestPolicyActionReachesTheProviderBody(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	budget := int64(4096)
	actions := map[string]config.RaylineARCPolicyBinding{
		// Actions name trained models; the cards serve provider model ids.
		"think-up":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"think-budget":  policyAction("think", "none", "think-trained", nil, &budget, ""),
		"claude":        policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"off":           policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"off-up":        policyAction("off", "up", "off-trained", policyTestEffort("none"), nil, policyTestUp),
		"claude-off-up": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	chat := func(t *testing.T, action string) map[string]json.RawMessage {
		t.Helper()
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[action].ActionID })
		return dispatchPolicyRequest(t, router, "episode-"+action)
	}

	t.Run("a steered effort action on Chat", func(t *testing.T) {
		body := chat(t, "think-up")
		assertJSONField(t, body, "model", `"vendor/think"`)
		assertJSONField(t, body, "reasoning", `{"effort":"high"}`)
		if _, present := body["reasoning_effort"]; present {
			t.Fatalf("reasoning_effort travelled beside the action's reasoning: %s", body["reasoning_effort"])
		}
		messages := chatMessages(t, body)
		if last := string(messages[len(messages)-1]); !strings.Contains(last, policyTestUp) {
			t.Fatalf("the tail user message does not carry the steer: %s", last)
		}
	})
	t.Run("a budget action on Chat", func(t *testing.T) {
		body := chat(t, "think-budget")
		assertJSONField(t, body, "reasoning", `{"max_tokens":4096}`)
		messages := chatMessages(t, body)
		if last := string(messages[len(messages)-1]); strings.Contains(last, "steering instruction") {
			t.Fatalf("a neutral action carried a steer: %s", last)
		}
	})
	t.Run("a thinking-off action on Chat", func(t *testing.T) {
		body := chat(t, "off")
		// The off signal is the one the router derives for a
		// use_reasoning:false worker; this card declares no reasoning family,
		// so, as in prod, no reasoning control travels.
		assertJSONField(t, body, "model", `"vendor/off"`)
		if _, present := body["reasoning"]; present {
			t.Fatalf("a thinking-off action added reasoning controls: %s", body["reasoning"])
		}
	})
	t.Run("a steered thinking-off action on Chat", func(t *testing.T) {
		body := chat(t, "off-up")
		assertJSONField(t, body, "model", `"vendor/off"`)
		if _, present := body["reasoning"]; present {
			t.Fatalf("a thinking-off action added reasoning controls: %s", body["reasoning"])
		}
		messages := chatMessages(t, body)
		if last := string(messages[len(messages)-1]); !strings.Contains(last, policyTestUp) {
			t.Fatalf("the tail user message does not carry the steer: %s", last)
		}
	})
	t.Run("a steered thinking-off action on Messages", func(t *testing.T) {
		body := chat(t, "claude-off-up")
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		assertJSONField(t, body, "thinking", `{"type":"disabled"}`)
		if _, present := body["output_config"]; present {
			t.Fatalf("a thinking-off action carried an effort: %s", body["output_config"])
		}
		if !strings.Contains(string(body["messages"]), policyTestUp) {
			t.Fatalf("the messages do not carry the steer: %s", body["messages"])
		}
	})
	t.Run("an effort action on Messages", func(t *testing.T) {
		body := chat(t, "claude")
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		assertJSONField(t, body, "output_config", `{"effort":"medium"}`)
		var thinking struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(body["thinking"], &thinking); err != nil || thinking.Type != "adaptive" {
			t.Fatalf("thinking = %s, want adaptive", body["thinking"])
		}
	})
}

func awaitPolicySelectorArmed(t *testing.T, router *OpenAIRouter) {
	t.Helper()
	registered, ok := router.ModelSelector.Get(selection.MethodRaylineARC)
	if !ok {
		t.Fatal("the ARC selector is not registered")
	}
	selector, ok := registered.(*raylineARCSelector)
	if !ok {
		t.Fatalf("the ARC selector is %T", registered)
	}
	deadline := time.Now().Add(15 * time.Second)
	for selector.armedComponents() == nil {
		if time.Now().After(deadline) {
			t.Fatal("the policy selector never armed against the fake service")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// dispatchPolicyRequest runs one client Messages request through the request
// phases and returns the provider-bound body.
func dispatchPolicyRequest(t *testing.T, router *OpenAIRouter, episode string) map[string]json.RawMessage {
	t.Helper()
	ctx := &RequestContext{
		Headers:      map[string]string{},
		RequestID:    "policy-e2e-" + episode,
		StartTime:    time.Now(),
		TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	client := `{"model":"auto","max_tokens":32000,"messages":[` +
		`{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":"Looking at it."},` +
		`{"role":"user","content":"go on"}]}`
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
	var body map[string]json.RawMessage
	if err := json.Unmarshal(mutation.GetBody(), &body); err != nil {
		t.Fatalf("provider body: %v\n%s", err, mutation.GetBody())
	}
	return body
}

func assertJSONField(t *testing.T, body map[string]json.RawMessage, field string, want string) {
	t.Helper()
	var got, expected any
	if err := json.Unmarshal(body[field], &got); err != nil {
		t.Fatalf("%s is %q: %v", field, body[field], err)
	}
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Fatal(err)
	}
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(expected)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("%s = %s, want %s", field, body[field], want)
	}
}

func chatMessages(t *testing.T, body map[string]json.RawMessage) []json.RawMessage {
	t.Helper()
	var messages []json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil || len(messages) == 0 {
		t.Fatalf("messages = %s", body["messages"])
	}
	return messages
}

func writePolicyDispatchConfig(t *testing.T, policyURL string, actions map[string]config.RaylineARCPolicyBinding) string {
	t.Helper()
	upBinding := thinkinglever.Binding{Lever: thinkinglever.LeverSteeringSuffix}
	var bindings strings.Builder
	for _, action := range actions {
		fmt.Fprintf(&bindings, "              - action_id: %s\n                worker: %q\n                level: %s\n                model: %s\n",
			action.ActionID, action.Worker, action.Level, action.Model)
		if action.Effort != nil {
			fmt.Fprintf(&bindings, "                effort: %s\n", *action.Effort)
		}
		if action.ReasoningMaxTokens != nil {
			fmt.Fprintf(&bindings, "                reasoning_max_tokens: %d\n", *action.ReasoningMaxTokens)
		}
	}
	rendered := strings.NewReplacer(
		"{{POLICY_URL}}", policyURL,
		"{{ALIAS}}", policyTestAlias,
		"{{PACKAGE}}", policyTestPackage,
		"{{BINDINGS}}", strings.TrimRight(bindings.String(), "\n"),
		"{{UP_SUFFIX}}", policyTestUp,
		"{{NONE_SHA}}", upBinding.ControlSHA256(thinkinglever.Level{}),
		"{{UP_SHA}}", upBinding.ControlSHA256(thinkinglever.Level{Suffix: policyTestUp}),
	).Replace(policyDispatchConfigTemplate)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const policyDispatchConfigTemplate = `version: v0.3

providers:
  defaults:
    model: think
  models:
    - name: think
      provider_model_id: vendor/think
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openrouter-think
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: "off"
      provider_model_id: vendor/off
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openrouter-off
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: claude
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: anthropic-claude
          base_url: https://api.anthropic.com
          provider: anthropic
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: claude-off
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: anthropic-claude-off
          base_url: https://api.anthropic.com
          provider: anthropic
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: think
      modality: text
    - name: "off"
      modality: text
    - name: claude
      modality: text
    - name: claude-off
      modality: text
  decisions:
    - name: rayline-arc-policy-e2e
      description: Policy-service ARC route for the dispatch e2e
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: think
          use_reasoning: true
        - model: "off"
          use_reasoning: false
        - model: claude
          use_reasoning: true
        - model: claude-off
          use_reasoning: false
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
            bindings:
{{BINDINGS}}
          thinking_lever:
            enabled: true
            source: policy
            workers:
              think:
                admission: certified
                lever: prompt_steering_suffix
                emit: on_change
                neutral_level: none
                placements:
                  - append_tail_user_text
                  - insert_user_after_tool_run
                levels:
                  - level: none
                    rank: 0
                    control_sha256: {{NONE_SHA}}
                  - level: up
                    rank: 1
                    suffix: "{{UP_SUFFIX}}"
                    control_sha256: {{UP_SHA}}
              "off":
                admission: certified
                lever: prompt_steering_suffix
                emit: on_change
                neutral_level: none
                placements:
                  - append_tail_user_text
                levels:
                  - level: none
                    rank: 0
                    control_sha256: {{NONE_SHA}}
                  - level: up
                    rank: 1
                    suffix: "{{UP_SUFFIX}}"
                    control_sha256: {{UP_SHA}}
              claude-off:
                admission: certified
                lever: prompt_steering_suffix
                emit: on_change
                neutral_level: none
                placements:
                  - append_tail_user_text
                levels:
                  - level: none
                    rank: 0
                    control_sha256: {{NONE_SHA}}
                  - level: up
                    rank: 1
                    suffix: "{{UP_SUFFIX}}"
                    control_sha256: {{UP_SHA}}
          episode:
            id_header: x-rayline-session
            backend: memory
            key_prefix: "vsr:rayline-arc-policy-e2e:"
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

// A client's Responses request is decided as openai_responses -- input items
// and instructions -- and its chosen action reaches the provider body.
func TestPolicyDecidesAResponsesRequest(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage,
		[]string{actions["think"].ActionID, actions["off"].ActionID, actions["claude"].ActionID, actions["claude-off"].ActionID})
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["think"].ActionID })

	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "policy-e2e-responses", StartTime: time.Now(), TraceContext: context.Background()}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/responses"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: "episode-responses"},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	client := `{"model":"auto","max_output_tokens":32000,"instructions":"You are Codex.","input":[` +
		`{"type":"message","role":"user","content":"List the files."},` +
		`{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"README.md"}]}`
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	sent := fake.received()
	if len(sent) != 1 || sent[0].RequestFormat != policyFormatResponses || len(sent[0].Request.Input) != 3 ||
		sent[0].Request.Messages != nil || sent[0].Request.Instructions == nil || *sent[0].Request.Instructions != "You are Codex." {
		t.Fatalf("decide requests = %+v", sent)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatal(err)
	}
	assertJSONField(t, body, "model", `"vendor/think"`)
	assertJSONField(t, body, "reasoning", `{"effort":"high"}`)
	if !strings.Contains(string(body["messages"]), policyTestUp) {
		t.Fatalf("the provider body does not carry the steer: %s", body["messages"])
	}
}
