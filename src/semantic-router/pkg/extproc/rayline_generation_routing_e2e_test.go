package extproc

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// writeGenerationRoutingConfig is the policy dispatch cell as configurable
// ARC: its policy_service reads generation routing with the caller's V.
func writeGenerationRoutingConfig(t *testing.T, policyURL string, actions map[string]config.RaylineARCPolicyBinding) string {
	t.Helper()
	path := writePolicyDispatchConfig(t, policyURL, actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const anchor = "            package_alias: "
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the policy dispatch config has no package_alias to anchor on")
	}
	rendered := strings.Replace(string(raw), anchor,
		"            generation_routing:\n              caller_value: true\n"+anchor, 1)
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil { //nolint:gosec // a t.TempDir() path
		t.Fatal(err)
	}
	return path
}

// generationTurn runs one client request through the request phases and
// returns the immediate response, if any, and the provider-bound body.
func generationTurn(t *testing.T, router *OpenAIRouter, episode, path, client string) (*ext_proc.ImmediateResponse, []byte) {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "generation-e2e-" + episode, StartTime: time.Now(),
		TraceContext: context.Background(),
	}
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
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if response.GetImmediateResponse() != nil {
		// As the stream processor does: a refusal is written in the client's
		// own protocol.
		return router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse(), nil
	}
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("no provider body was written")
	}
	return nil, mutation.GetBody()
}

// A configurable-ARC generation turn carrying routing.{value_usd,
// temperature}, on each of the three protocols, runs through ExtProc to the
// provider: the routing block reaches neither the provider body nor the
// policy service's copy of the client request, and the turn is decided at the
// caller's V. A refused block never reaches the provider at all.
func TestGenerationRoutingNeverReachesTheProvider(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage,
		[]string{actions["think"].ActionID, actions["off"].ActionID, actions["claude"].ActionID, actions["claude-off"].ActionID})
	router, err := NewOpenAIRouter(writeGenerationRoutingConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	const routing = `"routing":{"value_usd":2,"temperature":0.5}`
	for _, tc := range []struct {
		name, path, action, client string
	}{
		{"messages", "/v1/messages", "claude", `{"model":"auto","max_tokens":1024,` + routing +
			`,"messages":[{"role":"user","content":"fix the failing test"}]}`},
		{"chat completions", "/v1/chat/completions", "think", `{"model":"auto","max_completion_tokens":1024,` + routing +
			`,"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"fix the failing test"}]}`},
		{"responses", "/v1/responses", "think", `{"model":"auto","max_output_tokens":1024,` + routing +
			`,"input":[{"type":"message","role":"user","content":"fix the failing test"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(fake.receivedBodies())
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[tc.action].ActionID })
			immediate, provider := generationTurn(t, router, "generation-"+strings.ReplaceAll(tc.name, " ", "-"), tc.path, tc.client)
			if immediate != nil {
				t.Fatalf("refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(provider, &body); err != nil {
				t.Fatalf("provider body: %v\n%s", err, provider)
			}
			if _, present := body["routing"]; present || strings.Contains(string(provider), "value_usd") {
				t.Fatalf("routing reached the provider: %s", provider)
			}
			if _, present := body["model"]; !present {
				t.Fatalf("not a provider body: %s", provider)
			}
			sent := fake.receivedBodies()
			if len(sent) != before+1 {
				t.Fatalf("decide calls %d, want one more than %d", len(sent), before)
			}
			decide := decodeJSONObject(t, sent[len(sent)-1])
			if strings.Contains(string(sent[len(sent)-1]), "value_usd") {
				t.Fatalf("routing reached the policy service's client request: %s", sent[len(sent)-1])
			}
			point := decide["selection"].(map[string]any)["operating_point"].(map[string]any)
			if point["cost_weight"] != 0.5 {
				t.Fatalf("operating_point %v, want the caller's V 2 as cost_weight 0.5", point)
			}
		})
	}

	for _, refused := range []struct{ path, client string }{
		{"/v1/messages", `{"model":"auto","max_tokens":1024,"routing":{"value_usd":9},"messages":[{"role":"user","content":"x"}]}`},
		{"/v1/chat/completions", `{"model":"auto","routing":{"value_usd":9},"messages":[{"role":"user","content":"x"}]}`},
		{"/v1/responses", `{"model":"auto","routing":{"value_usd":9},"input":[{"type":"message","role":"user","content":"x"}]}`},
	} {
		t.Run("a refused block reaches no one on "+refused.path, func(t *testing.T) {
			before := len(fake.receivedBodies())
			immediate, provider := generationTurn(t, router, "generation-refused", refused.path, refused.client)
			if immediate == nil || immediate.GetStatus().GetCode() != 400 || provider != nil {
				t.Fatalf("immediate %v provider %s", immediate, provider)
			}
			var envelope struct {
				Type  string `json:"type"`
				Error struct {
					Type    string `json:"type"`
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(immediate.GetBody(), &envelope); err != nil ||
				envelope.Error.Type != "invalid_request_error" || envelope.Error.Code != "value_out_of_bounds" ||
				!strings.Contains(envelope.Error.Message, "value_usd must lie in") {
				t.Fatalf("refusal body %s, want the contract code in the client's envelope", immediate.GetBody())
			}
			if refused.path == "/v1/messages" && envelope.Type != "error" {
				t.Fatalf("a Messages refusal is not the Anthropic envelope: %s", immediate.GetBody())
			}
			if len(fake.receivedBodies()) != before {
				t.Fatal("a refused turn was decided")
			}
		})
	}
}
