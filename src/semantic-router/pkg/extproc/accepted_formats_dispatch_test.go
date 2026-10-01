//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const acceptedFormatsDispatchConfig = `
version: v0.3
providers:
  defaults:
    model: claude
  models:
    - name: claude
      provider_model_id: anthropic/claude-opus-5
      accepted_formats: [anthropic, openai]
      backend_refs:
        - name: openrouter-claude
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: ACCEPTED_FORMATS_DISPATCH_KEY
    - name: single
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
      backend_refs:
        - name: openrouter-single
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: ACCEPTED_FORMATS_DISPATCH_KEY
routing:
  modelCards:
    - name: claude
    - name: single
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: claude
`

// A model that accepts several formats takes each request in the client's own
// format when it is listed, and in its first format otherwise; a model with a
// single api_format takes every request in it, as before.
func TestAcceptedFormatsDispatchPerRequest(t *testing.T) {
	t.Setenv("ACCEPTED_FORMATS_DISPATCH_KEY", "public-accepted-formats-key")
	cfg, err := config.ParseYAMLBytes([]byte(acceptedFormatsDispatchConfig))
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
	for name, test := range map[string]struct {
		model    string
		client   llmprotocol.WireFormat
		wantPath string
		wantWire llmprotocol.WireFormat
	}{
		"a Chat client keeps Chat":                      {"claude", llmprotocol.OpenAIChatV1, "/api/v1/chat/completions", llmprotocol.OpenAIChatV1},
		"a Messages client keeps Messages":              {"claude", llmprotocol.AnthropicMessagesV1, "/api/v1/messages", llmprotocol.AnthropicMessagesV1},
		"a Responses client takes the first format":     {"claude", llmprotocol.OpenAIResponsesV1, "/api/v1/messages", llmprotocol.AnthropicMessagesV1},
		"a single api_format serves every client":       {"single", llmprotocol.OpenAIChatV1, "/api/v1/messages", llmprotocol.AnthropicMessagesV1},
		"a single api_format serves its own client too": {"single", llmprotocol.AnthropicMessagesV1, "/api/v1/messages", llmprotocol.AnthropicMessagesV1},
	} {
		t.Run(name, func(t *testing.T) {
			request := testNeutralRequest("auto", "fix the failing test")
			ctx := routingTestContext(test.client, request)
			ctx.VSRSelectedDecision = &config.Decision{Name: "default"}
			response, err := router.handleEntrypointModelRouting(
				request, "auto", "default", entropy.ReasoningDecision{}, test.model, ctx,
			)
			if err != nil {
				t.Fatalf("routing: %v", err)
			}
			if ctx.TargetFormat != test.wantWire {
				t.Fatalf("target format = %q, want %q", ctx.TargetFormat, test.wantWire)
			}
			common := response.GetRequestBody().GetResponse()
			if got := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]; got != test.wantPath {
				t.Fatalf(":path = %q, want %q", got, test.wantPath)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
				t.Fatalf("provider body: %v", err)
			}
			_, messagesBody := body["max_tokens"]
			if messagesBody != (test.wantWire == llmprotocol.AnthropicMessagesV1) {
				t.Fatalf("the body is not a %s body: %s", test.wantWire, common.GetBodyMutation().GetBody())
			}
		})
	}
}

// An ARC policy action reaches a worker that accepts Chat and Messages in the
// format each client sent: the same action travels as OpenRouter's reasoning
// object for a Chat client and as output_config.effort for a Messages client.
func TestPolicyActionFollowsThePerRequestFormat(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	action := policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, "")
	// Every modelRef serves an action; only the claude action is chosen.
	actions := map[string]config.RaylineARCPolicyBinding{
		"claude":        action,
		"think-up":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":           policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude-off-up": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, declared := range actions {
		catalog = append(catalog, declared.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rendered := string(raw)
	for from, to := range map[string]string{
		"    - name: claude\n      provider_model_id: anthropic/claude-opus-5\n      api_format: anthropic\n":              "    - name: claude\n      provider_model_id: anthropic/claude-opus-5\n      accepted_formats: [anthropic, openai]\n",
		"        - name: anthropic-claude\n          base_url: https://api.anthropic.com\n          provider: anthropic\n": "        - name: openrouter-claude\n          base_url: https://openrouter.ai/api/v1\n          provider: openrouter\n",
	} {
		if strings.Count(rendered, from) != 1 {
			t.Fatalf("the config template changed; update the accepted_formats rewrite:\n%s", from)
		}
		rendered = strings.Replace(rendered, from, to, 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action.ActionID })

	chat := dispatchPolicyClientRequest(t, router, "episode-accepted-chat", "/v1/chat/completions",
		`{"model":"auto","messages":[{"role":"user","content":"fix the failing test"}]}`)
	assertJSONField(t, chat, "model", `"anthropic/claude-opus-5"`)
	assertJSONField(t, chat, "reasoning", `{"effort":"medium"}`)
	if _, present := chat["output_config"]; present {
		t.Fatalf("a Chat turn carried output_config: %s", chat["output_config"])
	}

	messages := dispatchPolicyRequest(t, router, "episode-accepted-messages")
	assertJSONField(t, messages, "model", `"anthropic/claude-opus-5"`)
	assertJSONField(t, messages, "output_config", `{"effort":"medium"}`)
	assertJSONField(t, messages, "thinking", `{"type":"adaptive"}`)
	if _, present := messages["reasoning"]; present {
		t.Fatalf("a Messages turn carried the Chat reasoning object: %s", messages["reasoning"])
	}
}
