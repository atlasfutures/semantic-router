//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const responsesInputIDsConfig = `
version: v0.3
providers:
  defaults:
    model: gpt
  models:
    - name: gpt
      provider_model_id: gpt-5.6-sol
      api_format: responses
      backend_refs:
        - name: openai-gpt
          base_url: https://api.openai.com/v1
          provider: openai
          api_key_env: RESPONSES_INPUT_IDS_KEY
routing:
  modelCards:
    - name: gpt
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: gpt
`

// A multi-message Chat or Messages conversation dispatched to a Responses
// backend reaches it with no input item id the client did not send. The
// Router used to invent one per item, and every id-less message's first item
// got the same one; a Messages tool result also carried an invented id.
func TestTranslatedRequestReachesResponsesWithoutInventedItemIDs(t *testing.T) {
	t.Setenv("RESPONSES_INPUT_IDS_KEY", "public-responses-input-ids-key")
	cfg, err := config.ParseYAMLBytes([]byte(responsesInputIDsConfig))
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
	engine := protocolcodec.NewBuiltinEngine()
	for name, test := range map[string]struct {
		client llmprotocol.WireFormat
		body   string
	}{
		"Chat": {llmprotocol.OpenAIChatV1, `{"model":"auto","messages":[{"role":"user","content":"one"},` +
			`{"role":"assistant","content":"two"},{"role":"user","content":"three"}]}`},
		"Messages with a tool result": {llmprotocol.AnthropicMessagesV1, `{"model":"auto","max_tokens":64,"messages":[` +
			`{"role":"user","content":"weather?"},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"city":"Paris"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"sunny"}]},` +
			`{"role":"user","content":"summarize"}],` +
			`"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`},
	} {
		t.Run(name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(test.client, []byte(test.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			request.Generation++
			ctx := routingTestContext(test.client, &request)
			ctx.VSRSelectedDecision = &config.Decision{Name: "default"}
			response, err := router.handleEntrypointModelRouting(
				&request, "auto", "default", entropy.ReasoningDecision{}, "gpt", ctx,
			)
			if err != nil {
				t.Fatalf("routing: %v", err)
			}
			if ctx.TargetFormat != llmprotocol.OpenAIResponsesV1 {
				t.Fatalf("target format = %q", ctx.TargetFormat)
			}
			sent := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			var wire struct {
				Input []map[string]json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(sent, &wire); err != nil || len(wire.Input) < 3 {
				t.Fatalf("provider body = %s (%v)", sent, err)
			}
			for _, item := range wire.Input {
				if id, present := item["id"]; present {
					t.Fatalf("an input item carries invented id %s: %s", id, sent)
				}
			}
		})
	}
}
