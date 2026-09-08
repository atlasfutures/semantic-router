package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responseLegRefusals are refusals the response leg raises for a client whose
// protocol is Anthropic Messages. Each one is built by createErrorResponse
// (router.go:188), which always writes the OpenAI envelope {"error":{...}}.
//
// The request phases re-encode an immediate response into the client's protocol
// before sending it (processor_core.go:227 and :248 call
// encodeImmediateResponseForClient). The response phases do not: they hand the
// response straight to sendResponse (processor_core.go:282-303).
var responseLegRefusals = map[string]func(*OpenAIRouter, *RequestContext) *ext_proc.ProcessingResponse{
	// processor_res_body_pipeline.go:31 - the upstream body does not decode.
	"undecodable upstream response": func(router *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
		return router.handleNonStreamingResponseBody([]byte("not a model response"), ctx, 0)
	},
	// res_filter_jailbreak.go:74 - the response-stage guardrail blocks the output.
	"response jailbreak blocked": func(router *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
		ctx.VSRMatchedResponseJailbreak = []string{"unsafe_completion"}
		ctx.VSRSelectedDecision = &config.Decision{
			Name: "guarded",
			Plugins: []config.DecisionPlugin{{
				Type: "response_jailbreak",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
					"action":  "block",
				}),
			}},
		}
		return router.enforceResponseJailbreakFromSignal(ctx, "guarded")
	},
}

// refusalBody drives one response-leg refusal for an Anthropic Messages client
// and returns the body the client receives.
func refusalBody(t *testing.T, drive func(*OpenAIRouter, *RequestContext) *ext_proc.ProcessingResponse) []byte {
	t.Helper()
	ctx := &RequestContext{
		Headers:      map[string]string{},
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		RequestID:    "response-leg-refusal",
	}
	response := drive(&OpenAIRouter{}, ctx)
	body := response.GetImmediateResponse().GetBody()
	if len(body) == 0 {
		t.Fatalf("expected a refusal body, got %v", response)
	}
	return body
}

// TestResponseLegRefusalIsEncodedInTheClientProtocol is the ask.
//
// An Anthropic Messages client parses errors as {"type":"error","error":{...}}.
// A response-leg refusal reaches it as the OpenAI {"error":{...}} object, which
// carries no "type" discriminator, so the client does not recognise it as an
// error and reads it as a message.
func TestResponseLegRefusalIsEncodedInTheClientProtocol(t *testing.T) {
	for name, drive := range responseLegRefusals {
		body := refusalBody(t, drive)
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(body, &envelope); err != nil {
			t.Fatalf("%s: decode refusal body: %v (body=%s)", name, err, body)
		}
		if envelope.Type != "error" {
			t.Errorf("%s: an Anthropic client is handed %s", name, body)
		}
	}
}

// TestResponseLegRefusalIsOpenAIShapedToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
func TestResponseLegRefusalIsOpenAIShapedToday(t *testing.T) {
	for name, drive := range responseLegRefusals {
		body := refusalBody(t, drive)
		var envelope struct {
			Type  string `json:"type"`
			Error struct {
				Code int `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(body, &envelope); err != nil {
			t.Fatalf("%s: decode refusal body: %v (body=%s)", name, err, body)
		}
		if envelope.Type != "" || envelope.Error.Code == 0 {
			t.Errorf("%s: the recorded behaviour has changed: %s", name, body)
		}
	}
}
