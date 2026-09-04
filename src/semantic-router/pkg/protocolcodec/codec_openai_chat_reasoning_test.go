package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatReasoningControlsRoundTripThroughNeutralIR(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()
	request, envelope, _, err := codec.DecodeRequest([]byte(`{
		"model":"reasoning-model",
		"messages":[{"role":"user","content":"solve this"}],
		"reasoning_effort":"high",
		"reasoning_budget_tokens":512
	}`), policy)
	if err != nil {
		t.Fatalf("decode reasoning request: %v", err)
	}
	if request.ReasoningEffort != "high" || request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 512 {
		t.Fatalf("neutral reasoning controls = effort %q budget %v", request.ReasoningEffort, request.ReasoningBudgetTokens)
	}

	request.Generation++
	body, _, err := codec.EncodeRequest(request, envelope, policy)
	if err != nil {
		t.Fatalf("encode reasoning request: %v", err)
	}
	var wire map[string]interface{}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	// The neutral budget is re-encoded both ways Chat can hold it: its own
	// reasoning_budget_tokens, and OpenRouter's reasoning.max_tokens. The
	// effort does not travel beside the bound -- OpenRouter refuses the pair.
	if wire["reasoning_budget_tokens"] != float64(512) {
		t.Fatalf("encoded reasoning controls = %#v", wire)
	}
	if _, present := wire["reasoning_effort"]; present {
		t.Fatalf("an effort level travelled beside the bound: %#v", wire)
	}
}

func TestOpenAIChatEncodesRouterSelectedReasoningEffort(t *testing.T) {
	request := llmprotocol.Request{
		Generation:      2,
		Model:           "reasoning-model",
		ReasoningEffort: "medium",
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "solve this",
			}},
		}},
	}
	body, _, err := (OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode neutral reasoning request: %v", err)
	}
	var wire struct {
		ReasoningEffort string `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	if wire.ReasoningEffort != "medium" {
		t.Fatalf("reasoning_effort = %q", wire.ReasoningEffort)
	}
}

func TestOpenAIChatDecodesSupportedReasoningAliases(t *testing.T) {
	for _, field := range []string{"reasoning_content", "reasoning"} {
		t.Run(field, func(t *testing.T) {
			body := []byte(`{"id":"response-1","model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"answer","` + field + `":"analysis"},"finish_reason":"stop"}]}`)
			response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			var reasoning string
			for _, content := range response.Output[0].Content {
				if content.Kind == llmprotocol.ContentReasoning {
					reasoning += content.Text
				}
			}
			if reasoning != "analysis" {
				t.Fatalf("reasoning = %q", reasoning)
			}
		})
	}
}

func TestOpenAIChatStrictDecoderAcceptsClosedLogprobEvidence(t *testing.T) {
	body := []byte(`{
		"id":"response-1",
		"model":"model-a",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"answer"},
			"finish_reason":"stop",
			"logprobs":{"content":[{
				"token":"answer",
				"bytes":[97],
				"logprob":-0.1,
				"top_logprobs":[{"token":"answer","bytes":[97],"logprob":-0.1}]
			}]}
		}]
	}`)
	response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decode response with logprob evidence: %v", err)
	}
	if len(response.Evidence.TokenLogprobs) != 1 ||
		len(response.Evidence.TokenLogprobs[0].Alternatives) != 1 {
		t.Fatalf("neutral logprob evidence = %+v", response.Evidence.TokenLogprobs)
	}
}

// OpenRouter's reasoning object takes one control, not two. Its documentation
// puts "One of the following (not both):" directly above effort and max_tokens,
// and a top-level reasoning_effort is folded into that same object, so a body
// carrying a bound and an effort is refused with
//
//	Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
//
// Measured on the dev cell 2026-09-04: every thinking-arm turn on build 17 was
// answered 400 that way. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
//
// So a bound travels alone, and an effort level travels only when there is no
// bound to send.
func TestChatSendsOneReasoningControl(t *testing.T) {
	tests := []struct {
		name       string
		request    llmprotocol.Request
		wantBound  *int64
		wantEffort string
	}{
		{
			// The a6 shape: the client says nothing about thinking and the
			// allowance is all the Router has to bound the turn with.
			name: "a derived bound travels without an effort",
			request: llmprotocol.Request{
				ReasoningMode: llmprotocol.ReasoningModeAdaptive,
				Sampling:      llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(4096)},
			},
			wantBound: llmprotocol.Int64(4096),
		},
		{
			// Claude Code's default body: adaptive thinking with an effort
			// level beside it. Build 17 sent both and was refused.
			name: "a client effort does not travel beside the bound",
			request: llmprotocol.Request{
				ReasoningMode:   llmprotocol.ReasoningModeAdaptive,
				ReasoningEffort: "high",
				Sampling:        llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(32000)},
			},
			wantBound: llmprotocol.Int64(32000),
		},
		{
			name: "a stated budget travels without an effort",
			request: llmprotocol.Request{
				ReasoningMode:         llmprotocol.ReasoningModeEnabled,
				ReasoningEffort:       "high",
				ReasoningBudgetTokens: llmprotocol.Int64(2048),
				Sampling:              llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(8192)},
			},
			wantBound: llmprotocol.Int64(2048),
		},
		{
			// With no allowance there is no bound to derive, so the effort is
			// the only control the turn can carry.
			name: "an effort travels alone when there is no bound",
			request: llmprotocol.Request{
				ReasoningMode:   llmprotocol.ReasoningModeEnabled,
				ReasoningEffort: "high",
			},
			wantEffort: "high",
		},
		{
			// Thinking-off is the one case where an effort is the control:
			// "none" is the off-signal and there is nothing to bound.
			name: "thinking off states the off-signal and no bound",
			request: llmprotocol.Request{
				ReasoningMode: llmprotocol.ReasoningModeDisabled,
				Sampling:      llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(4096)},
			},
			wantEffort: "none",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := test.request
			request.Generation = 1
			request.Model = "routed-model"
			request.Messages = []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}},
			}}
			body, _, err := (OpenAIChatCodec{}).EncodeRequest(
				request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy(),
			)
			if err != nil {
				t.Fatalf("encode reasoning request: %v", err)
			}
			var wire struct {
				ReasoningEffort string `json:"reasoning_effort"`
				Reasoning       *struct {
					MaxTokens *int64  `json:"max_tokens"`
					Effort    *string `json:"effort"`
				} `json:"reasoning"`
			}
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatalf("decode encoded wire: %v", err)
			}
			if wire.Reasoning != nil && wire.Reasoning.Effort != nil {
				t.Fatalf("reasoning.effort travelled beside the bound: %s", body)
			}
			if wire.ReasoningEffort != "" && wire.Reasoning != nil && wire.Reasoning.MaxTokens != nil {
				t.Fatalf("both reasoning controls travelled, which OpenRouter refuses: %s", body)
			}
			if wire.ReasoningEffort != test.wantEffort {
				t.Fatalf("reasoning_effort = %q, want %q: %s", wire.ReasoningEffort, test.wantEffort, body)
			}
			switch {
			case test.wantBound == nil && wire.Reasoning != nil:
				t.Fatalf("a bound was sent where none applies: %s", body)
			case test.wantBound != nil && (wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil):
				t.Fatalf("no bound was sent: %s", body)
			case test.wantBound != nil && *wire.Reasoning.MaxTokens != *test.wantBound:
				t.Fatalf("reasoning.max_tokens = %d, want %d", *wire.Reasoning.MaxTokens, *test.wantBound)
			}
		})
	}
}
