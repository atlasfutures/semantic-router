package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// dispatchOutputBoundCase is one request through the router's ingress,
// dispatch and encode, to a single worker of the given wire format.
type dispatchOutputBoundCase struct {
	target   llmprotocol.WireFormat
	card     int
	body     string
	decision *config.Decision
	mutate   func(*llmprotocol.Request)
}

func dispatchWithOutputBound(t *testing.T, test dispatchOutputBoundCase) (*llmprotocol.Request, map[string]json.RawMessage) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(test.target)
	params := router.Config.ModelConfig[logicalModel]
	params.MaxOutputTokens = test.card
	router.Config.ModelConfig[logicalModel] = params
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "output-bound", TraceContext: context.Background(), VSRSelectedDecision: test.decision,
	}
	request, immediate := router.prepareProtocolRequest([]byte(test.body), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused %s: %+v", test.body, ctx.ImmediateProtocolError)
	}
	if test.mutate != nil {
		test.mutate(request)
	}
	if _, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	return request, wire
}

func wireOutputLimit(wire map[string]json.RawMessage) string {
	for _, field := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
		if value, ok := wire[field]; ok {
			return field + "=" + string(value)
		}
	}
	return ""
}

const unboundedResponsesBody = `{"model":"m","input":"write the long tool call"}`

// A Responses request that states no output limit, dispatched to a Messages
// worker, is sent the worker's card max_output_tokens, and the fallback when
// the card declares none. It is never the codec's generated default.
func TestDispatchOutputBoundOnMessagesUsesCardThenFallback(t *testing.T) {
	for name, test := range map[string]struct {
		card int
		want string
	}{
		"card declared": {16000, "max_tokens=16000"},
		"card silent":   {0, "max_tokens=32000"},
	} {
		t.Run(name, func(t *testing.T) {
			request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.AnthropicMessagesV1, card: test.card, body: unboundedResponsesBody,
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
			if !request.RouterSetMaxOutputTokens {
				t.Fatal("the router's bound is not marked as the router's")
			}
		})
	}
}

// A limit the caller stated is sent as stated, whatever the card says.
func TestDispatchOutputBoundKeepsAnExplicitLimit(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
			target: target, card: 16000, body: `{"model":"m","input":"hi","max_output_tokens":500}`,
		})
		if got := wireOutputLimit(wire); got == "" || got[len(got)-4:] != "=500" {
			t.Fatalf("-> %s: dispatched %q, want the caller's 500", target, got)
		}
		if request.RouterSetMaxOutputTokens {
			t.Fatalf("-> %s: the caller's limit was marked as the router's", target)
		}
	}
}

// Chat and Responses do not require a limit: with no card value the request
// goes out unbounded, as the caller asked. A card value is sent on them too.
func TestDispatchOutputBoundLeavesOptionalTargetsUnboundedWithoutACard(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{target: target, body: unboundedResponsesBody})
		if got := wireOutputLimit(wire); got != "" {
			t.Fatalf("-> %s without a card: dispatched %q, want no limit", target, got)
		}
		_, wire = dispatchWithOutputBound(t, dispatchOutputBoundCase{target: target, card: 16000, body: unboundedResponsesBody})
		if got := wireOutputLimit(wire); got == "" || got[len(got)-6:] != "=16000" {
			t.Fatalf("-> %s with a card: dispatched %q, want 16000", target, got)
		}
	}
}

// A card limit is the router's, not the caller's allowance, so a Chat turn
// that asks to reason gets no reasoning bound derived from it: its effort
// travels as it would with no limit at all.
func TestDispatchOutputBoundDerivesNoReasoningBoundOnChat(t *testing.T) {
	_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.OpenAIChatV1, card: 16000,
		body: `{"model":"m","input":"hi","reasoning":{"effort":"high"}}`,
	})
	if _, bounded := wire["reasoning"]; bounded {
		t.Fatalf("a reasoning bound was derived from the card limit: %s", wire["reasoning"])
	}
	if string(wire["reasoning_effort"]) != `"high"` {
		t.Fatalf("reasoning_effort = %s, want the caller's high", wire["reasoning_effort"])
	}
}

// Messages refuses a thinking budget that is not below max_tokens. A budget at
// or above the bound gets the bound on top of it; one below is left alone.
// Either way the encoder accepts the request.
func TestDispatchOutputBoundKeepsRoomAboveAThinkingBudget(t *testing.T) {
	for name, test := range map[string]struct {
		card   int
		budget int64
		want   string
	}{
		"budget above the card":     {8000, 40000, "max_tokens=48000"},
		"budget above the fallback": {0, 40000, "max_tokens=72000"},
		"budget below the fallback": {0, 4096, "max_tokens=32000"},
		"budget below the card":     {16000, 4096, "max_tokens=16000"},
	} {
		t.Run(name, func(t *testing.T) {
			_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.AnthropicMessagesV1, card: test.card, body: unboundedResponsesBody,
				mutate: func(request *llmprotocol.Request) {
					request.ReasoningMode = llmprotocol.ReasoningModeEnabled
					request.ReasoningBudgetTokens = &test.budget
				},
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
			var thinking struct {
				BudgetTokens int64 `json:"budget_tokens"`
			}
			if err := json.Unmarshal(wire["thinking"], &thinking); err != nil || thinking.BudgetTokens != test.budget {
				t.Fatalf("thinking = %s, want budget %d", wire["thinking"], test.budget)
			}
		})
	}
}

// The decision's request_params max_tokens_limit caps the router's bound as it
// caps a caller's limit.
func TestDispatchOutputBoundStaysWithinTheDecisionLimit(t *testing.T) {
	payload, err := config.NewStructuredPayload(map[string]interface{}{"max_tokens_limit": 2000})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name:    "capped",
		Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}},
	}
	_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 16000, body: unboundedResponsesBody, decision: decision,
	})
	if got := wireOutputLimit(wire); got != "max_tokens=2000" {
		t.Fatalf("dispatched %q, want the decision's 2000", got)
	}
}

func maxTokensLimitDecision(t *testing.T, limit int) *config.Decision {
	t.Helper()
	payload, err := config.NewStructuredPayload(map[string]interface{}{"max_tokens_limit": limit})
	if err != nil {
		t.Fatal(err)
	}
	return &config.Decision{
		Name:    "capped",
		Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}},
	}
}

// The router never synthesizes a limit its target refuses. Responses refuses
// max_output_tokens below 16, so a smaller bound is raised to 16, unless 16
// is above the decision's max_tokens_limit: then the request goes unbounded,
// as its caller sent it. Chat and Messages accept any positive bound.
func TestDispatchOutputBoundRespectsTheTargetMinimum(t *testing.T) {
	for name, test := range map[string]struct {
		target llmprotocol.WireFormat
		card   int
		limit  int
		want   string
	}{
		"responses card 1":             {llmprotocol.OpenAIResponsesV1, 1, 0, "max_output_tokens=16"},
		"responses card 15":            {llmprotocol.OpenAIResponsesV1, 15, 0, "max_output_tokens=16"},
		"responses card 16":            {llmprotocol.OpenAIResponsesV1, 16, 0, "max_output_tokens=16"},
		"responses card 17":            {llmprotocol.OpenAIResponsesV1, 17, 0, "max_output_tokens=17"},
		"responses capped to 16":       {llmprotocol.OpenAIResponsesV1, 64, 16, "max_output_tokens=16"},
		"responses capped to 15":       {llmprotocol.OpenAIResponsesV1, 64, 15, ""},
		"responses card 10, cap 12":    {llmprotocol.OpenAIResponsesV1, 10, 12, ""},
		"responses card 10, cap 20":    {llmprotocol.OpenAIResponsesV1, 10, 20, "max_output_tokens=16"},
		"chat card 1":                  {llmprotocol.OpenAIChatV1, 1, 0, "max_completion_tokens=1"},
		"messages card 1":              {llmprotocol.AnthropicMessagesV1, 1, 0, "max_tokens=1"},
		"messages capped to 1":         {llmprotocol.AnthropicMessagesV1, 64, 1, "max_tokens=1"},
		"responses no card, capped":    {llmprotocol.OpenAIResponsesV1, 0, 10, ""},
		"messages fallback capped low": {llmprotocol.AnthropicMessagesV1, 0, 10, "max_tokens=10"},
	} {
		t.Run(name, func(t *testing.T) {
			var decision *config.Decision
			if test.limit > 0 {
				decision = maxTokensLimitDecision(t, test.limit)
			}
			request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: test.target, card: test.card, body: unboundedResponsesBody, decision: decision,
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
			if test.want == "" && request.RouterSetMaxOutputTokens {
				t.Fatal("an unbounded request is marked as carrying the router's limit")
			}
		})
	}
}

// A decision that selects a LoRA adapter dispatches under the adapter's name.
// The adapter has no card of its own and takes its base model's limit.
func TestDispatchOutputBoundTakesALoRAAdaptersBaseCard(t *testing.T) {
	router, base := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	params := router.Config.ModelConfig[base]
	params.MaxOutputTokens = 12000
	params.LoRAs = []config.LoRAAdapter{{Name: "sql-adapter"}}
	router.Config.ModelConfig[base] = params
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "output-bound-lora", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(unboundedResponsesBody), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused: %+v", ctx.ImmediateProtocolError)
	}
	if _, err := router.prepareProviderDispatch(request, "sql-adapter", "", false, ctx); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	if got := wireOutputLimit(wire); got != "max_tokens=12000" {
		t.Fatalf("dispatched %q, want the base card's 12000", got)
	}
}
