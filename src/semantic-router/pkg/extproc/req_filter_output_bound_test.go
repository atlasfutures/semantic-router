package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
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
	// window is the card's context_window_size and contextTokens the
	// request's token estimate, both zero unless a test sets them.
	window        int
	contextTokens int
}

func dispatchWithOutputBound(t *testing.T, test dispatchOutputBoundCase) (*llmprotocol.Request, map[string]json.RawMessage) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(test.target)
	params := router.Config.ModelConfig[logicalModel]
	params.MaxOutputTokens = test.card
	params.ContextWindowSize = test.window
	router.Config.ModelConfig[logicalModel] = params
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "output-bound", TraceContext: context.Background(), VSRSelectedDecision: test.decision,
		VSRContextTokenCount: test.contextTokens,
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

// Under a decision's max_tokens_limit the thinking expansion never dispatches
// above the cap: max_tokens becomes the cap and the budget is lowered to fit,
// or thinking is turned off when the cap cannot hold the minimum budget and
// an output token. A cap with room keeps the budget and the bound on top.
// Every case encodes.
func TestDispatchOutputBoundKeepsThinkingWithinTheDecisionLimit(t *testing.T) {
	for name, test := range map[string]struct {
		card       int
		limit      int
		budget     int64
		wantLimit  string
		wantBudget int64 // 0: thinking disabled
	}{
		"cap 2000, budget 40000":          {0, 2000, 40000, "max_tokens=2000", 1024},
		"cap below the minimum budget":    {0, 1000, 40000, "max_tokens=1000", 0},
		"cap at the minimum budget":       {0, 1024, 40000, "max_tokens=1024", 0},
		"cap one above the minimum":       {0, 1025, 40000, "max_tokens=1025", 1024},
		"card leaves room beside the cap": {1500, 3000, 2000, "max_tokens=3000", 1500},
		"cap comfortably above":           {16000, 100000, 40000, "max_tokens=56000", 40000},
	} {
		t.Run(name, func(t *testing.T) {
			request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.AnthropicMessagesV1, card: test.card, body: unboundedResponsesBody,
				decision: maxTokensLimitDecision(t, test.limit),
				mutate: func(request *llmprotocol.Request) {
					request.ReasoningMode = llmprotocol.ReasoningModeEnabled
					request.ReasoningBudgetTokens = &test.budget
				},
			})
			if got := wireOutputLimit(wire); got != test.wantLimit {
				t.Fatalf("dispatched %q, want %q", got, test.wantLimit)
			}
			var thinking struct {
				Type         string `json:"type"`
				BudgetTokens int64  `json:"budget_tokens"`
			}
			if err := json.Unmarshal(wire["thinking"], &thinking); err != nil {
				t.Fatalf("thinking = %s: %v", wire["thinking"], err)
			}
			if test.wantBudget == 0 {
				if thinking.Type != "disabled" || request.ReasoningBudgetTokens != nil {
					t.Fatalf("thinking = %s, want it disabled", wire["thinking"])
				}
				return
			}
			if thinking.Type != "enabled" || thinking.BudgetTokens != test.wantBudget {
				t.Fatalf("thinking = %s, want budget %d", wire["thinking"], test.wantBudget)
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

// The card's output limit is sent whole only when the prompt leaves that much
// room in the card's context window. A provider refuses input plus output
// limit above its window, so a long prompt the ARC mask admitted would
// otherwise fail at the provider on the limit the router itself added.
func TestDispatchOutputBoundStaysWithinTheContextRoom(t *testing.T) {
	for name, test := range map[string]struct {
		target        llmprotocol.WireFormat
		card, window  int
		contextTokens int
		decision      *config.Decision
		want          string
	}{
		"room below the card clamps":                                    {llmprotocol.OpenAIChatV1, 384000, 1000000, 700000, nil, "max_completion_tokens=300000"},
		"room above the card sends the card":                            {llmprotocol.OpenAIChatV1, 384000, 1000000, 100000, nil, "max_completion_tokens=384000"},
		"no window declared, no clamp":                                  {llmprotocol.OpenAIChatV1, 384000, 0, 700000, nil, "max_completion_tokens=384000"},
		"no estimate, no clamp":                                         {llmprotocol.OpenAIChatV1, 384000, 1000000, 0, nil, "max_completion_tokens=384000"},
		"the tighter of room and decision cap":                          {llmprotocol.OpenAIChatV1, 384000, 1000000, 700000, maxTokensLimitDecision(t, 2000), "max_completion_tokens=2000"},
		"room tighter than the decision cap":                            {llmprotocol.OpenAIChatV1, 384000, 1000000, 999000, maxTokensLimitDecision(t, 2000), "max_completion_tokens=1000"},
		"no room left sends the card, as if no window":                  {llmprotocol.OpenAIChatV1, 384000, 262144, 262144, nil, "max_completion_tokens=384000"},
		"messages with no room sends the fallback, not a smaller limit": {llmprotocol.AnthropicMessagesV1, 0, 40000, 40000, nil, "max_tokens=32000"},
		"messages fallback clamps to the room":                          {llmprotocol.AnthropicMessagesV1, 0, 40000, 30000, nil, "max_tokens=10000"},
		"responses room below 16 goes unbounded":                        {llmprotocol.OpenAIResponsesV1, 384000, 1000000, 999990, nil, ""},
	} {
		t.Run(name, func(t *testing.T) {
			body := unboundedResponsesBody
			_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: test.target, card: test.card, body: body, decision: test.decision,
				window: test.window, contextTokens: test.contextTokens,
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
		})
	}
}

// A Messages thinking budget keeps the bound on top of it within the room,
// the way it stays within a decision cap: the budget is lowered to what the
// room leaves beside the bound, never below Anthropic's minimum.
func TestDispatchOutputBoundKeepsThinkingWithinTheContextRoom(t *testing.T) {
	budget := int64(8000)
	request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 4000, body: unboundedResponsesBody,
		window: 100000, contextTokens: 94000,
		mutate: func(request *llmprotocol.Request) {
			request.ReasoningMode, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeEnabled, &budget
		},
	})
	if got := wireOutputLimit(wire); got != "max_tokens=6000" {
		t.Fatalf("dispatched %q, want max_tokens at the room 6000", got)
	}
	if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 2000 {
		t.Fatalf("budget = %v, want lowered to the 2000 the room leaves beside the bound", request.ReasoningBudgetTokens)
	}
}

// The room is measured against the prompt as it will be dispatched. The
// routing estimate was made before a stored history or a retrieval was
// prepended, so a request that grew after it is recounted, and the larger
// number wins.
func TestDispatchOutputBoundRecountsTheDispatchedPrompt(t *testing.T) {
	_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.OpenAIChatV1, card: 384000, body: unboundedResponsesBody,
		window: 1000000, contextTokens: 100,
		mutate: func(request *llmprotocol.Request) {
			history := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: strings.Repeat("restored history ", 200000),
			}}}
			request.Messages = append([]llmprotocol.Message{history}, request.Messages...)
		},
	})
	got := wireOutputLimit(wire)
	var limit int64
	if _, err := fmt.Sscanf(got, "max_completion_tokens=%d", &limit); err != nil || limit >= 384000 {
		t.Fatalf("dispatched %q, want a limit clamped below the card by the recounted prompt", got)
	}
}

// Whichever constraint holds the plan is named: in the log, and in the
// refusal of a v5 control the cap cannot hold, which otherwise sends an
// operator to change a decision limit that was not the problem.
func TestDispatchOutputBoundNamesTheConstraintThatHeldIt(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{"m": {MaxOutputTokens: 4000}},
	}}}
	budget := int64(8000)
	// The card fits the room; the thinking budget on top does not. The clamp
	// came from the context, and the plan says so.
	plan := router.planDispatchOutputBound("m", llmprotocol.AnthropicMessagesV1, 0, &budget, 100000, 94000)
	if plan.context != outputBoundClampedToContext || plan.limitedBy != outputBoundLimitedByContext || plan.maxTokens != 6000 {
		t.Fatalf("plan = %+v, want clamped_to_context by the context at 6000", plan)
	}
	// A decision cap tighter than the room is the decision's doing.
	plan = router.planDispatchOutputBound("m", llmprotocol.AnthropicMessagesV1, 5000, &budget, 100000, 94000)
	if plan.context != "" || plan.limitedBy != outputBoundLimitedByDecision || plan.maxTokens != 5000 {
		t.Fatalf("plan = %+v, want the decision's 5000 with no context status", plan)
	}
	// Room that cannot hold the minimum budget beside the bound disables
	// thinking, and the control refusal names the context, not the decision.
	plan = router.planDispatchOutputBound("m", llmprotocol.AnthropicMessagesV1, 0, &budget, 100000, 99000)
	if plan.thinking != outputBoundThinkingDisabled || plan.limitedBy != outputBoundLimitedByContext {
		t.Fatalf("plan = %+v, want thinking disabled by the context", plan)
	}
	if err := thinkingControlOverCapError(plan); !errors.Is(err, errThinkingControlOverContext) {
		t.Fatalf("refusal = %v, want the context-window refusal", err)
	}
	plan = router.planDispatchOutputBound("m", llmprotocol.AnthropicMessagesV1, 4500, &budget, 0, 0)
	if err := thinkingControlOverCapError(plan); !errors.Is(err, errThinkingControlOverCap) {
		t.Fatalf("refusal = %v, want the decision-cap refusal", err)
	}
}

// Once compression has run, the routing estimate describes a prompt that no
// longer exists; the fresh count of the compressed request stands alone, so
// the room, and the limit, grow back.
func TestDispatchContextTokensFollowsCompression(t *testing.T) {
	t.Parallel()
	request := testNeutralRequest("auto", "a short prompt after compression")
	stale := &RequestContext{VSRContextTokenCount: 90000}
	if got := dispatchContextTokens(request, stale); got != 90000 {
		t.Fatalf("uncompressed count = %d, want the larger routing estimate 90000", got)
	}
	stale.ContextCompressionApplied = true
	if got := dispatchContextTokens(request, stale); got >= 90000 || got <= 0 {
		t.Fatalf("compressed count = %d, want the fresh count of the compressed request", got)
	}
}

// Compression reserves the whole bound, unclamped: it exists to free that
// room, so a prompt that currently leaves less must be compressed to fit the
// card's limit, not told the limit is smaller.
func TestCompressionReservesTheUnclampedBound(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{"m": {MaxOutputTokens: 30000, ContextWindowSize: 100000, APIFormat: config.APIFormatOpenAI}},
	}}}
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, VSRContextTokenCount: 90000}
	if got := router.compressionOutputReserve("m", ctx, testNeutralRequest("auto", "long")); got != 30000 {
		t.Fatalf("reserve = %d, want the card's 30000 rather than the 10000 the prompt leaves", got)
	}
}

// A plan that stayed under every cap names no constraint; only a cap that
// held it does.
func TestDispatchOutputBoundNamesNoConstraintItStayedUnder(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{"m": {MaxOutputTokens: 4000}},
	}}}
	plan := router.planDispatchOutputBound("m", llmprotocol.OpenAIChatV1, 5000, nil, 100000, 94000)
	if plan.limitedBy != "" || plan.context != "" || plan.maxTokens != 4000 {
		t.Fatalf("plan = %+v, want the card unheld and unnamed", plan)
	}
}

// Semantic tool selection runs after the limit is planned and can replace
// the tools with larger retrieved definitions. The limit is measured once
// more before encoding and only lowered, never raised, never below the
// target's minimum or a Messages thinking budget.
func TestDispatchOutputBoundIsReclampedAfterToolSelection(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{"m": {ContextWindowSize: 1000000}},
	}}}
	limit := int64(384000)
	request := testNeutralRequest("auto", "hi")
	request.Sampling.MaxOutputTokens, request.RouterSetMaxOutputTokens = &limit, true
	request.Tools = []llmprotocol.Tool{{Name: "retrieved", Description: strings.Repeat("schema ", 500000)}}
	ctx := &RequestContext{VSRContextTokenCount: 100}
	dispatch := &providerDispatch{logicalModel: "m", targetFormat: llmprotocol.OpenAIChatV1}
	router.reclampDispatchOutputBound(request, dispatch, ctx)
	if *request.Sampling.MaxOutputTokens >= 384000 {
		t.Fatalf("limit = %d, want lowered by the retrieved tools", *request.Sampling.MaxOutputTokens)
	}
	// A caller's own limit is never touched.
	caller := int64(384000)
	request.Sampling.MaxOutputTokens, request.RouterSetMaxOutputTokens = &caller, false
	router.reclampDispatchOutputBound(request, dispatch, ctx)
	if *request.Sampling.MaxOutputTokens != 384000 {
		t.Fatalf("a caller's limit was lowered to %d", *request.Sampling.MaxOutputTokens)
	}
	// On Messages the limit stays above the thinking budget it must exceed.
	budget := int64(300000)
	request.Sampling.MaxOutputTokens, request.RouterSetMaxOutputTokens, request.ReasoningBudgetTokens = &limit, true, &budget
	router.reclampDispatchOutputBound(request, &providerDispatch{logicalModel: "m", targetFormat: llmprotocol.AnthropicMessagesV1}, ctx)
	if *request.Sampling.MaxOutputTokens <= budget {
		t.Fatalf("limit = %d, want above the %d budget", *request.Sampling.MaxOutputTokens, budget)
	}
}
