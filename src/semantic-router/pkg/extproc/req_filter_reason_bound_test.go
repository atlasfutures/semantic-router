package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An arm reasons because its own configuration says so, not because the client
// asked. The a6 corpus asked for no reasoning at all and every turn on
// deepseek/deepseek-v4-flash@thinking-on still reasoned: routing_decision
// records reasoning_enabled true, reasoning_effort medium, from the arm's
// ModelRef. The client's request carries no thinking field, so the Chat encoder
// sends no bound, and the bound the encoder does know how to send never
// applies to the traffic that hangs. Three of 17 thinking-arm turns on the dev
// cell 2026-09-03 ran to the 630 s platform deadline.
//
// The provider boundary is where the arm's reasoning setting is applied
// (adaptProviderRequest -> setReasoningModeToRequestBodyForModelAndProvider),
// so it is where the bound belongs. The rule is the encoder's: what the client
// allowed for output, floored at the smallest budget a provider accepts.
func TestArmThatReasonsByConfigStatesItsBound(t *testing.T) {
	tests := []struct {
		name      string
		allowance string
		wantBound int64
	}{
		{
			name:      "the client's output allowance is the bound",
			allowance: `"max_completion_tokens":4096`,
			wantBound: 4096,
		},
		{
			name:      "a small allowance is floored",
			allowance: `"max_completion_tokens":512`,
			wantBound: 1024,
		},
		{
			// Not every leg writes the allowance as max_completion_tokens.
			name:      "max_tokens is an output allowance too",
			allowance: `"max_tokens":8192`,
			wantBound: 8192,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
				test.allowance+`}`)
			bound := reasoningBoundOf(t, body)
			if bound == nil {
				t.Fatalf("an arm that reasons by configuration sent no bound: %v", body)
			}
			if *bound != test.wantBound {
				t.Fatalf("reasoning.max_tokens = %d, want %d", *bound, test.wantBound)
			}
		})
	}
}

// A bound the request already carries is the one that stands: the Chat encoder
// derived it from the client's own request and the boundary does not overrule
// it with a number of its own.
func TestArmKeepsABoundTheRequestAlreadyCarries(t *testing.T) {
	body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
		`"max_completion_tokens":4096,"reasoning":{"max_tokens":2048}}`)
	bound := reasoningBoundOf(t, body)
	if bound == nil || *bound != 2048 {
		t.Fatalf("the bound the request carried was replaced: %v", body)
	}
}

// With no output allowance there is nothing to derive a bound from, and
// capping a client that asked for no cap is not the Router's to do. The arm's
// effort is then the only reasoning control the request can carry.
func TestArmKeepsItsEffortWithoutAnOutputAllowance(t *testing.T) {
	body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`)
	if bound := reasoningBoundOf(t, body); bound != nil {
		t.Fatalf("a bound was invented with no output allowance: %v", body)
	}
	if body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %v, want the arm's effort", body["reasoning_effort"])
	}
}

// reasoning.max_tokens is OpenRouter's control. Every other OpenAI-compatible
// backend would see an unknown member.
func TestOnlyOpenRouterGetsTheReasoningBound(t *testing.T) {
	body := reasoningBodyForProfile(t,
		`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":4096}`,
		true,
		&config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"},
	)
	if _, present := body["reasoning"]; present {
		t.Fatalf("an OpenRouter-only member reached another backend: %v", body)
	}
	if body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %v, want the arm's effort", body["reasoning_effort"])
	}
}

// A thinking-off arm gets no bound. Sending one asks it to start reasoning.
func TestThinkingOffArmGetsNoBound(t *testing.T) {
	body := reasoningBodyForProfile(t,
		`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":4096}`,
		false,
		openRouterProviderProfile(),
	)
	if _, present := body["reasoning"]; present {
		t.Fatalf("a thinking-off arm was told to reason: %v", body)
	}
}

func openRouterProviderProfile() *config.ProviderProfile {
	return &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}
}

// armReasoningBody routes one body through the provider boundary on an arm
// whose ModelRef turns reasoning on with effort high.
func armReasoningBody(t *testing.T, body string) map[string]interface{} {
	t.Helper()
	return reasoningBodyForProfile(t, body, true, openRouterProviderProfile())
}

func reasoningBodyForProfile(
	t *testing.T,
	body string,
	enabled bool,
	profile *config.ProviderProfile,
) map[string]interface{} {
	t.Helper()
	router := newArmReasoningRouter()
	encoded, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		[]byte(body), "gpt-5-mini", enabled, router.Config.GetDecisionByName("arc"), profile, nil,
	)
	require.NoError(t, err)
	var decoded map[string]interface{}
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	return decoded
}

func reasoningBoundOf(t *testing.T, body map[string]interface{}) *int64 {
	t.Helper()
	reasoning, present := body["reasoning"].(map[string]interface{})
	if !present {
		return nil
	}
	maxTokens, present := reasoning["max_tokens"].(float64)
	if !present {
		return nil
	}
	bound := int64(maxTokens)
	return &bound
}

func newArmReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"gpt-oss": {Type: "reasoning_effort", Parameter: "reasoning_effort"},
			},
		},
		[]config.Decision{
			reasoningDecision("arc", "", 0, "gpt-5-mini", boolPtr(true), "high"),
		},
		map[string]config.ModelParams{"gpt-5-mini": {ReasoningFamily: "gpt-oss"}},
	)
}

// The provider boundary sends one reasoning control, never two. OpenRouter
// reads a top-level reasoning_effort as reasoning.effort and refuses a body
// that also carries reasoning.max_tokens:
//
//	Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
//
// Measured on the dev cell 2026-09-04, that 400 answered every thinking-arm
// turn on build 17, which sent the arm's effort beside the bound. The arm's
// configured effort is the one that goes: the bound is the number that stops
// the turn, and the effort dial cannot be stated beside it. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
func TestOneReasoningControlReachesOpenRouter(t *testing.T) {
	t.Run("a bound leaves no effort anywhere", func(t *testing.T) {
		body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
			`"max_completion_tokens":4096}`)
		bound := reasoningBoundOf(t, body)
		if bound == nil || *bound != 4096 {
			t.Fatalf("the arm's turn was not bounded: %v", body)
		}
		assertNoReasoningEffort(t, body)
	})

	t.Run("a client effort does not survive beside the bound", func(t *testing.T) {
		body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
			`"max_completion_tokens":4096,"reasoning_effort":"low"}`)
		if bound := reasoningBoundOf(t, body); bound == nil {
			t.Fatalf("a client-stated effort left the turn unbounded: %v", body)
		}
		assertNoReasoningEffort(t, body)
	})

	t.Run("a carried bound also travels alone", func(t *testing.T) {
		body := armReasoningBody(t, `{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
			`"max_completion_tokens":4096,"reasoning":{"max_tokens":2048}}`)
		bound := reasoningBoundOf(t, body)
		if bound == nil || *bound != 2048 {
			t.Fatalf("the bound the request carried was replaced: %v", body)
		}
		assertNoReasoningEffort(t, body)
	})
}

// assertNoReasoningEffort fails when an effort level reached the wire in either
// placement OpenRouter reads: the top-level alias or the reasoning object.
func assertNoReasoningEffort(t *testing.T, body map[string]interface{}) {
	t.Helper()
	if effort, present := body["reasoning_effort"]; present {
		t.Fatalf("reasoning_effort %v travelled beside the bound; OpenRouter refuses both", effort)
	}
	reasoning, present := body["reasoning"].(map[string]interface{})
	if !present {
		return
	}
	if effort, present := reasoning["effort"]; present {
		t.Fatalf("reasoning.effort %v travelled beside the bound", effort)
	}
}

// The second place the bound is derived is the provider boundary, for an arm
// that reasons by its own configuration. It reads the dispatched body, where
// the request_params floor has already raised max_completion_tokens, so it is
// told separately what the client allowed. The a6 corpus is this shape: the
// client asks for no reasoning at all and the arm reasons anyway.
func TestArmBoundComesFromTheClientAllowanceNotTheFloor(t *testing.T) {
	router := newArmReasoningRouter()
	ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{
		ClientMaxOutputTokens: llmprotocol.Int64(512),
	}}
	encoded, err := router.setReasoningModeToRequestBodyForModelAndProvider(
		[]byte(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}],`+
			`"max_completion_tokens":65536}`),
		"gpt-5-mini", true, router.Config.GetDecisionByName("arc"),
		openRouterProviderProfile(), ctx,
	)
	require.NoError(t, err)
	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(encoded, &body))

	bound := reasoningBoundOf(t, body)
	if bound == nil || *bound != 1024 {
		t.Fatalf("the bound came from the floor, not the client's 512: %v", body)
	}
	if body["max_completion_tokens"] != float64(65536) {
		t.Fatalf("the floor's raise was undone: %v", body["max_completion_tokens"])
	}
}
