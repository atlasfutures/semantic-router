package extproc

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// writeClaudeOverChatPolicyConfig is the e2e policy config with the
// claude-off worker dispatched as a Chat Completions request through
// OpenRouter, as a cell that sends every worker over Chat does, and the two
// open-weight workers published as kimi (off) and glm (think) are.
func writeClaudeOverChatPolicyConfig(t *testing.T, policyURL string) string {
	t.Helper()
	path := writeConsistentPolicyConfig(t, policyURL, "strict")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	from := `    - name: claude-off
      provider_model_id: anthropic/claude-opus-5
      api_format: anthropic
`
	to := `    - name: claude-off
      provider_model_id: anthropic/claude-opus-5
      api_format: openai
`
	backendFrom := `        - name: anthropic-claude-off
          base_url: https://api.anthropic.com
          provider: anthropic
`
	backendTo := `        - name: anthropic-claude-off
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
`
	cards := [][2]string{
		{"    - name: think\n      modality: text\n", "    - name: think\n      modality: text\n      publisher: z-ai\n"},
		{"    - name: \"off\"\n      modality: text\n", "    - name: \"off\"\n      modality: text\n      publisher: moonshotai\n"},
	}
	rendered := string(raw)
	for _, swap := range append([][2]string{{from, to}, {backendFrom, backendTo}}, cards...) {
		if !strings.Contains(rendered, swap[0]) {
			t.Fatalf("the policy e2e config no longer carries %q", swap[0])
		}
		rendered = strings.Replace(rendered, swap[0], swap[1], 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// piChatHistory is a pi Chat Completions request whose history holds one
// earlier assistant turn, with the given reasoning members, that called a
// tool, the tool's result, and a new user turn.
func piChatHistory(reasoning string) string {
	return `{"model":"auto","stream":true,"max_completion_tokens":32000,"messages":[` +
		`{"role":"developer","content":"You are pi."},` +
		`{"role":"user","content":"Create fib.py and run it."},` +
		`{"role":"assistant","content":null,` + reasoning + `,` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"python3 fib.py\"}"}}]},` +
		`{"role":"tool","tool_call_id":"call_1","content":"0 1 1 2 3 5 8 13 21 34"},` +
		`{"role":"user","content":"Now print 15 numbers."}],` +
		`"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]}`
}

const kimiChatReasoning = `"reasoning_content":"kimi reasoning",` +
	`"reasoning_details":[{"type":"reasoning.text","text":"kimi reasoning","format":"unknown","index":0}]`

// End to end: a conversation whose episode has no record whose history holds
// another family's reasoning reaches a Claude worker dispatched over Chat
// without it. OpenRouter hands an assistant message's reasoning to the model,
// and Anthropic answers a foreign model's thinking with a content_filter
// refusal. Claude's own reasoning still reaches it, and an open-weight
// worker still gets the reasoning_content it needs in a tool history.
func TestForeignReasoningNeverReachesAClaudeWorkerOverChat(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	assistantOf := func(t *testing.T, body map[string]json.RawMessage) map[string]json.RawMessage {
		t.Helper()
		var messages []map[string]json.RawMessage
		if err := json.Unmarshal(body["messages"], &messages); err != nil {
			t.Fatalf("messages = %s", body["messages"])
		}
		for _, message := range messages {
			if string(message["role"]) == `"assistant"` {
				return message
			}
		}
		t.Fatalf("the assistant turn was lost: %s", body["messages"])
		return nil
	}

	t.Run("kimi's reasoning to Claude over Chat", func(t *testing.T) {
		claude := actions["claude-off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		logs := captureLogs(t)
		body := dispatchPolicyClientRequest(t, router, "episode-family-1", "/v1/chat/completions", piChatHistory(kimiChatReasoning))
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "kimi reasoning") || strings.Contains(wire, "reasoning_content") ||
			strings.Contains(wire, "reasoning_details") {
			t.Fatalf("a Claude worker was sent another family's reasoning: %s", wire)
		}
		assistant := assistantOf(t, body)
		if !strings.Contains(string(assistant["tool_calls"]), "call_1") || !strings.Contains(wire, `"tool_call_id":"call_1"`) {
			t.Fatalf("the tool call or its result was lost: %s", wire)
		}
		dropped := findLogEvent(t, logs, "reasoning_dropped")
		if fmt.Sprint(dropped["foreign_dropped"]) != "1" || fmt.Sprint(dropped["dropped"]) != "1" {
			t.Fatalf("reasoning_dropped = %v, want 1 foreign drop", dropped)
		}
	})
	t.Run("Claude's own reasoning to Claude over Chat", func(t *testing.T) {
		claude := actions["claude-off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		body := dispatchPolicyClientRequest(t, router, "episode-family-2", "/v1/chat/completions", piChatHistory(
			`"reasoning_content":"claude reasoning",`+
				`"reasoning_details":[{"type":"reasoning.text","text":"claude reasoning","signature":"claude-signature","format":"anthropic-claude-v1","index":0}]`))
		assistant := assistantOf(t, body)
		details := string(assistant["reasoning_details"])
		if !strings.Contains(details, "claude-signature") || !strings.Contains(details, "anthropic-claude-v1") {
			t.Fatalf("Claude's own reasoning_details did not reach it: %s", body["messages"])
		}
	})
	t.Run("kimi's reasoning to an open-weight worker over Chat", func(t *testing.T) {
		off := actions["off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
		body := dispatchPolicyClientRequest(t, router, "episode-family-3", "/v1/chat/completions", piChatHistory(kimiChatReasoning))
		assertJSONField(t, body, "model", `"vendor/off"`)
		if got := string(assistantOf(t, body)["reasoning_content"]); got != `"kimi reasoning"` {
			t.Fatalf("an open-weight worker lost the reasoning_content of its tool history: %s", body["messages"])
		}
	})
	t.Run("a Messages client's unsigned thinking to Claude over Chat", func(t *testing.T) {
		claude := actions["claude-off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		body := dispatchPolicyClientRequest(t, router, "episode-family-4", "/v1/messages",
			`{"model":"auto","max_tokens":4096,"messages":[`+
				`{"role":"user","content":"fix the failing test"},`+
				`{"role":"assistant","content":[{"type":"thinking","thinking":"kimi reasoning","signature":""},{"type":"text","text":"Looking at it."}]},`+
				`{"role":"user","content":"go on"}]}`)
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "kimi reasoning") || strings.Contains(wire, "reasoning_content") {
			t.Fatalf("a Claude worker was sent unsigned thinking: %s", wire)
		}
		if !strings.Contains(wire, "Looking at it.") {
			t.Fatalf("the assistant's visible text was lost: %s", wire)
		}
	})
}

// Golden: the messages a Claude worker dispatched over Chat is sent for a pi
// history that mixes another family's reasoning with Claude's own.
func TestClaudeOverChatReasoningGolden(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	raw, err := os.ReadFile("testdata/claude_over_chat_reasoning_golden.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Request      json.RawMessage `json:"request"`
		WantMessages json.RawMessage `json:"want_messages"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	claude := actions["claude-off"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	body := dispatchPolicyClientRequest(t, router, "episode-family-golden", "/v1/chat/completions", string(golden.Request))
	assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
	var got, want any
	if err := json.Unmarshal(body["messages"], &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(golden.WantMessages, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messages differ from the golden:\n got %s\nwant %s", body["messages"], golden.WantMessages)
	}
}

// The reverse direction, end to end: Claude's reasoning in a history reaches
// a kimi or glm worker over Chat by the disposition table's rule. Signed
// thinking keeps its text as reasoning_content and loses the signature no
// other host can verify, and reasoning_details reach the worker as the client
// sent them. Nothing is counted as foreign.
func TestClaudeReasoningReachesAnOpenWeightWorkerOverChat(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	for _, worker := range []struct{ action, model string }{{"off", "vendor/off"}, {"think", "vendor/think"}} {
		action := actions[worker.action].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
		t.Run(worker.action+": a Messages client's signed thinking", func(t *testing.T) {
			logs := captureLogs(t)
			body := dispatchPolicyClientRequest(t, router, "episode-reverse-messages-"+worker.action, "/v1/messages",
				`{"model":"auto","max_tokens":4096,"messages":[`+
					`{"role":"user","content":"run the tests"},`+
					`{"role":"assistant","content":[{"type":"thinking","thinking":"claude reasoning","signature":"claude-signature"},`+
					`{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"go test"}}]},`+
					`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}`)
			assertJSONField(t, body, "model", `"`+worker.model+`"`)
			wire := string(body["messages"])
			if strings.Contains(wire, "claude-signature") || !strings.Contains(wire, `"reasoning_content":"claude reasoning"`) {
				t.Fatalf("Claude's thinking did not reach the worker as unsigned reasoning_content: %s", wire)
			}
			for _, entry := range logs.All() {
				if entry.ContextMap()["event"] == "reasoning_dropped" {
					t.Fatalf("Claude's reasoning was dropped for a non-Claude worker: %v", entry.ContextMap())
				}
			}
		})
		t.Run(worker.action+": a Chat client's Claude reasoning_details", func(t *testing.T) {
			body := dispatchPolicyClientRequest(t, router, "episode-reverse-chat-"+worker.action, "/v1/chat/completions", piChatHistory(
				`"reasoning_content":"claude reasoning",`+
					`"reasoning_details":[{"type":"reasoning.text","text":"claude reasoning","signature":"claude-signature","format":"anthropic-claude-v1","index":0}]`))
			wire := string(body["messages"])
			if !strings.Contains(wire, `"reasoning_content":"claude reasoning"`) || !strings.Contains(wire, "anthropic-claude-v1") {
				t.Fatalf("Claude's reasoning did not reach the worker as the client sent it: %s", wire)
			}
		})
	}
}

// A Claude worker gets the same history whatever wire format reaches it: on
// Messages too, another model's thinking is dropped and counted as foreign.
func TestForeignReasoningIsCountedForAClaudeWorkerOverMessages(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	claude := actions["claude"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	logs := captureLogs(t)
	body := dispatchPolicyClientRequest(t, router, "episode-family-messages", "/v1/chat/completions", piChatHistory(kimiChatReasoning))
	assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
	if wire := string(body["messages"]); strings.Contains(wire, "kimi reasoning") || strings.Contains(wire, `"thinking"`) {
		t.Fatalf("a Claude worker on Messages was sent another model's reasoning: %s", wire)
	}
	dropped := findLogEvent(t, logs, "reasoning_dropped")
	if fmt.Sprint(dropped["foreign_dropped"]) != "1" || fmt.Sprint(dropped["unsigned_dropped"]) != "0" {
		t.Fatalf("reasoning_dropped = %v, want the drop counted as foreign", dropped)
	}
}

// Cache stability: the bytes a Claude worker is sent depend only on the
// history and the worker. The same turn sent twice is byte-identical, and the
// next turn of the run repeats the previous turn's messages as its prefix.
func TestClaudeOverChatReasoningDropIsCacheStable(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	claude := actions["claude-off"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	turn := piChatHistory(kimiChatReasoning)
	first := dispatchPolicyClientRequest(t, router, "episode-stable-1", "/v1/chat/completions", turn)
	again := dispatchPolicyClientRequest(t, router, "episode-stable-2", "/v1/chat/completions", turn)
	if string(first["messages"]) != string(again["messages"]) {
		t.Fatalf("one history sent twice differs:\n%s\n%s", first["messages"], again["messages"])
	}
	next := strings.Replace(turn, `{"role":"user","content":"Now print 15 numbers."}]`,
		`{"role":"user","content":"Now print 15 numbers."},`+
			`{"role":"assistant","content":"Done.","reasoning_details":[{"type":"reasoning.text","text":"claude reasoning","signature":"claude-signature","format":"anthropic-claude-v1","index":0}]},`+
			`{"role":"user","content":"And 20?"}]`, 1)
	if next == turn {
		t.Fatal("the next turn was not built")
	}
	following := dispatchPolicyClientRequest(t, router, "episode-stable-3", "/v1/chat/completions", next)
	prefix := strings.TrimSuffix(string(first["messages"]), "]")
	if !strings.HasPrefix(string(following["messages"]), prefix+",") {
		t.Fatalf("the next turn does not repeat the previous turn's messages:\nprev %s\nnext %s", first["messages"], following["messages"])
	}
}
