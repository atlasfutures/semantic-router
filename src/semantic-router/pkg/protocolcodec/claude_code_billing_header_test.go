package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code prepends one system block to every request:
//
//	x-anthropic-billing-header: cc_version=2.1.260.ada; cc_entrypoint=sdk-cli; cch=<hash>; cc_prompt_id=<uuid>;
//
// cch changes on every turn and cc_prompt_id with every new prompt. Anthropic
// knows the line and caches around it. Every other host reads it as prompt
// text, so the prompt prefix differs from about token 30 onward and no
// provider prefix cache can match: 8 of 8 completed MiMo turns on the dev cell
// reported zero cached prompt tokens on 2026-09-04, and the same bodies with
// the two fields held constant cached 86-99%.
//
// The line says nothing to a model. A Chat or Responses target drops it and
// counts the drop under system.x-anthropic-billing-header; an Anthropic arm
// carries it, because Anthropic is the one host that reads it as metadata.
const (
	claudeCodeBillingHeaderTurn1 = "x-anthropic-billing-header: cc_version=2.1.260.ada; " +
		"cc_entrypoint=sdk-cli; cch=1f3a9c2e; cc_prompt_id=7d1c2f6e-3b4a-4c5d-9e8f-0a1b2c3d4e5f;"
	claudeCodeBillingHeaderTurn2 = "x-anthropic-billing-header: cc_version=2.1.260.ada; " +
		"cc_entrypoint=sdk-cli; cch=9b04e7d1; cc_prompt_id=7d1c2f6e-3b4a-4c5d-9e8f-0a1b2c3d4e5f;"
)

func claudeCodeBillingHeaderBody(header string) []byte {
	return []byte(`{` +
		`"model":"rayline-router","max_tokens":64,` +
		`"system":[` +
		`{"type":"text","text":` + jsonString(header) + `},` +
		`{"type":"text","text":"You are a Claude agent."}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],` +
		`"stream":true}`)
}

func jsonString(text string) string {
	encoded, _ := json.Marshal(text)
	return string(encoded)
}

// The block is absent from the Chat body, the rest of the system prompt
// survives, and the drop is counted.
func TestBillingHeaderIsDroppedAndCountedForChat(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		claudeCodeBillingHeaderBody(claudeCodeBillingHeaderTurn1), nil,
	)
	if err != nil {
		t.Fatalf("a Claude Code turn carrying the billing header was refused for a Chat arm: %v", err)
	}
	if bytes.Contains(result.Body, []byte(billingAttributionPrefix)) {
		t.Fatalf("the Chat body still carries the billing header: %s", result.Body)
	}
	if !bytes.Contains(result.Body, []byte("You are a Claude agent.")) {
		t.Fatalf("the Chat body lost the system prompt beside the header: %s", result.Body)
	}
	requireDroppedField(t, result.Diagnostics, fieldSystemBillingAttribution)
}

// The Responses encoder walks the same instruction list and gets the same
// treatment, so a Responses arm that carries Claude Code traffic caches too.
func TestBillingHeaderIsDroppedAndCountedForResponses(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
		claudeCodeBillingHeaderBody(claudeCodeBillingHeaderTurn1), nil,
	)
	if err != nil {
		t.Fatalf("a Claude Code turn carrying the billing header was refused for a Responses arm: %v", err)
	}
	if bytes.Contains(result.Body, []byte(billingAttributionPrefix)) {
		t.Fatalf("the Responses body still carries the billing header: %s", result.Body)
	}
	if !bytes.Contains(result.Body, []byte("You are a Claude agent.")) {
		t.Fatalf("the Responses body lost the system prompt beside the header: %s", result.Body)
	}
	requireDroppedField(t, result.Diagnostics, fieldSystemBillingAttribution)
}

// The point of the drop: two turns that differ only in the per-turn fields
// encode to the same Chat body, so a provider prefix cache can match.
func TestBillingHeaderTurnsEncodeToOneChatPrefix(t *testing.T) {
	engine := NewBuiltinEngine()
	var bodies [][]byte
	for _, header := range []string{claudeCodeBillingHeaderTurn1, claudeCodeBillingHeaderTurn2} {
		result, err := engine.TranslateRequest(
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
			claudeCodeBillingHeaderBody(header), nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, result.Body)
	}
	if !bytes.Equal(bodies[0], bodies[1]) {
		t.Fatalf("two turns differing only in cch encode to different Chat bodies:\n%s\n%s", bodies[0], bodies[1])
	}
}

// Anthropic reads the line as metadata and caches around it, so routing to an
// Anthropic arm must not erase it.
func TestBillingHeaderReachesAnAnthropicArm(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		claudeCodeBillingHeaderBody(claudeCodeBillingHeaderTurn1), func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(result.Body, []byte(claudeCodeBillingHeaderTurn1)) {
		t.Fatalf("routing to an Anthropic arm erased the billing header: %s", result.Body)
	}
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Field == fieldSystemBillingAttribution {
			t.Fatalf("an Anthropic arm counted a drop it did not make: %+v", diagnostic)
		}
	}
}

// A system block that opens with the prefix and is not the line travels
// whole. The line is exactly one line of "key=value;" fields carrying
// cc_version; prose after the prefix, a second line, or fields without the
// version are prompt text that happens to start that way.
func TestSystemTextThatMerelyStartsLikeTheHeaderTravelsToChat(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, prompt := range map[string]string{
		"second line":      billingAttributionPrefix + " cc_version=2.1.260.ada;\nYou are a Claude agent.",
		"prose":            billingAttributionPrefix + " retain this instruction. You are a Claude agent.",
		"no version field": billingAttributionPrefix + " cc_entrypoint=sdk-cli; note=You are a Claude agent.;",
	} {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"model":"rayline-router","max_tokens":64,` +
				`"system":[{"type":"text","text":` + jsonString(prompt) + `}],` +
				`"messages":[{"role":"user","content":"hello"}]}`)
			result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(result.Body, []byte("You are a Claude agent.")) {
				t.Fatalf("a system block that is not the line was dropped as the billing header: %s", result.Body)
			}
			for _, diagnostic := range result.Diagnostics {
				if diagnostic.Field == fieldSystemBillingAttribution {
					t.Fatalf("prompt text was counted as the billing header: %+v", diagnostic)
				}
			}
		})
	}
}

// A system list holding only the line encodes to no system message at all,
// rather than an empty one the provider refuses.
func TestBillingHeaderAloneLeavesNoSystemMessage(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"rayline-router","max_tokens":64,` +
		`"system":[{"type":"text","text":` + jsonString(claudeCodeBillingHeaderTurn1) + `}],` +
		`"messages":[{"role":"user","content":"hello"}]}`)
	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	for _, message := range wire.Messages {
		if message.Role == "system" {
			t.Fatalf("an instruction holding only the billing header still produced a system message: %s", result.Body)
		}
	}
	requireDroppedField(t, result.Diagnostics, fieldSystemBillingAttribution)
}

// The captured turn carries the header with the per-turn fields, and the Chat
// leg strips it. This is the shape the dev cell forwarded verbatim, which is
// what the goldens missed while the fixture lacked cch and cc_prompt_id.
func TestClaudeCodeToolTurnCaptureLosesTheBillingHeaderOnChat(t *testing.T) {
	engine := NewBuiltinEngine()
	body, err := os.ReadFile(filepath.Join("testdata", "claude-code", "tool-turn-request.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte("cch=")) || !bytes.Contains(body, []byte("cc_prompt_id=")) {
		t.Fatal("the capture no longer carries the per-turn billing fields, so this test proves nothing")
	}
	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(result.Body, []byte(billingAttributionPrefix)) {
		t.Fatalf("the Chat body still carries the billing header: %s", result.Body)
	}
	requireDroppedField(t, result.Diagnostics, fieldSystemBillingAttribution)
}
