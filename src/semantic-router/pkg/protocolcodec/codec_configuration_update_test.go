package protocolcodec

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func configurationUpdateRequest() llmprotocol.Request {
	text := func(role llmprotocol.Role, value string) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
	}
	return llmprotocol.Request{
		Generation: 1,
		Model:      "anthropic/claude-opus-5.5",
		Messages: []llmprotocol.Message{
			text(llmprotocol.RoleUser, "first"),
			text(llmprotocol.RoleAssistant, "answer"),
			{Role: llmprotocol.RoleSystem, ReasoningEffort: "low"},
			text(llmprotocol.RoleUser, "second"),
		},
	}
}

func TestOpenAIChatEncodesConfigurationUpdateInPlace(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	request := configurationUpdateRequest()
	request.MessageEffortUpdates = true
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatalf("decode wire: %v", err)
	}
	if len(wire.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(wire.Messages))
	}
	// OpenRouter documents exactly this shape, empty content included; the
	// item must stay where the router put it.
	want := `{"role":"system","content":"","configuration_update":{"reasoning":{"effort":"low"}}}`
	if got := string(wire.Messages[2]); got != want {
		t.Fatalf("configuration message = %s, want %s", got, want)
	}
}

// Without the Router's per-dispatch opt-in, a per-message effort is omitted
// and counted on Chat (US-004): a non-OpenRouter backend would not know the
// member.
func TestOpenAIChatOmitsMessageEffortWithoutOptIn(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, configurationUpdateRequest(), llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatalf("decode wire: %v", err)
	}
	if len(wire.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(wire.Messages))
	}
	assertDiagnosticFields(t, result.Diagnostics, "messages[].output_config.effort")
}

// Responses omits a per-message effort with a diagnostic and Messages
// re-encodes it as output_config.effort; neither refuses the turn (US-004).
func TestMessageEffortIsNeverRefused(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	responses, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, configurationUpdateRequest(), llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("Responses encode: %v", err)
	}
	assertDiagnosticFields(t, responses.Diagnostics, "messages[].output_config.effort")
	messages, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, configurationUpdateRequest(), llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("Messages encode: %v", err)
	}
	if !strings.Contains(string(messages.Body), `"output_config":{"effort":"low"}`) {
		t.Fatalf("Messages body lost the per-message effort: %s", messages.Body)
	}
}

func TestMessageEffortMustBeASystemMessageWithAPlainName(t *testing.T) {
	for name, message := range map[string]llmprotocol.Message{
		"user role":  {Role: llmprotocol.RoleUser, ReasoningEffort: "low"},
		"not a name": {Role: llmprotocol.RoleSystem, ReasoningEffort: "Low!"},
	} {
		t.Run(name, func(t *testing.T) {
			request := configurationUpdateRequest()
			request.Messages[2] = message
			err := llmprotocol.ValidateRequest(request, llmprotocol.DefaultPolicy().Limits)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != "invalid_message_reasoning_effort" {
				t.Fatalf("error = %v, want invalid_message_reasoning_effort", err)
			}
		})
	}
	// The lever's vocabulary is wider than Anthropic's: minimal and none pass.
	for _, effort := range []string{"minimal", "none", "xhigh"} {
		request := configurationUpdateRequest()
		request.Messages[2].ReasoningEffort = effort
		if err := llmprotocol.ValidateRequest(request, llmprotocol.DefaultPolicy().Limits); err != nil {
			t.Fatalf("effort %q refused: %v", effort, err)
		}
	}
}

func TestOpenAIChatRefusesClientConfigurationUpdate(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"a"},` +
		`{"role":"system","content":"","configuration_update":{"reasoning":{"effort":"max"}}},` +
		`{"role":"user","content":"b"}]}`)
	_, _, _, err := (OpenAIChatCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_configuration_update" {
		t.Fatalf("error = %v, want unsupported_configuration_update", err)
	}
}
