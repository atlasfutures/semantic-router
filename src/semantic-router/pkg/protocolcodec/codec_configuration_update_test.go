package protocolcodec

import (
	"encoding/json"
	"errors"
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
			{Role: llmprotocol.RoleSystem, Configuration: &llmprotocol.ConfigurationUpdate{ReasoningEffort: "low"}},
			text(llmprotocol.RoleUser, "second"),
		},
	}
}

func TestOpenAIChatEncodesConfigurationUpdateInPlace(t *testing.T) {
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

func TestConfigurationUpdateIsRefusedWhereNotEncodable(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatalf("engine: %v", err)
			}
			_, err = engine.EncodeRequest(format, configurationUpdateRequest(), llmprotocol.Envelope{})
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_configuration_update" {
				t.Fatalf("error = %v, want unsupported_configuration_update", err)
			}
		})
	}
}

func TestConfigurationUpdateMessageMustBeContentlessSystem(t *testing.T) {
	for name, message := range map[string]llmprotocol.Message{
		"user role": {Role: llmprotocol.RoleUser, Configuration: &llmprotocol.ConfigurationUpdate{ReasoningEffort: "low"}},
		"with content": {Role: llmprotocol.RoleSystem,
			Content:       []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "x"}},
			Configuration: &llmprotocol.ConfigurationUpdate{ReasoningEffort: "low"}},
		"no effort": {Role: llmprotocol.RoleSystem, Configuration: &llmprotocol.ConfigurationUpdate{}},
	} {
		t.Run(name, func(t *testing.T) {
			request := configurationUpdateRequest()
			request.Messages[2] = message
			err := llmprotocol.ValidateRequest(request, llmprotocol.DefaultPolicy().Limits)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != "invalid_configuration_message" {
				t.Fatalf("error = %v, want invalid_configuration_message", err)
			}
		})
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
