package protocolcodec

import (
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The two /v1/responses requests Codex CLI 0.156.1 sends for a task with one
// tool call, captured with multi-agent tools, web search, and reasoning
// summaries turned off in Codex. Prompt text is shortened; IDs are placeholders.
const codexToolLoopFixture = "testdata/clients/codex-cli-0.156.1-tool-loop.json"

const codexPromptCacheKey = "00000000-0000-4000-8000-000000000004"

func loadCodexToolLoop(t *testing.T) []json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile(codexToolLoopFixture)
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 {
		t.Fatalf("fixture has %d turns, want 2", len(turns))
	}
	return turns
}

func TestCodexToolLoopDecodesUnderEitherLossyPolicy(t *testing.T) {
	for _, lossy := range []llmprotocol.LossyPolicy{llmprotocol.LossyReject, llmprotocol.LossyAllowWithDiagnostic} {
		policy := llmprotocol.DefaultPolicy()
		policy.LossyFeatures = lossy
		engine, err := NewEngine(NewBuiltinRegistry(), policy)
		if err != nil {
			t.Fatal(err)
		}
		for turn, body := range loadCodexToolLoop(t) {
			_, _, diagnostics, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
			if err != nil {
				t.Fatalf("lossy=%s turn %d: Codex request rejected: %v", lossy, turn+1, err)
			}
			// include is carried to a Responses target and counted where it is
			// dropped, at encode; see codex_responses_test.go.
			assertDroppedFields(t, diagnostics, "client_metadata")
		}
	}
}

// prompt_cache_key and include reach a Responses arm and are dropped and
// counted on Chat and Messages arms (fork #106): they name a Responses cache
// shard and Responses reasoning output. client_metadata reaches no arm.
func TestCodexToolLoopDispatchCarriesCacheKeyAndIncludeOnlyToResponses(t *testing.T) {
	engine := NewBuiltinEngine()
	for turn, body := range loadCodexToolLoop(t) {
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
		if err != nil {
			t.Fatalf("turn %d: %v", turn+1, err)
		}
		// Routing rewrites the model, and the Router clears its own storage
		// controls before dispatch, as materializeResponseObjectContext does.
		request.Model = "routed-model"
		request.Store = nil
		request.Generation++
		for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
			encoded, encodeErr := engine.EncodeRequest(format, request, envelope)
			if encodeErr != nil {
				t.Fatalf("turn %d to %s: %v", turn+1, format, encodeErr)
			}
			var dispatch map[string]json.RawMessage
			if unmarshalErr := json.Unmarshal(encoded.Body, &dispatch); unmarshalErr != nil {
				t.Fatal(unmarshalErr)
			}
			if _, forwarded := dispatch["client_metadata"]; forwarded {
				t.Fatalf("turn %d to %s forwarded client_metadata: %s", turn+1, format, encoded.Body)
			}
			if format == llmprotocol.OpenAIResponsesV1 {
				if string(dispatch["prompt_cache_key"]) != `"`+codexPromptCacheKey+`"` {
					t.Fatalf("turn %d to %s: prompt_cache_key = %s", turn+1, format, dispatch["prompt_cache_key"])
				}
				if _, carried := dispatch["include"]; !carried {
					t.Fatalf("turn %d to %s lost include: %s", turn+1, format, encoded.Body)
				}
				continue
			}
			for _, dropped := range []string{"prompt_cache_key", "include"} {
				if _, forwarded := dispatch[dropped]; forwarded {
					t.Fatalf("turn %d to %s forwarded %s: %s", turn+1, format, dropped, encoded.Body)
				}
			}
			assertDroppedFields(t, encoded.Diagnostics, "prompt_cache_key", "include")
		}
	}
}

func TestCodexToolResultTurnReachesChatAsToolMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, loadCodexToolLoop(t)[1])
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var chat chatRequestWire
	if err := json.Unmarshal(encoded.Body, &chat); err != nil {
		t.Fatal(err)
	}
	last := len(chat.Messages) - 1
	call, result := chat.Messages[last-1], chat.Messages[last]
	if call.Role != "assistant" || len(call.ToolCalls) != 1 || call.ToolCalls[0].ID != "call_1" ||
		call.ToolCalls[0].Function.Name != "exec_command" {
		t.Fatalf("tool call did not reach Chat: %+v", call)
	}
	if result.Role != "tool" || result.ToolCallID != "call_1" {
		t.Fatalf("tool result did not reach Chat: %+v", result)
	}
}

// include is carried, not refused: a value the Router does not model shapes
// what the provider returns, not what the model is asked (decision 36). Only
// a value of the wrong JSON type is refused.
func TestResponsesIncludeIsCarriedNotRefused(t *testing.T) {
	tests := []struct {
		include  string
		category llmprotocol.ErrorCategory
	}{
		{include: `[]`},
		{include: `null`},
		{include: `["reasoning.encrypted_content"]`},
		{include: `["message.output_text.logprobs"]`},
		{include: `["reasoning.encrypted_content","web_search_call.results"]`},
		{include: `"reasoning.encrypted_content"`, category: llmprotocol.ErrorInvalidRequest},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.include, func(t *testing.T) {
			body := []byte(`{"model":"m","input":"hello","include":` + test.include + `}`)
			_, _, diagnostics, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			if test.category != "" {
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(err, &protocolError) || protocolError.Category != test.category {
					t.Fatalf("include %s returned %v, want %s", test.include, err, test.category)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(diagnostics) != 0 {
				t.Fatalf("include %s produced diagnostics %+v", test.include, diagnostics)
			}
		})
	}
}

func assertDroppedFields(t *testing.T, diagnostics llmprotocol.Diagnostics, fields ...string) {
	t.Helper()
	dropped := make(map[string]bool, len(diagnostics))
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range fields {
		if !dropped[field] {
			t.Fatalf("%s was not reported as dropped: %+v", field, diagnostics)
		}
	}
}
