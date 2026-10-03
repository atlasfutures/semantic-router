package protocolcodec

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// reasoningDetailsDrops counts the content.reasoning_details drops recorded
// for one target.
func reasoningDetailsDrops(diagnostics llmprotocol.Diagnostics, target llmprotocol.WireFormat) int {
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "content.reasoning_details" && diagnostic.Action == llmprotocol.DiagnosticDropped &&
			diagnostic.Target == target {
			count++
		}
	}
	return count
}

// Every path that loses reasoning_details records it once, and every path
// that carries them records nothing. reasoning_details are OpenRouter Chat's:
// a Chat target or client gets them back, a Responses client gets them
// minted as encrypted_content, and nothing else can hold them.
func TestReasoningDetailsDropsAreRecordedOnEveryPath(t *testing.T) {
	details := `[{"type":"reasoning.encrypted","id":"call_1","data":"gemini-thought-signature","format":"google-gemini-v1","index":0}]`
	chatTurn := func(reasoning string) []byte {
		return []byte(`{"model":"m","messages":[{"role":"user","content":"weather?"},` +
			`{"role":"assistant","content":null,` + reasoning + `"reasoning_details":` + details + `,` +
			`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},` +
			`{"role":"tool","tool_call_id":"call_1","content":"sunny"}]}`)
	}
	minted := string(mintReasoningDetails(json.RawMessage(details)))
	responsesTurn := []byte(`{"model":"m","store":false,"include":["reasoning.encrypted_content"],"input":[` +
		`{"role":"user","content":[{"type":"input_text","text":"weather?"}]},` +
		`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Check the tool."}],"encrypted_content":` + minted + `},` +
		`{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)

	requests := []struct {
		name   string
		source llmprotocol.WireFormat
		body   []byte
		target llmprotocol.WireFormat
		routed bool
		want   int
	}{
		{"chat text+details to anthropic", llmprotocol.OpenAIChatV1, chatTurn(`"reasoning":"Check the tool.",`), llmprotocol.AnthropicMessagesV1, true, 1},
		{"chat details only to anthropic", llmprotocol.OpenAIChatV1, chatTurn(""), llmprotocol.AnthropicMessagesV1, true, 1},
		{"chat text+details to responses", llmprotocol.OpenAIChatV1, chatTurn(`"reasoning":"Check the tool.",`), llmprotocol.OpenAIResponsesV1, true, 1},
		{"chat details only to responses", llmprotocol.OpenAIChatV1, chatTurn(""), llmprotocol.OpenAIResponsesV1, true, 1},
		{"chat to chat", llmprotocol.OpenAIChatV1, chatTurn(`"reasoning":"Check the tool.",`), llmprotocol.OpenAIChatV1, true, 0},
		{"responses minted to anthropic", llmprotocol.OpenAIResponsesV1, responsesTurn, llmprotocol.AnthropicMessagesV1, true, 1},
		{"responses minted to responses", llmprotocol.OpenAIResponsesV1, responsesTurn, llmprotocol.OpenAIResponsesV1, false, 1},
		{"responses minted to routed responses", llmprotocol.OpenAIResponsesV1, responsesTurn, llmprotocol.OpenAIResponsesV1, true, 1},
		{"responses minted to chat", llmprotocol.OpenAIResponsesV1, responsesTurn, llmprotocol.OpenAIChatV1, true, 0},
	}
	for _, test := range requests {
		t.Run("request/"+test.name, func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(test.source, test.body)
			if err != nil {
				t.Fatal(err)
			}
			if test.routed {
				request.Model = "routed"
				request.Generation++
			}
			result, err := engine.EncodeRequest(test.target, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if got := reasoningDetailsDrops(result.Diagnostics, test.target); got != test.want {
				t.Fatalf("%d reasoning_details drops in %v, want %d", got, diagnosticFields(result.Diagnostics), test.want)
			}
		})
	}

	responses := []struct {
		name   string
		client llmprotocol.WireFormat
		want   int
	}{
		{"anthropic", llmprotocol.AnthropicMessagesV1, 1},
		{"responses", llmprotocol.OpenAIResponsesV1, 0},
		{"chat", llmprotocol.OpenAIChatV1, 0},
	}
	for _, test := range responses {
		t.Run("response/buffered/"+test.name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(
				llmprotocol.OpenAIChatV1, test.client, loadProviderFixture(t, openRouterResponseReasoning), renameResponseModel,
			)
			if err != nil {
				t.Fatal(err)
			}
			if got := reasoningDetailsDrops(result.Diagnostics, test.client); got != test.want {
				t.Fatalf("%d reasoning_details drops in %v, want %d", got, diagnosticFields(result.Diagnostics), test.want)
			}
		})
		t.Run("response/stream/"+test.name, func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, test.client, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, pushed, err := stream.Push(loadProviderFixture(t, openRouterStreamReasoning))
			if err != nil {
				t.Fatal(err)
			}
			_, _, finalized, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			all := append(pushed, finalized...)
			if got := reasoningDetailsDrops(all, test.client); got != test.want {
				t.Fatalf("%d reasoning_details drops in %v, want %d", got, diagnosticFields(all), test.want)
			}
		})
	}
}
