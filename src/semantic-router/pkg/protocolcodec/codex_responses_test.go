package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The fixtures are Codex CLI 0.154.0 request bodies captured against a local
// stub, trimmed: instructions, message text, tool list and client metadata
// are shortened, and every member Codex sends is kept as it was sent.
func codexRequestFixture(t *testing.T, turn string) map[string]json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile("testdata/codex/" + turn + "-request.json")
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

func codexBody(t *testing.T, body map[string]json.RawMessage) string {
	t.Helper()
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func routeResponsesRequestDiagnostics(t *testing.T, body string, target llmprotocol.WireFormat) ([]byte, llmprotocol.Diagnostics) {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeRequestForMutation() error = %v", err)
	}
	request.Model = "selected-arm"
	request.ReasoningEffort = "medium"
	request.Generation++
	result, err := engine.EncodeRequest(target, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest(%s) error = %v", target, err)
	}
	return result.Body, result.Diagnostics
}

func droppedFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			fields = append(fields, diagnostic.Field)
		}
	}
	return fields
}

// A Codex turn routes to every target format. A Responses arm gets include,
// prompt_cache_key, reasoning.summary, store:false, the web_search and
// namespace tools, and the resent encrypted reasoning item back as Codex sent
// them; another format drops and counts each one instead of refusing the turn.
func TestCodexTurnRoutesToEveryFormat(t *testing.T) {
	body := codexBody(t, codexRequestFixture(t, "turn2"))

	t.Run("responses", func(t *testing.T) {
		routed, _ := routeResponsesRequestDiagnostics(t, body, llmprotocol.OpenAIResponsesV1)
		var wire map[string]json.RawMessage
		if err := json.Unmarshal(routed, &wire); err != nil {
			t.Fatal(err)
		}
		for field, want := range map[string]string{
			"include":          `["reasoning.encrypted_content"]`,
			"prompt_cache_key": `"00000000-0000-7000-8000-00000000c0de"`,
			"store":            `false`,
			"reasoning":        `{"effort":"medium","summary":"auto"}`,
		} {
			if got := string(wire[field]); got != want {
				t.Errorf("%s = %s, want %s", field, got, want)
			}
		}
		for _, tool := range []string{`"type":"web_search"`, `"name":"multi_agent_v1"`, `"name":"exec_command"`} {
			if !bytes.Contains(routed, []byte(tool)) {
				t.Errorf("tool %s did not reach the Responses arm: %s", tool, routed)
			}
		}
		if !bytes.Contains(routed, []byte(`"encrypted_content":"gAAAAB-fake-encrypted-content-for-capture"`)) {
			t.Fatalf("the resent reasoning item lost its encrypted_content: %s", routed)
		}
	})
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			routed, diagnostics := routeResponsesRequestDiagnostics(t, body, target)
			for _, gone := range []string{"encrypted_content", "prompt_cache_key", `"summary"`, "web_search", "multi_agent_v1"} {
				if bytes.Contains(routed, []byte(gone)) {
					t.Fatalf("%s reached %s: %s", gone, target, routed)
				}
			}
			for _, kept := range []string{"echo capture", "call_1", "exec_command"} {
				if !bytes.Contains(routed, []byte(kept)) {
					t.Fatalf("%q was lost routing to %s: %s", kept, target, routed)
				}
			}
			dropped := strings.Join(droppedFields(diagnostics), ",")
			for _, field := range []string{"include", "prompt_cache_key", "reasoning.summary", "content.reasoning", "tools.web_search", "tools.namespace"} {
				if !strings.Contains(dropped, field) {
					t.Errorf("the drop of %s was not counted; dropped: %s", field, dropped)
				}
			}
		})
	}
}

// store:false asks for what a target that keeps nothing already does, so it
// is no reason to refuse an Anthropic arm. store:true still is.
func TestExplicitStoreFalseNeedsNoStorage(t *testing.T) {
	body := `{"model":"m","store":false,"input":"hello"}`
	routed, _ := routeResponsesRequestDiagnostics(t, body, llmprotocol.AnthropicMessagesV1)
	if bytes.Contains(routed, []byte("store")) {
		t.Fatalf("store reached an Anthropic target: %s", routed)
	}
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","store":true,"input":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Generation++
	if _, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope); err == nil {
		t.Fatal("store:true was encoded for a target that cannot store")
	}
}

// A reasoning.summary with nothing to hold it is dropped, not written into a
// reasoning object of its own: that object would turn reasoning back on for an
// arm dispatched without it.
func TestCarriedReasoningSummaryNeedsAReasoningObject(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":"hello","reasoning":{"effort":"high","summary":"auto"}}`))
	if err != nil {
		t.Fatal(err)
	}
	request.ReasoningEffort = ""
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(result.Body, []byte("reasoning")) {
		t.Fatalf("a reasoning object was written for an arm without reasoning: %s", result.Body)
	}
	if dropped := droppedFields(result.Diagnostics); len(dropped) != 1 || dropped[0] != "reasoning.summary" {
		t.Fatalf("dropped = %v, want [reasoning.summary]", dropped)
	}
}
