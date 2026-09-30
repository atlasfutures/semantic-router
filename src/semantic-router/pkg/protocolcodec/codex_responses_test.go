package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
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
// prompt_cache_key, reasoning.summary and store:false back as Codex sent them;
// another format drops and counts each one instead of refusing the turn. The
// multi_agent_v1 namespace tools reach every target: Responses gets the
// namespace tool back, and Chat and Messages call each function by its
// qualified name. The web_search tool and the resent encrypted reasoning item
// are dropped on every target.
func TestCodexTurnRoutesToEveryFormat(t *testing.T) {
	body := codexBody(t, codexRequestFixture(t, "turn2"))

	t.Run("responses", func(t *testing.T) {
		routed, diagnostics := routeResponsesRequestDiagnostics(t, body, llmprotocol.OpenAIResponsesV1)
		dropped := strings.Join(droppedFields(diagnostics), ",")
		for _, field := range []string{"tools.web_search"} {
			if !strings.Contains(dropped, field) {
				t.Errorf("the %s drop was not counted; dropped: %s", field, dropped)
			}
		}
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
		if !bytes.Contains(routed, []byte(`"name":"exec_command"`)) {
			t.Errorf("the function tool did not reach the Responses arm: %s", routed)
		}
		if bytes.Contains(routed, []byte("web_search")) {
			t.Errorf("web_search reached the Responses arm: %s", routed)
		}
		if !bytes.Contains(routed, []byte(`{"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[{"type":"function","name":"close_agent"`)) {
			t.Errorf("the namespace tool did not reach the Responses arm whole: %s", routed)
		}
		if bytes.Contains(routed, []byte("gAAAAB")) || bytes.Contains(routed, []byte(`"rs_1"`)) {
			t.Fatalf("the resent encrypted reasoning item reached the Responses arm: %s", routed)
		}
		if !strings.Contains(dropped, "content.reasoning") {
			t.Errorf("the encrypted reasoning drop was not counted; dropped: %s", dropped)
		}
	})
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			routed, diagnostics := routeResponsesRequestDiagnostics(t, body, target)
			for _, gone := range []string{"encrypted_content", "prompt_cache_key", `"summary"`, "web_search", `"namespace"`} {
				if bytes.Contains(routed, []byte(gone)) {
					t.Fatalf("%s reached %s: %s", gone, target, routed)
				}
			}
			for _, kept := range []string{"echo capture", "call_1", "exec_command", `"multi_agent_v1__close_agent"`} {
				if !bytes.Contains(routed, []byte(kept)) {
					t.Fatalf("%q was lost routing to %s: %s", kept, target, routed)
				}
			}
			dropped := strings.Join(droppedFields(diagnostics), ",")
			for _, field := range []string{"include", "prompt_cache_key", "reasoning.summary", "content.reasoning", "tools.web_search"} {
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

// A carried tool is a declared tool: it satisfies a required choice and counts
// toward the tool limit.
func TestCarriedToolsCountAsDeclaredTools(t *testing.T) {
	engine := NewBuiltinEngine()
	required := `{"model":"m","input":"hello","tool_choice":"required","tools":[{"type":"web_search"}]}`
	if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(required)); err != nil {
		t.Fatalf("a required choice over a carried tool was refused: %v", err)
	}
	policy := llmprotocol.DefaultPolicy()
	tools := make([]string, policy.Limits.Tools+1)
	for index := range tools {
		tools[index] = `{"type":"web_search"}`
	}
	overLimit := `{"model":"m","input":"hello","tools":[` + strings.Join(tools, ",") + `]}`
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(overLimit))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "tools_limit" {
		t.Fatalf("carried tools past the limit returned %v, want tools_limit", err)
	}
}

// A reasoning item that carries encrypted_content is still refused when it
// also names a member the codec refuses on purpose.
func TestEncryptedReasoningStillRefusesARefusedMember(t *testing.T) {
	body := `{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"blob",` +
		`"prompt_cache_breakpoint":{"mode":"explicit"}}]}`
	if _, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body)); err == nil {
		t.Fatal("a refused member rode the encrypted-reasoning carrier")
	}
}

// A choice over tools that were all dropped is dropped with them, on Messages
// as on Chat and Responses, rather than sent to a provider that refuses it.
func TestAnthropicDropsAChoiceOverDroppedTools(t *testing.T) {
	body := `{"model":"m","input":"hello","tool_choice":"required","tools":[{"type":"web_search"}]}`
	routed, diagnostics := routeResponsesRequestDiagnostics(t, body, llmprotocol.AnthropicMessagesV1)
	if bytes.Contains(routed, []byte("tool_choice")) {
		t.Fatalf("a choice with no tools reached Messages: %s", routed)
	}
	if dropped := strings.Join(droppedFields(diagnostics), ","); !strings.Contains(dropped, "tool_choice") {
		t.Fatalf("the choice drop was not counted; dropped: %s", dropped)
	}
}

// An encrypted reasoning item is still checked against its variant before it
// is carried: a member of another item kind is refused, not forwarded.
func TestEncryptedReasoningStillChecksItsVariant(t *testing.T) {
	body := `{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"blob","call_id":"call_1"}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "invalid_input_item_variant" {
		t.Fatalf("a cross-variant member returned %v, want invalid_input_item_variant", err)
	}
}

// A summary-only reasoning object is the client's own request for the default
// effort with a summary; routing it keeps the summary.
func TestSummaryOnlyReasoningKeepsItsSummary(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":"hello","reasoning":{"summary":"auto"}}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "selected-arm"
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(result.Body, []byte(`"reasoning":{"summary":"auto"}`)) {
		t.Fatalf("the summary-only reasoning object was lost: %s", result.Body)
	}
}

// A request that routes unchanged is normally replayed byte for byte, but not
// when it declares a carried tool: the client bytes would hand the provider
// the web_search tool every target drops.
func TestCarriedToolsDisableReplay(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","input":"hello","tools":[{"type":"web_search"}]}`
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(result.Body, []byte("web_search")) {
		t.Fatalf("an unchanged request replayed its carried tools: %s", result.Body)
	}
}

// A request that routes unchanged is not replayed when it resends an encrypted
// reasoning item: the client bytes would hand the provider a blob no target
// is sent.
func TestEncryptedReasoningDisablesReplay(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","input":[{"type":"message","role":"user","content":"hi"},` +
		`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"blob"}]}`
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(result.Body, []byte("blob")) {
		t.Fatalf("an unchanged request replayed its encrypted reasoning: %s", result.Body)
	}
}

// The carried client members are checked against the shapes the Responses
// API defines before they are accepted.
func TestMalformedCarriedMembersAreRefused(t *testing.T) {
	for field, body := range map[string]string{
		"include":           `{"model":"m","input":"hi","include":{}}`,
		"prompt_cache_key":  `{"model":"m","input":"hi","prompt_cache_key":[]}`,
		"reasoning.summary": `{"model":"m","input":"hi","reasoning":{"effort":"low","summary":{}}}`,
	} {
		t.Run(field, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest {
				t.Fatalf("malformed %s returned %v, want invalid_request", field, err)
			}
		})
	}
}

// A resent encrypted reasoning item is dropped on Messages without leaving an
// empty message behind: Anthropic refuses a message with no content.
func TestDroppedCarriedItemLeavesNoEmptyAnthropicMessage(t *testing.T) {
	body := codexBody(t, codexRequestFixture(t, "turn2"))
	routed, _ := routeResponsesRequestDiagnostics(t, body, llmprotocol.AnthropicMessagesV1)
	var wire struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(routed, &wire); err != nil {
		t.Fatal(err)
	}
	for index, message := range wire.Messages {
		if string(message.Content) == "[]" {
			t.Fatalf("message %d reached Messages with empty content: %s", index, routed)
		}
	}
}

// A namespaced call round-trips on every target. In history, Responses gets
// the call back with its namespace and Chat and Messages get it under the
// qualified name the tool was declared by. A call a Chat or Messages provider
// returns under that name gets its namespace back from the request's own
// tools, and a name the request did not declare is never split.
func TestNamespacedCallsRoundTrip(t *testing.T) {
	body := `{"model":"m","input":[` +
		`{"type":"message","role":"user","content":"spawn a helper"},` +
		`{"type":"function_call","id":"fc_1","call_id":"call_1","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"agent-7"}],` +
		`"tools":[{"type":"namespace","name":"multi_agent_v1","description":"Sub-agents.","tools":[` +
		`{"type":"function","name":"spawn_agent","parameters":{"type":"object"}}]}]}`
	responses, _ := routeResponsesRequestDiagnostics(t, body, llmprotocol.OpenAIResponsesV1)
	if !bytes.Contains(responses, []byte(`"name":"spawn_agent","namespace":"multi_agent_v1"`)) {
		t.Fatalf("the call lost its namespace on Responses: %s", responses)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		routed, _ := routeResponsesRequestDiagnostics(t, body, target)
		if bytes.Count(routed, []byte(`"multi_agent_v1__spawn_agent"`)) != 2 {
			t.Fatalf("%s did not get the qualified name on both the tool and the call: %s", target, routed)
		}
	}

	request, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	namespaces := llmprotocol.ToolNamespaces(request.Tools)
	returned := &llmprotocol.ToolCall{ID: "call_2", Name: "multi_agent_v1__spawn_agent", Arguments: "{}"}
	if !llmprotocol.RestoreToolNamespace(returned, namespaces) || returned.Namespace != "multi_agent_v1" || returned.Name != "spawn_agent" {
		t.Fatalf("restored call = %+v", returned)
	}
	undeclared := &llmprotocol.ToolCall{Name: "other__tool"}
	if llmprotocol.RestoreToolNamespace(undeclared, namespaces) || undeclared.Name != "other__tool" {
		t.Fatalf("an undeclared name was split: %+v", undeclared)
	}
}

// A namespace's name and description are bounded like the tool's own: the
// qualified name is what a format without namespaces is sent.
func TestNamespaceMetadataCountsTowardToolLimits(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	long := strings.Repeat("n", policy.Limits.ToolNameBytes)
	body := `{"model":"m","input":"hi","tools":[{"type":"namespace","name":"` + long + `","tools":[` +
		`{"type":"function","name":"f","parameters":{"type":"object"}}]}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "tool_text_limit" {
		t.Fatalf("an oversized qualified name returned %v, want tool_text_limit", err)
	}
}

// A buffered namespaced call replayed as a Responses stream names its
// namespace from the first event on.
func TestSyntheticStreamStartsANamespacedCallWithItsNamespace(t *testing.T) {
	events, err := neutralToolCallEvents(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted}, llmprotocol.Event{},
		llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c", Name: "spawn_agent", Namespace: "multi_agent_v1"}})
	if err != nil {
		t.Fatal(err)
	}
	if events[0].ToolCall == nil || events[0].ToolCall.Namespace != "multi_agent_v1" {
		t.Fatalf("the start event lost the namespace: %+v", events[0].ToolCall)
	}
}

// Where the format has no namespaces, a function is declared with its
// namespace's description ahead of its own, which may be the only place a
// generic function says what it acts on.
func TestFlattenedToolsKeepTheNamespaceDescription(t *testing.T) {
	body := `{"model":"m","input":"hi","tools":[{"type":"namespace","name":"files","description":"Open file handles.",` +
		`"tools":[{"type":"function","name":"close","description":"Close one.","parameters":{"type":"object"}}]}]}`
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		routed, _ := routeResponsesRequestDiagnostics(t, body, target)
		if !bytes.Contains(routed, []byte(`"description":"Open file handles.\n\nClose one."`)) {
			t.Fatalf("%s lost the namespace description: %s", target, routed)
		}
	}
}

// A call in history is bounded by its qualified name, the name a format
// without namespaces is sent.
func TestHistoricalNamespacedCallCountsTowardTheNameLimit(t *testing.T) {
	long := strings.Repeat("n", llmprotocol.DefaultPolicy().Limits.ToolNameBytes)
	body := `{"model":"m","input":[{"type":"message","role":"user","content":"hi"},` +
		`{"type":"function_call","call_id":"c","namespace":"` + long + `","name":"f","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c","output":"ok"}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "tool_call_limit" {
		t.Fatalf("an oversized qualified call name returned %v, want tool_call_limit", err)
	}
}
