package execution

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
)

func TestRequestPrefixIgnoresMovedCacheMetadataWithoutChangingTheRequest(t *testing.T) {
	directive := &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"}
	initial := llmprotocol.Request{
		AutoCache:    directive,
		Instructions: []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "system", Cache: directive}}}},
		Tools:        []llmprotocol.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object","properties":{"cache_control":{"type":"string"}}}`), Cache: directive}},
		Messages:     []llmprotocol.Message{{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "result", Cache: directive}}}}}}},
	}
	before, err := json.Marshal(initial)
	if err != nil {
		t.Fatal(err)
	}
	prefix, err := requestPrefix(initial)
	if err != nil {
		t.Fatal(err)
	}
	after, err := json.Marshal(initial)
	if err != nil || string(before) != string(after) {
		t.Fatal("prefix computation changed the provider request")
	}
	var next llmprotocol.Request
	if err := json.Unmarshal(before, &next); err != nil {
		t.Fatal(err)
	}
	next.AutoCache = nil
	next.Instructions[0].Content[0].Cache = nil
	next.Tools[0].Cache = nil
	next.Messages[0].Content[0].ToolResult.Content[0].Cache = nil
	next.Messages = append(next.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue", Cache: directive}}})
	moved, err := requestPrefix(next)
	if err != nil || !extendsPrefix(prefix, moved) {
		t.Fatalf("breakpoint move erased the previous prefix: %v", err)
	}
	next.Instructions[0].Content[0].Text = "different system"
	changed, _ := requestPrefix(next)
	if extendsPrefix(prefix, changed) {
		t.Fatal("system rewrite retained the previous epoch")
	}
	next.Instructions[0].Content[0].Text = "system"
	next.Tools[0].InputSchema = json.RawMessage(`{"type":"object","properties":{"cache_control":{"type":"number"}}}`)
	changed, _ = requestPrefix(next)
	if extendsPrefix(prefix, changed) {
		t.Fatal("semantic schema data named cache_control was erased from comparison")
	}
}

func TestDispatchCacheTTLRequiresAnthropicModel(t *testing.T) {
	body := []byte(`{"cache_control":{"type":"ephemeral","ttl":"1h"},"messages":[]}`)
	for _, deployment := range []Deployment{
		{Kind: "local", RegistryModel: "qwen3.8-27b", Protocol: llmprotocol.AnthropicMessagesV1},
		{Kind: "cloud", RegistryModel: "gpt-5.4", Protocol: llmprotocol.AnthropicMessagesV1},
	} {
		if ttl, source := dispatchedCacheTTL(deployment, body); ttl != 0 || source != "" {
			t.Fatalf("non-Anthropic wire implied a provider cache lifetime: %d %s", ttl, source)
		}
	}
	if ttl, source := dispatchedCacheTTL(Deployment{RegistryModel: "claude-opus-4-6"}, body); ttl != 3600 || source != "anthropic cache_control ephemeral 1h" {
		t.Fatalf("Anthropic evidence lost: %d %s", ttl, source)
	}
}

func TestClaudeSignedHistorySurvivesInEachProtocol(t *testing.T) {
	cases := []struct {
		name           string
		protocol       llmprotocol.WireFormat
		body, expected string
	}{
		{"messages", llmprotocol.AnthropicMessagesV1,
			`{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":[{"type":"thinking","thinking":"consider","signature":"sig-claude"},{"type":"text","text":"answer"}]},{"role":"user","content":"next"}]}`,
			`"type":"thinking"`},
		{"chat", llmprotocol.OpenAIChatV1,
			`{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer","reasoning":"consider","reasoning_details":[{"type":"reasoning.text","text":"consider","signature":"sig-claude","format":"anthropic-claude-v1","index":0}]},{"role":"user","content":"next"}]}`,
			`"format":"anthropic-claude-v1"`},
		{"responses", llmprotocol.OpenAIResponsesV1,
			`{"model":"claude","input":[{"role":"user","content":"question"},{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"consider"}],"signature":"sig-claude","format":"anthropic-claude-v1"},{"role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"role":"user","content":"next"}]}`,
			`"format":"anthropic-claude-v1"`},
	}
	engine := protocolcodec.NewBuiltinEngine()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request, envelope, _, err := engine.DecodeRequestForMutation(tc.protocol, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			prepareReasoning(&request, Deployment{Kind: "cloud", ID: "claude", RegistryModel: "claude-opus-4-6", Protocol: tc.protocol}, &branchState{})
			request.Generation++
			encoded, err := engine.EncodeRequest(tc.protocol, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(encoded.Body), tc.expected) || !strings.Contains(string(encoded.Body), `"signature":"sig-claude"`) {
				t.Fatalf("own-model signed reasoning was lost: %s", encoded.Body)
			}
		})
	}
}

// The reasoning items a Responses client got for Claude's turn, resent as
// codex resends them (type, summary, content and the minted
// encrypted_content), reach the next Claude worker over Responses with the
// signature and format OpenRouter's /responses takes back, and no minted blob.
func TestClaudeReasoningResentByCodexReachesAResponsesWorkerSigned(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	answer := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-6",` +
		`"content":[{"type":"thinking","thinking":"consider","signature":"sig-claude"},{"type":"redacted_thinking","data":"opaque-blob"},{"type":"text","text":"answer"}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":5}}`
	translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, []byte(answer), nil)
	if err != nil {
		t.Fatal(err)
	}
	var response struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(translated.Body, &response); err != nil {
		t.Fatal(err)
	}
	items := []json.RawMessage{json.RawMessage(`{"role":"user","content":"question"}`)}
	for _, item := range response.Output {
		if string(item["type"]) == `"reasoning"` {
			resent := map[string]json.RawMessage{}
			for _, name := range []string{"type", "id", "summary", "content", "encrypted_content"} {
				if value, present := item[name]; present {
					resent[name] = value
				}
			}
			item = resent
		}
		raw, _ := json.Marshal(item)
		items = append(items, raw)
	}
	items = append(items, json.RawMessage(`{"role":"user","content":"next"}`))
	input, _ := json.Marshal(items)
	body := `{"model":"claude","store":false,"include":["reasoning.encrypted_content"],"input":` + string(input) + `}`
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	prepareReasoning(&request, Deployment{Kind: "cloud", ID: "claude", RegistryModel: "claude-opus-4-6", Protocol: llmprotocol.OpenAIResponsesV1}, &branchState{})
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	worker := string(encoded.Body)
	if !strings.Contains(worker, `"signature":"sig-claude"`) || !strings.Contains(worker, `"format":"anthropic-claude-v1"`) ||
		!strings.Contains(worker, `"encrypted_content":"opaque-blob"`) || strings.Contains(worker, "vsr.reasoning_details") {
		t.Fatalf("the resent turn did not reach the Claude worker signed: %s", worker)
	}
}

func TestEncryptedIssuerIsStagedBeforeHistoryMutation(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	body := []byte(`{"model":"auto","input":[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"encrypted-blob"},{"role":"user","content":"next"}]}`)
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	previous := []string{"aaaaaaaaaaaaaaaa"}
	state := &branchState{issuers: append([]string(nil), previous...)}
	deployment := Deployment{Kind: "cloud", ID: "responses", URL: "https://provider.example/v1/responses", WireModel: "gpt", IssuerSHA256: "approved-profile", Protocol: llmprotocol.OpenAIResponsesV1}
	staged, _ := prepareReasoning(&request, deployment, state)
	if !reflect.DeepEqual(state.issuers, previous) || len(staged) != 2 || staged[0] != previous[0] || staged[1] != encryptedDeploymentIssuer(deployment) {
		t.Fatalf("staging changed committed history or lost held provenance: old=%v next=%v", state.issuers, staged)
	}
	request.Messages = nil // later encoding/filtering must not change the staged commit.
	if len(staged) != 2 {
		t.Fatal("mutating prepared history changed staged issuers")
	}
	deployment.RegistryProvider = "openrouter"
	if encryptedDeploymentIssuer(deployment) != reasoning.ReasoningIssuerUnknown {
		t.Fatal("OpenRouter was allowed to replay opaque account-scoped Responses blobs")
	}
}

func TestAttributionUsesExactOriginalWireIndexAndAssistantRole(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user"},{"role":"assistant"},{"role":"user","content":[{"type":"tool_result","tool_use_id":"one"},{"type":"tool_result","tool_use_id":"two"}]}]}`)
	roles, err := wireMessageRoles(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	state := &branchState{}
	state.recordAttribution(llmprotocol.AnthropicMessagesV1, roles, Decision{ActionID: "action", Worker: "local", RouteID: "route", Result: "custom"})
	if got := state.evidenceForRequest("instance", llmprotocol.AnthropicMessagesV1, roles).Attribution; len(got) != 0 {
		t.Fatal("future assistant was attributed before it existed")
	}
	grown := append(append([]string(nil), roles...), "assistant", "user")
	got := state.evidenceForRequest("instance", llmprotocol.AnthropicMessagesV1, grown).Attribution
	if len(got) != 1 || got[0].Message != 3 || got[0].ActionID != "action" {
		t.Fatalf("wire grouping changed attribution index: %+v", got)
	}
	grown[3] = "user"
	grown[4] = "assistant"
	if len(state.evidenceForRequest("instance", llmprotocol.AnthropicMessagesV1, grown).Attribution) != 0 {
		t.Fatal("a later assistant was guessed to belong to this execution")
	}
	if len(state.evidenceForRequest("instance", llmprotocol.OpenAIResponsesV1, []string{"user", "assistant", "user", "assistant"}).Attribution) != 0 {
		t.Fatal("Responses input indices were guessed")
	}
	state.resetEpoch(false)
	if len(state.ledger) != 0 {
		t.Fatal("compaction retained attribution from the previous epoch")
	}
}
