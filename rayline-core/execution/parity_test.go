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

func TestClaudeSignedHistorySurvivesBothProtocolDirections(t *testing.T) {
	cases := []struct {
		name           string
		source, target llmprotocol.WireFormat
		body, expected string
	}{
		{"messages to chat", llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
			`{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":[{"type":"thinking","thinking":"consider","signature":"sig-claude"},{"type":"text","text":"answer"}]},{"role":"user","content":"next"}]}`,
			`"format":"anthropic-claude-v1"`},
		{"chat to messages", llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
			`{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer","reasoning":"consider","reasoning_details":[{"type":"reasoning.text","text":"consider","signature":"sig-claude","format":"anthropic-claude-v1","index":0}]},{"role":"user","content":"next"}]}`,
			`"type":"thinking"`},
	}
	engine := protocolcodec.NewBuiltinEngine()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request, envelope, _, err := engine.DecodeRequestForMutation(tc.source, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			deployment := Deployment{Kind: "cloud", ID: "claude", RegistryModel: "claude-opus-4-6", Protocol: tc.target}
			state := &branchState{provenance: reasoning.WithReasoningProvenance(nil, opaqueDeploymentIssuer(deployment), []string{"sig-claude"})}
			prepareReasoning(&request, tc.source, deployment, state)
			request.Generation++
			encoded, err := engine.EncodeRequest(tc.target, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(encoded.Body), tc.expected) || !strings.Contains(string(encoded.Body), `"signature":"sig-claude"`) {
				t.Fatalf("own-model signed reasoning was lost: %s", encoded.Body)
			}
		})
	}
}

func TestRestoredOpaqueChatReasoningIsFilteredAfterRestoration(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	body := []byte(`{"model":"claude","max_tokens":1024,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer","reasoning_details":[{"type":"reasoning.encrypted","data":"foreign-blob","format":"anthropic-claude-v1","index":0}]},{"role":"user","content":"next"}]}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	state := &branchState{provenance: reasoning.WithReasoningProvenance(nil, "another-issuer", []string{"foreign-blob"})}
	prepareReasoning(&request, envelope.Format, Deployment{Kind: "cloud", ID: "claude", RegistryModel: "claude-opus-4-6", Protocol: llmprotocol.AnthropicMessagesV1}, state)
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded.Body), "foreign-blob") || !strings.Contains(string(encoded.Body), "answer") {
		t.Fatalf("restored foreign reasoning was replayed or visible text removed: %s", encoded.Body)
	}
}

func TestEncryptedIssuerIsStagedBeforeHistoryMutation(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	body := []byte(`{"model":"auto","input":[{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"encrypted-blob"},{"role":"user","content":"next"}]}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	previous := []string{"aaaaaaaaaaaaaaaa"}
	state := &branchState{issuers: append([]string(nil), previous...)}
	deployment := Deployment{Kind: "cloud", ID: "responses", URL: "https://provider.example/v1/responses", WireModel: "gpt", IssuerSHA256: "approved-profile", Protocol: llmprotocol.OpenAIResponsesV1}
	staged, _ := prepareReasoning(&request, envelope.Format, deployment, state)
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
