package protocolcodec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file pins rule 1 of the accept-by-default ingress contract: a member no
// wire struct names never refuses a request, wherever it sits. Ingress still
// refuses malformed JSON and every hard limit; those refusals keep their own
// goldens.
//
// The seven cases below are the read-only sweep of 2026-09-05 against the
// 2.1.260 client. Each one answered 400 on the dev cell before this change,
// and each 400 was a user-visible failed turn: the gateway does not replay a
// body-level 400.

// anthropicSweepRequest wraps one user content block list, tool list and
// top-level extension into the smallest complete Messages body. A block list
// holding a tool_result also gets the assistant tool_use turn it answers,
// because an unpaired result is a shape refusal and not the class under test.
func anthropicSweepRequest(t *testing.T, blocks, tools, extra string) []byte {
	t.Helper()
	body := `{"model":"client-model","max_tokens":64`
	if extra != "" {
		body += "," + extra
	}
	if tools != "" {
		body += `,"tools":` + tools
	}
	body += `,"messages":[`
	if strings.Contains(blocks, `"tool_result"`) {
		body += `{"role":"assistant","content":[{"type":"tool_use","id":"call_1",` +
			`"name":"lookup","input":{}}]},`
	}
	body += `{"role":"user","content":` + blocks + `}]}`
	if !json.Valid([]byte(body)) {
		t.Fatalf("sweep request is not valid JSON: %s", body)
	}
	return []byte(body)
}

func TestIngressAcceptsSweepRequests(t *testing.T) {
	engine := NewBuiltinEngine()
	cases := []struct {
		name   string
		blocks string
		tools  string
		extra  string
	}{
		{
			name:   "cache_control_evict_on_complete",
			blocks: `[{"type":"text","text":"hello","cache_control":{"type":"ephemeral","evict_on_complete":true}}]`,
		},
		{
			name:   "tools_defer_loading",
			blocks: `[{"type":"text","text":"hello"}]`,
			tools:  `[{"name":"lookup","input_schema":{"type":"object"},"defer_loading":true}]`,
		},
		{
			name:   "output_config_task_budget",
			blocks: `[{"type":"text","text":"hello"}]`,
			extra:  `"output_config":{"task_budget":2048}`,
		},
		{
			name:   "document_title",
			blocks: `[{"type":"document","title":"Release notes","source":{"type":"text","media_type":"text/plain","data":"the document body"}}]`,
		},
		{
			name:   "tool_type_web_search",
			blocks: `[{"type":"text","text":"hello"}]`,
			tools:  `[{"type":"web_search_20250305","name":"web_search","input_schema":{"type":"object"}}]`,
		},
		{
			name:   "tool_type_advisor",
			blocks: `[{"type":"text","text":"hello"}]`,
			tools:  `[{"type":"advisor_20260301","name":"advisor","input_schema":{"type":"object"}}]`,
		},
		{
			name: "tool_result_image",
			blocks: `[{"type":"tool_result","tool_use_id":"call_1","content":[` +
				`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk="}}]}]`,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			body := anthropicSweepRequest(t, testCase.blocks, testCase.tools, testCase.extra)
			if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body); err != nil {
				t.Fatalf("ingress refused a sweep request: %v", err)
			}
		})
	}
}

// TestIngressAcceptsUnknownNestedKeyUnderEveryModelledStruct walks every
// modelled struct an Anthropic request body reaches and puts a member no
// contract names inside it. Rule 2 of the old policy refused each of these as
// invalid_json; the class is unbounded, so enumerating the structs is the only
// closure available.
func TestIngressAcceptsUnknownNestedKeyUnderEveryModelledStruct(t *testing.T) {
	engine := NewBuiltinEngine()
	cases := []struct {
		name   string
		blocks string
		tools  string
		extra  string
	}{
		{
			name:   "content_block",
			blocks: `[{"type":"text","text":"hello","future_field":true}]`,
		},
		{
			name:   "cache_control",
			blocks: `[{"type":"text","text":"hello","cache_control":{"type":"ephemeral","future_field":true}}]`,
		},
		{
			name:   "media_source",
			blocks: `[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk=","future_field":true}}]`,
		},
		{
			name:   "tool_result_nested_block",
			blocks: `[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"ok","future_field":true}]}]`,
		},
		{
			name:   "tool",
			blocks: `[{"type":"text","text":"hello"}]`,
			tools:  `[{"name":"lookup","input_schema":{"type":"object"},"future_field":true}]`,
		},
		{
			name:   "output_config",
			blocks: `[{"type":"text","text":"hello"}]`,
			extra:  `"output_config":{"effort":"high","future_field":true}`,
		},
		{
			name:   "thinking",
			blocks: `[{"type":"text","text":"hello"}]`,
			extra:  `"thinking":{"type":"adaptive","future_field":true}`,
		},
		{
			name:   "metadata",
			blocks: `[{"type":"text","text":"hello"}]`,
			extra:  `"metadata":{"user_id":"u1","future_field":true}`,
		},
		{
			name:   "tool_choice",
			blocks: `[{"type":"text","text":"hello"}]`,
			tools:  `[{"name":"lookup","input_schema":{"type":"object"}}]`,
			extra:  `"tool_choice":{"type":"auto","future_field":true}`,
		},
		{
			name:   "request_envelope",
			blocks: `[{"type":"text","text":"hello"}]`,
			extra:  `"future_field":true`,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			body := anthropicSweepRequest(t, testCase.blocks, testCase.tools, testCase.extra)
			if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body); err != nil {
				t.Fatalf("ingress refused an unknown nested member: %v", err)
			}
		})
	}
}

// TestAnthropicTargetCarriesUnknownNestedMembers pins the carry half of rule
// 1. The member the contract does not name has to come back out on a Messages
// target, or accepting it at ingress would only move the loss one hop.
func TestAnthropicTargetCarriesUnknownNestedMembers(t *testing.T) {
	engine := NewBuiltinEngine()
	body := anthropicSweepRequest(t,
		`[{"type":"text","text":"hello","cache_control":{"type":"ephemeral","evict_on_complete":true}}]`,
		`[{"name":"lookup","input_schema":{"type":"object"},"defer_loading":true}]`,
		`"output_config":{"effort":"high","task_budget":2048}`,
	)
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatalf("ingress refused the request: %v", err)
	}
	// A fresh envelope forces a real encode rather than a byte-for-byte replay
	// of the source body, which would pass without carrying anything.
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("Messages target refused the carried request: %v", err)
	}
	for _, member := range []string{"evict_on_complete", "defer_loading", "task_budget"} {
		if !strings.Contains(string(encoded.Body), member) {
			t.Fatalf("Messages target dropped %q: %s", member, encoded.Body)
		}
	}
}

// TestChatTargetCountsUnknownNestedMembers pins the count half. A Chat target
// cannot name these members, so it drops them -- but a silent drop is what
// made the four incidents of this week cost an investigation each.
func TestChatTargetCountsUnknownNestedMembers(t *testing.T) {
	engine := NewBuiltinEngine()
	body := anthropicSweepRequest(t,
		`[{"type":"text","text":"hello","cache_control":{"type":"ephemeral","evict_on_complete":true}}]`,
		`[{"name":"lookup","input_schema":{"type":"object"},"defer_loading":true}]`,
		`"output_config":{"effort":"high","task_budget":2048}`,
	)
	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatalf("Chat target refused the request: %v", err)
	}
	dropped := make(map[string]bool, len(result.Diagnostics))
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range []string{
		"content.cache_control.evict_on_complete",
		"tools.defer_loading",
		"output_config.task_budget",
	} {
		if !dropped[field] {
			t.Fatalf("Chat target dropped %q without counting it: %v", field, diagnosticFields(result.Diagnostics))
		}
	}
}

func diagnosticFields(diagnostics llmprotocol.Diagnostics) []string {
	fields := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		fields = append(fields, fmt.Sprintf("%s(%s)", diagnostic.Field, diagnostic.Action))
	}
	return fields
}
