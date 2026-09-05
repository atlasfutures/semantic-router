package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Rule 2 of the accept-by-default design: one disposition table per target
// format, in place of a branch per field. These tests pin the table's three
// actions against real request shapes.

const documentBody = "The handbook says ten."

func anthropicDispositionRequest(t *testing.T) []byte {
	t.Helper()
	return anthropicSweepRequest(t,
		`[{"type":"document","title":"Release notes","source":{"type":"text",`+
			`"media_type":"text/plain","data":"`+documentBody+`"}},`+
			`{"type":"text","text":"summarise it","citations":[]}]`,
		`[{"type":"web_search_20250305"},{"name":"lookup","input_schema":{"type":"object"}}]`,
		"",
	)
}

// The silent-drop fix. A document whose source is text holds the text the turn
// is about. Before this, the block was carried whole and a Chat target dropped
// it: the provider answered a question with its subject removed, the prompt
// token count did not rise, and nothing in the answer said what was missing.
func TestTextDocumentReachesEveryTargetAsText(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, target, anthropicDispositionRequest(t), nil,
			)
			if err != nil {
				t.Fatalf("%s refused a text document: %v", target, err)
			}
			body := string(result.Body)
			// The text itself is what the provider bills and answers on.
			if !strings.Contains(body, documentBody) {
				t.Fatalf("%s dropped the document text: %s", target, body)
			}
			// The title is the only other thing a reader of the block gets.
			if !strings.Contains(body, "Release notes") {
				t.Fatalf("%s dropped the document title: %s", target, body)
			}
			assertApproximatedDiagnosticField(t, result.Diagnostics, "content.document")
		})
	}
}

func TestDispositionTableCountsWhatChatCannotExpress(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		anthropicDispositionRequest(t), nil,
	)
	if err != nil {
		t.Fatalf("Chat refused the request: %v", err)
	}
	for _, field := range []string{"content.citations", "tools.type"} {
		assertDroppedDiagnosticField(t, result.Diagnostics, field)
	}
	if strings.Contains(string(result.Body), "web_search_20250305") {
		t.Fatalf("Chat carried a server tool it cannot run: %s", result.Body)
	}
}

// The Messages target carries every row byte for byte. That is the whole
// asymmetry the table states: a format drops only what it cannot express, and
// Messages can express its own contract.
func TestDispositionTableCarriesEverythingToMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		anthropicDispositionRequest(t), func(request *llmprotocol.Request) error {
			// Mutating forces a real encode instead of a byte-for-byte replay
			// of the source body, which would pass without carrying anything.
			request.Model = "routed-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Messages refused the request: %v", err)
	}
	body := string(result.Body)
	for _, member := range []string{
		`"web_search_20250305"`, `"citations"`, `"title"`, documentBody,
	} {
		if !strings.Contains(body, member) {
			t.Fatalf("Messages dropped %s: %s", member, body)
		}
	}
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Target == llmprotocol.AnthropicMessagesV1 &&
			diagnostic.Action == llmprotocol.DiagnosticDropped {
			t.Fatalf("Messages counted a drop it did not make: %+v", diagnostic)
		}
	}
}

// A tool block's cache breakpoint and caller are the two members that answered
// 500 to every follow-up turn of a tool-using session on 2026-09-04. They are
// rows in the table now rather than a hand-rolled branch.
func TestToolBlockMembersAreTableRows(t *testing.T) {
	engine := NewBuiltinEngine()
	body := anthropicSweepRequest(t,
		`[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"ok"}],`+
			`"cache_control":{"type":"ephemeral"}}]`,
		"", "",
	)
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil,
	)
	if err != nil {
		t.Fatalf("Chat refused a tool turn carrying a cache breakpoint: %v", err)
	}
	assertDroppedDiagnosticField(t, result.Diagnostics, "content.cache_control")
	if dispositionFor("content.cache_control", llmprotocol.AnthropicMessagesV1).Action != dispositionCarry {
		t.Fatal("Messages must carry a cache breakpoint")
	}
}

// Every row states a disposition for both targets that can lack the member,
// and none states one for Messages. A row that named Messages would be a claim
// that Messages cannot express its own contract.
func TestDispositionTableRowsAreComplete(t *testing.T) {
	for _, row := range anthropicRequestDispositions {
		if row.Path == "" {
			t.Fatal("a disposition row names no field path")
		}
		for _, target := range []llmprotocol.WireFormat{
			llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		} {
			disposition, stated := row.Targets[target]
			if !stated {
				t.Fatalf("row %q states nothing for %s", row.Path, target)
			}
			if disposition.Reason == "" {
				t.Fatalf("row %q gives %s no reason", row.Path, target)
			}
		}
		if _, stated := row.Targets[llmprotocol.AnthropicMessagesV1]; stated {
			t.Fatalf("row %q claims Messages cannot carry its own member", row.Path)
		}
	}
}
