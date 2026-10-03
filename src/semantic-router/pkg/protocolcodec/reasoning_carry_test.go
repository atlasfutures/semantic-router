package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func reasoningHistory() llmprotocol.Request {
	return llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "q1"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "claude thought", Signature: "sig-1", Reasoning: llmprotocol.ReasoningScopeText},
			{Kind: llmprotocol.ContentText, Text: "a1"},
		}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "q2"}}},
		// kimi on OpenRouter Messages: thinking with an empty signature, alone.
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "kimi thought", Reasoning: llmprotocol.ReasoningScopeText},
		}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "kimi again", Reasoning: llmprotocol.ReasoningScopeText},
			{Kind: llmprotocol.ContentText, Text: "a2"},
		}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "q3"}}},
	}}
}

func reasoningOf(request llmprotocol.Request) (signed, unsigned int) {
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentReasoning {
				continue
			}
			if content.Signature != "" {
				signed++
			} else {
				unsigned++
			}
		}
	}
	return signed, unsigned
}

// A Messages target keeps signed thinking and drops what has no signature,
// removing a message that held nothing else; visible text is untouched.
func TestCarryReasoningToMessagesDropsUnsignedThinking(t *testing.T) {
	request := reasoningHistory()
	original := reasoningHistory()
	carry := CarryReasoningTo(&request, llmprotocol.AnthropicMessagesV1)
	if carry != (ReasoningCarry{UnsignedDropped: 2}) {
		t.Fatalf("carry = %+v, want 2 unsigned dropped", carry)
	}
	if signed, unsigned := reasoningOf(request); signed != 1 || unsigned != 0 {
		t.Fatalf("after carry: %d signed, %d unsigned reasoning; want 1, 0", signed, unsigned)
	}
	if len(request.Messages) != len(original.Messages)-1 {
		t.Fatalf("messages = %d, want the reasoning-only message removed (%d)", len(request.Messages), len(original.Messages)-1)
	}
	last := request.Messages[len(request.Messages)-2]
	if len(last.Content) != 1 || last.Content[0].Text != "a2" {
		t.Fatalf("the mixed message kept %+v, want only its text", last.Content)
	}
}

// A Chat or Responses target cannot verify an Anthropic signature: it is
// stripped and counted, the thinking text is carried as reasoning (never as
// visible text), and the capability gate no longer refuses the turn.
// Unsigned reasoning stays reasoning: a Chat target needs it back.
func TestCarryReasoningToChatAndResponsesStripsTheSignature(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		request := reasoningHistory()
		carry := CarryReasoningTo(&request, target)
		if carry != (ReasoningCarry{SignaturesStripped: 1}) || carry.Dropped() != 0 || !carry.Changed() {
			t.Fatalf("%s: carry = %+v, want 1 signature stripped and nothing dropped", target, carry)
		}
		if signed, unsigned := reasoningOf(request); signed != 0 || unsigned != 3 {
			t.Fatalf("%s: %d signed, %d unsigned; want 0, 3", target, signed, unsigned)
		}
		first := request.Messages[1]
		if len(first.Content) != 2 ||
			first.Content[0].Kind != llmprotocol.ContentReasoning || first.Content[0].Text != "claude thought" || first.Content[0].Signature != "" ||
			first.Content[1].Kind != llmprotocol.ContentText || first.Content[1].Text != "a1" {
			t.Fatalf("%s: the signed message became %+v, want its thinking unsigned and its text", target, first.Content)
		}
		if len(request.Messages) != len(reasoningHistory().Messages) {
			t.Fatalf("%s: a message was removed", target)
		}
		if err := llmprotocol.RequireCapabilities(target, NewBuiltinRegistry().mustCapabilities(t, target),
			llmprotocol.RequiredCapabilities(request)); err != nil {
			t.Fatalf("%s: capability gate still refuses: %v", target, err)
		}
	}
}

// redacted_thinking is readable only by its issuer: a Chat or Responses
// target drops and counts it, removing a message it leaves empty; a Messages
// target keeps it.
func TestCarryReasoningToDropsRedactedThinkingForForeignTargets(t *testing.T) {
	history := func() llmprotocol.Request {
		return llmprotocol.Request{Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "q"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{redactedBlock("blob")}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{redactedBlock("blob-2"), {Kind: llmprotocol.ContentText, Text: "a"}}},
		}}
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		request := history()
		if carry := CarryReasoningTo(&request, target); carry != (ReasoningCarry{RedactedDropped: 2}) {
			t.Fatalf("%s: carry = %+v, want 2 redacted dropped", target, carry)
		}
		if len(request.Messages) != 2 || len(request.Messages[1].Content) != 1 || request.Messages[1].Content[0].Text != "a" {
			t.Fatalf("%s: messages = %+v", target, request.Messages)
		}
	}
	request := history()
	if carry := CarryReasoningTo(&request, llmprotocol.AnthropicMessagesV1); carry.Changed() {
		t.Fatalf("Messages dropped issuer-readable reasoning: %+v", carry)
	}
}

// Every reasoning kind the table names has the disposition CarryReasoningTo
// applies, and nothing a row drops is left behind.
func TestCarryReasoningFollowsTheDispositionTable(t *testing.T) {
	blocks := map[string]llmprotocol.Content{
		fieldReasoningSigned:   {Kind: llmprotocol.ContentReasoning, Text: "t", Signature: "s", Reasoning: llmprotocol.ReasoningScopeText},
		fieldReasoningUnsigned: {Kind: llmprotocol.ContentReasoning, Text: "t", Reasoning: llmprotocol.ReasoningScopeText},
		fieldRedactedThinking:  redactedBlock("blob"),
	}
	for path, block := range blocks {
		if reasoningProvenance(block) != path {
			t.Fatalf("%s: provenance = %q", path, reasoningProvenance(block))
		}
		for _, target := range []llmprotocol.WireFormat{
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		} {
			request := llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{block, {Kind: llmprotocol.ContentText, Text: "a"}},
			}}}
			carry := CarryReasoningTo(&request, target)
			wantKept := 2
			if dispositionFor(path, target).Action == dispositionDrop {
				wantKept = 1
			}
			if carry.Dropped() != 2-wantKept || len(request.Messages[0].Content) != wantKept {
				t.Fatalf("%s to %s: carry = %+v, content = %+v", path, target, carry, request.Messages[0].Content)
			}
		}
	}
}

// The carry builds new slices: a request prepared for one target shares no
// change with the request it came from.
func TestCarryReasoningToLeavesTheSourceUntouched(t *testing.T) {
	source := reasoningHistory()
	request := source
	CarryReasoningTo(&request, llmprotocol.OpenAIChatV1)
	if signed, unsigned := reasoningOf(source); signed != 1 || unsigned != 2 || len(source.Messages) != 6 ||
		len(source.Messages[1].Content) != 2 {
		t.Fatalf("source changed: %d signed, %d unsigned, %d messages", signed, unsigned, len(source.Messages))
	}
}

func (registry *Registry) mustCapabilities(t *testing.T, format llmprotocol.WireFormat) llmprotocol.CapabilitySet {
	t.Helper()
	capabilities, ok := registry.CapabilitiesFor(format)
	if !ok {
		t.Fatalf("no capabilities for %s", format)
	}
	return capabilities
}

func redactedBlock(data string) llmprotocol.Content {
	return carriedAnthropicBlock("redacted_thinking", []byte(`{"type":"redacted_thinking","data":"`+data+`"}`))
}

// Opaque reasoning is found in a response and dropped from a request by its
// data, removing a message it leaves empty; other content is untouched.
func TestOpaqueReasoningIsFoundAndDropped(t *testing.T) {
	response := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Content: []llmprotocol.Content{
		redactedBlock("blob-a"), {Kind: llmprotocol.ContentText, Text: "answer"},
	}}}}
	if blocks := ResponseOpaqueReasoning(response); len(blocks) != 1 || blocks[0] != "blob-a" {
		t.Fatalf("response opaque reasoning = %v", blocks)
	}
	request := llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{redactedBlock("blob-a"), {Kind: llmprotocol.ContentText, Text: "a1"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{redactedBlock("blob-a")}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{redactedBlock("blob-b")}},
	}}
	dropped := DropOpaqueReasoning(&request, func(data string) bool { return data == "blob-a" })
	if dropped != 2 || len(request.Messages) != 2 || len(request.Messages[0].Content) != 1 ||
		request.Messages[0].Content[0].Text != "a1" {
		t.Fatalf("dropped %d, messages %+v", dropped, request.Messages)
	}
}
