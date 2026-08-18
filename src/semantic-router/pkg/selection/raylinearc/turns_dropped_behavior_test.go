/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"fmt"
	"reflect"
	"testing"
)

// These tests prove what the drop tables in turns_dropped_tables_test.go do.
// Membership lives there; behavior lives here.

// dropCase pins that adding one block to a request changes nothing about the
// turns it produces. Comparing two parser outputs rather than a literal
// expectation keeps these tests independent of the tool rendering format.
type dropCase struct {
	name     string
	baseline string
	variant  string
}

func anthropicMessage(role string, blocks string) string {
	return fmt.Sprintf(
		`{"messages":[{"role":%q,"content":[%s]}]}`,
		role,
		blocks,
	)
}

func anthropicToolResult(blocks string) string {
	return fmt.Sprintf(`{"messages":[
		{"role":"assistant","content":[
			{"type":"tool_use","id":"c1","name":"run","input":{"a":1}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"c1","content":[%s]}
		]}
	]}`, blocks)
}

// anthropicDropCases places the block first, last, and alone. Leading and
// trailing catch a dropped block that leaks an empty string into the join;
// alone catches one that suppresses the turn entirely.
func anthropicDropCases(blockType string) []dropCase {
	block := fmt.Sprintf(`{"type":%q}`, blockType)
	text := `{"type":"text","text":"keep"}`
	cases := make([]dropCase, 0, 9)
	for _, role := range []string{"user", "assistant"} {
		cases = append(cases,
			dropCase{
				role + "_leading",
				anthropicMessage(role, text),
				anthropicMessage(role, block+","+text),
			},
			dropCase{
				role + "_trailing",
				anthropicMessage(role, text),
				anthropicMessage(role, text+","+block),
			},
			dropCase{
				role + "_only",
				anthropicMessage(role, ""),
				anthropicMessage(role, block),
			},
		)
	}
	return append(cases,
		dropCase{
			"tool_result_leading",
			anthropicToolResult(text),
			anthropicToolResult(block + "," + text),
		},
		dropCase{
			"tool_result_trailing",
			anthropicToolResult(text),
			anthropicToolResult(text + "," + block),
		},
		dropCase{
			"tool_result_only",
			anthropicToolResult(""),
			anthropicToolResult(block),
		},
	)
}

func responsesInput(items string) string {
	return fmt.Sprintf(`{"input":[%s]}`, items)
}

// responsesDropCases includes a same-role pair so a dropped item cannot break
// the coalescing that merges adjacent turns.
func responsesDropCases(itemType string) []dropCase {
	item := fmt.Sprintf(`{"type":%q}`, itemType)
	user := `{"type":"message","role":"user","content":"first"}`
	assistant := `{"type":"message","role":"assistant","content":"second"}`
	alsoUser := `{"type":"message","role":"user","content":"third"}`
	pair := user + "," + assistant
	samePair := user + "," + alsoUser
	return []dropCase{
		{"leading", responsesInput(pair), responsesInput(item + "," + pair)},
		{
			"between",
			responsesInput(pair),
			responsesInput(user + "," + item + "," + assistant),
		},
		{"trailing", responsesInput(pair), responsesInput(pair + "," + item)},
		{
			"between_same_role",
			responsesInput(samePair),
			responsesInput(user + "," + item + "," + alsoUser),
		},
		{"only", responsesInput(""), responsesInput(item)},
	}
}

func assertDropCases(
	t *testing.T,
	protocol InputProtocol,
	typeName string,
	cases []dropCase,
) {
	t.Helper()
	for _, test := range cases {
		t.Run(typeName+"/"+test.name, func(t *testing.T) {
			want, err := NormalizeTurns(protocol, []byte(test.baseline))
			if err != nil {
				t.Fatalf("baseline normalization failed: %v", err)
			}
			got, err := NormalizeTurns(protocol, []byte(test.variant))
			if err != nil {
				t.Fatalf("%q must drop, got error: %v", typeName, err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf(
					"%q changed the turns: got %#v, want %#v",
					typeName,
					got,
					want,
				)
			}
		})
	}
}

func TestAnthropicDroppedBlocksContributeNoText(t *testing.T) {
	for _, blockType := range sortedKeys(anthropicDroppedBlockTypes) {
		assertDropCases(
			t,
			ProtocolAnthropicMessages,
			blockType,
			anthropicDropCases(blockType),
		)
	}
}

func TestResponsesDroppedItemsContributeNoText(t *testing.T) {
	for _, itemType := range sortedKeys(droppedResponsesItemTypes) {
		assertDropCases(
			t,
			ProtocolOpenAIResponses,
			itemType,
			responsesDropCases(itemType),
		)
	}
}

// TestUnknownTypesStillFailClosed guards the property the drop tables exist to
// balance. Widening a table must never widen the default with it: a type in
// neither set has to reject the episode rather than route a silently shortened
// conversation to a trained selector.
func TestUnknownTypesStillFailClosed(t *testing.T) {
	unknown := []struct {
		name     string
		protocol InputProtocol
		request  string
	}{
		{
			"anthropic_user_block",
			ProtocolAnthropicMessages,
			anthropicMessage("user", `{"type":"not_a_real_block"}`),
		},
		{
			"anthropic_assistant_block",
			ProtocolAnthropicMessages,
			anthropicMessage("assistant", `{"type":"not_a_real_block"}`),
		},
		{
			"anthropic_tool_result_block",
			ProtocolAnthropicMessages,
			anthropicToolResult(`{"type":"not_a_real_block"}`),
		},
		{
			"responses_item",
			ProtocolOpenAIResponses,
			responsesInput(`{"type":"not_a_real_item"}`),
		},
	}
	for _, test := range unknown {
		t.Run(test.name, func(t *testing.T) {
			_, err := NormalizeTurns(test.protocol, []byte(test.request))
			if code := TurnNormalizationErrorCode(err); code != "unknown_item" {
				t.Fatalf("error code = %q (%v), want unknown_item", code, err)
			}
		})
	}
}

func openAIChatMessage(role string, parts string) string {
	return fmt.Sprintf(
		`{"messages":[{"role":%q,"content":[%s]}]}`,
		role,
		parts,
	)
}

func responsesMessage(role string, parts string) string {
	return fmt.Sprintf(
		`{"input":[{"type":"message","role":%q,"content":[%s]}]}`,
		role,
		parts,
	)
}

// TestOpenAIDroppedContentTypesContributeNoText walks the one drop table three
// protocols share. openAIContentText serves Chat and Responses messages alike,
// and the Anthropic tool_result reader consults the same table, so an entry
// removed here changes behavior in places the Chat goldens never look.
func TestOpenAIDroppedContentTypesContributeNoText(t *testing.T) {
	for _, contentType := range sortedKeys(openAIDroppedContentTypes) {
		part := fmt.Sprintf(`{"type":%q}`, contentType)
		text := `{"type":"text","text":"keep"}`
		for _, role := range []string{"user", "assistant"} {
			assertDropCases(t, ProtocolOpenAIChat, contentType, []dropCase{
				{
					"chat_" + role + "_leading",
					openAIChatMessage(role, text),
					openAIChatMessage(role, part+","+text),
				},
				{
					"chat_" + role + "_trailing",
					openAIChatMessage(role, text),
					openAIChatMessage(role, text+","+part),
				},
				{
					"chat_" + role + "_only",
					openAIChatMessage(role, ""),
					openAIChatMessage(role, part),
				},
			})
			assertDropCases(t, ProtocolOpenAIResponses, contentType, []dropCase{
				{
					"responses_" + role + "_leading",
					responsesMessage(role, text),
					responsesMessage(role, part+","+text),
				},
				{
					"responses_" + role + "_only",
					responsesMessage(role, ""),
					responsesMessage(role, part),
				},
			})
		}
		assertDropCases(t, ProtocolAnthropicMessages, contentType, []dropCase{
			{
				"anthropic_tool_result_leading",
				anthropicToolResult(text),
				anthropicToolResult(part + "," + text),
			},
			{
				"anthropic_tool_result_only",
				anthropicToolResult(""),
				anthropicToolResult(part),
			},
		})
	}
}
