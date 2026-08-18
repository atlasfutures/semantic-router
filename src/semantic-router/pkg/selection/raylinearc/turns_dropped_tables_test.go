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
	"reflect"
	"sort"
	"testing"
)

// The drop tables decide which request shapes route and which return 503, so
// they need two different tests. The membership tests below pin the exact
// contents, because a test that iterates a table cannot notice an entry
// deleted from it. The behavior tests then prove every listed type really
// contributes nothing, in every position the vendor schema allows it.
//
// When either list changes, re-measure against the vendor SDK unions rather
// than against a production failure report: a type reaches the fail-closed
// default long before anyone files the ticket.

var wantAnthropicDroppedBlockTypes = []string{
	"advisor_tool_result",
	"bash_code_execution_tool_result",
	"code_execution_tool_result",
	"container_upload",
	"document",
	"fallback",
	"image",
	"mcp_tool_result",
	"mcp_tool_use",
	"mid_conv_system",
	"redacted_thinking",
	"search_result",
	"server_tool_use",
	"text_editor_code_execution_tool_result",
	"thinking",
	"tool_search_tool_result",
	"web_fetch_tool_result",
	"web_search_tool_result",
}

var wantOpenAIDroppedContentTypes = []string{
	"file",
	"image_url",
	"input_audio",
	"input_file",
	"input_image",
	"output_image",
	"refusal",
}

var wantDroppedResponsesItemTypes = []string{
	"apply_patch_call",
	"apply_patch_call_output",
	"code_interpreter_call",
	"compaction",
	"computer_call",
	"computer_call_output",
	"custom_tool_call",
	"custom_tool_call_output",
	"file_search_call",
	"image_generation_call",
	"item_reference",
	"local_shell_call",
	"local_shell_call_output",
	"mcp_approval_request",
	"mcp_approval_response",
	"mcp_call",
	"mcp_list_tools",
	"reasoning",
	"shell_call",
	"shell_call_output",
	"web_search_call",
}

// anthropicRenderedBlockTypes and responsesRenderedItemTypes name the types the
// parser turns into turn text. A type must be rendered or dropped, never both
// and never neither.
var anthropicRenderedBlockTypes = []string{
	"compaction",
	"text",
	"tool_result",
	"tool_use",
}

var responsesRenderedItemTypes = []string{
	"function_call",
	"function_call_output",
	"message",
}

// openAIRenderedContentTypes are the message content parts that carry text.
// Chat and Responses both read them through openAIContentText.
var openAIRenderedContentTypes = []string{
	"input_text",
	"output_text",
	"text",
}

func sortedKeys(table map[string]bool) []string {
	keys := make([]string, 0, len(table))
	for key := range table {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func assertTableMembership(t *testing.T, table map[string]bool, want []string) {
	t.Helper()
	if got := sortedKeys(table); !reflect.DeepEqual(got, want) {
		t.Fatalf("drop table = %#v, want %#v", got, want)
	}
}

func TestAnthropicDroppedBlockTableMembership(t *testing.T) {
	assertTableMembership(
		t,
		anthropicDroppedBlockTypes,
		wantAnthropicDroppedBlockTypes,
	)
}

func TestResponsesDroppedItemTableMembership(t *testing.T) {
	assertTableMembership(
		t,
		droppedResponsesItemTypes,
		wantDroppedResponsesItemTypes,
	)
}

func TestOpenAIDroppedContentTableMembership(t *testing.T) {
	assertTableMembership(
		t,
		openAIDroppedContentTypes,
		wantOpenAIDroppedContentTypes,
	)
}

func TestRenderedAndDroppedTypesAreDisjoint(t *testing.T) {
	for _, blockType := range anthropicRenderedBlockTypes {
		if anthropicDroppedBlockTypes[blockType] {
			t.Errorf("anthropic %q is both rendered and dropped", blockType)
		}
	}
	for _, itemType := range responsesRenderedItemTypes {
		if droppedResponsesItemTypes[itemType] {
			t.Errorf("responses %q is both rendered and dropped", itemType)
		}
	}
	for _, contentType := range openAIRenderedContentTypes {
		if openAIDroppedContentTypes[contentType] {
			t.Errorf("openai %q is both rendered and dropped", contentType)
		}
	}
}
