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

package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// semantic-router #248, ADR 0131 on the policy-mode lever: kimi-k3's Chat
// cell spells a steer after a text tool run as a text part of the run's last
// tool message; another Chat worker keeps the user message after the run.
func TestTheLeverSpellsATextToolTailSteerOnTheToolMessageOnlyWhereTheCellSaysSo(t *testing.T) {
	result := llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c1", Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "README.md"},
		}},
	}}}
	call := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "ls", Arguments: "{}"},
	}}}
	messages := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go"), call, result}
	for _, tc := range []struct {
		worker  string
		spelled bool
	}{{leverWorker, false}, {evidencedWorker, true}} {
		t.Run(tc.worker, func(t *testing.T) {
			e := newLeverEpisode(t, true)
			e.wire = llmprotocol.OpenAIChatV1
			e.config = evidenceRouterConfig()
			lever := e.decision.Algorithm.RaylineARC.ThinkingLever
			lever.Workers[evidencedWorker] = lever.Workers[leverWorker]
			sent, ctx := e.turn(tc.worker, messages, true)
			record := map[string]interface{}{}
			appendRaylineARCThinkingFields(record, ctx)
			if record["thinking_emitted"] != true || record["thinking_placement"] != "insert_user_after_tool_run" {
				t.Fatalf("record %v; want the steer written after the tool run", record)
			}
			wire := encodedChatMessages(t, sent)
			last := marshalCanonical(imageTailChatMessages(t, `{"messages":[`+string(wire[len(wire)-1])+`]}`)[0])
			if !tc.spelled {
				if record["thinking_tool_tail_spelling"] != nil || last != `{"content":"`+leverDown+`","role":"user"}` {
					t.Fatalf("a cell that states no spelling: record %v, last message %s", record, last)
				}
				return
			}
			want := `{"content":[{"text":"README.md","type":"text"},{"text":"` + leverDown + `","type":"text"}],` +
				`"role":"tool","tool_call_id":"c1"}`
			if record["thinking_tool_tail_spelling"] != "append_to_tool" || last != want {
				t.Fatalf("kimi-k3 Chat: record %v, last message\n%s\nwant\n%s", record, last, want)
			}
		})
	}
}
