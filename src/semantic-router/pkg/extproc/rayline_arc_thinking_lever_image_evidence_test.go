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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const evidencedWorker = "kimi@default"

// evidenceRouterConfig serves leverWorker as z-ai/glm-5.3-flash and
// evidencedWorker as moonshotai/kimi-k3, both through OpenRouter: of the
// two Chat cells, only kimi-k3's states task-fidelity evidence for a steer
// after an image tool result (pathfinder #4028).
func evidenceRouterConfig() *config.RouterConfig {
	return &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			leverWorker: {
				PreferredEndpoints: []string{"openrouter"}, APIFormat: config.APIFormatOpenAI,
				ExternalModelIDs: map[string]string{"vllm": "z-ai/glm-5.3-flash"},
			},
			evidencedWorker: {
				PreferredEndpoints: []string{"openrouter"}, APIFormat: config.APIFormatOpenAI,
				ExternalModelIDs: map[string]string{"vllm": "moonshotai/kimi-k3"},
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "openrouter", Address: "openrouter.ai", Port: 443, ProviderProfileName: "openrouter"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"openrouter": {Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
		},
	}}
}

// ADR 0129 decision 5 on the policy-mode lever: the steer after an image
// tool result is refused on a Chat worker whose admission cell states no
// task-fidelity evidence, and written on kimi-k3's evidenced cell, where it
// joins the hoisted image message (#242) instead of following it.
func TestTheLeverWritesAnImageToolTailSteerOnlyWhereTheCellHasEvidence(t *testing.T) {
	image := llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c1", Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "shot.png"},
			{Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: "iVBORw0KGgo="},
		}},
	}}}
	call := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "shot", Arguments: "{}"},
	}}}
	messages := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go"), call, image}
	for _, tc := range []struct {
		worker   string
		evidence bool
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
			if !tc.evidence {
				if len(sent) != len(messages) || record["thinking_refused"] != "image_tool_tail_task_fidelity" {
					t.Fatalf("unevidenced cell: sent %d messages, record %v; want the refusal", len(sent), record)
				}
				return
			}
			if record["thinking_refused"] != nil || record["thinking_emitted"] != true || record["thinking_level_in_force"] != "down" {
				t.Fatalf("evidenced cell: record %v; want the steer written", record)
			}
			wire := encodedChatMessages(t, sent)
			last := marshalCanonical(imageTailChatMessages(t, `{"messages":[`+string(wire[len(wire)-1])+`]}`)[0])
			want := `{"content":[{"text":"[images returned by tool call c1]","type":"text"},` +
				`{"image_url":{"url":"data:image/png;base64,iVBORw0KGgo="},"type":"image_url"},` +
				`{"text":"` + leverDown + `","type":"text"}],"role":"user"}`
			if last != want {
				t.Fatalf("the steer did not join the hoisted image message:\n%s\nwant\n%s", last, want)
			}
		})
	}
}

// A router with no configuration names no admission cell, so it has no
// evidence: the steer is refused, never written on a guess.
func TestNoRegistryCellMeansNoImageToolTailEvidence(t *testing.T) {
	if raylineARCImageToolTailAdmitted(nil, evidencedWorker, llmprotocol.OpenAIChatV1) {
		t.Fatal("a router with no config admitted the steer")
	}
	cfg := evidenceRouterConfig()
	if raylineARCImageToolTailAdmitted(cfg, "unknown@default", llmprotocol.OpenAIChatV1) {
		t.Fatal("a worker with no endpoint admitted the steer")
	}
	if raylineARCImageToolTailAdmitted(cfg, evidencedWorker, llmprotocol.OpenAIResponsesV1) {
		t.Fatal("kimi-k3's Responses cell states no evidence")
	}
	if !raylineARCImageToolTailAdmitted(cfg, evidencedWorker, llmprotocol.OpenAIChatV1) {
		t.Fatal("kimi-k3's Chat cell states evidence")
	}
}
