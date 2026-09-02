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
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRaylineARCImageSignalSurvivesTheDroppedProjection(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name    string
		request *llmprotocol.Request
		want    bool
	}{
		{
			name: "text only",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "describe this"},
				},
			}}},
			want: false,
		},
		{
			name: "image block in a user message",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "describe this"},
					{
						Kind:      llmprotocol.ContentImage,
						MediaType: "image/png",
						URL:       "https://example.invalid/a.png",
					},
				},
			}}},
			want: true,
		},
		{
			// The provider receives the tool result verbatim, so an image
			// nested in one is image input just as much as a top-level block.
			name: "image nested in a tool result",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolResult,
					ToolResult: &llmprotocol.ToolResult{
						CallID: "call-1",
						Content: []llmprotocol.Content{{
							Kind:      llmprotocol.ContentImage,
							MediaType: "image/png",
							URL:       "https://example.invalid/b.png",
						}},
					},
				}},
			}}},
			want: true,
		},
		{
			name: "image block in the opening instructions",
			request: &llmprotocol.Request{
				Instructions: []llmprotocol.InstructionBlock{{
					Content: []llmprotocol.Content{{
						Kind:      llmprotocol.ContentImage,
						MediaType: "image/png",
						URL:       "https://example.invalid/c.png",
					}},
				}},
			},
			want: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := requestCarriesImageInput(test.request); got != test.want {
				t.Fatalf("requestCarriesImageInput() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestRaylineARCNonVisionArmsReadTheModelCards(t *testing.T) {
	t.Parallel()
	no := false
	yes := true
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {},
				"arm-1": {Vision: &no},
				"arm-2": {Vision: &yes},
			},
		},
	}}
	refs := []config.ModelRef{
		{Model: "arm-0"},
		{Model: "arm-1"},
		{Model: "arm-2"},
	}
	want := []bool{false, true, false}
	if got := router.nonVisionArms(refs); !reflect.DeepEqual(got, want) {
		t.Fatalf("nonVisionArms() = %v, want %v", got, want)
	}
}

// An unmarked catalog must behave exactly as it did before the flag existed,
// which is what makes the flag safe to add without touching any config.
func TestRaylineARCNonVisionArmsAreNilWhenNothingIsMarked(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"arm-0": {}},
		},
	}}
	if got := router.nonVisionArms([]config.ModelRef{{Model: "arm-0"}}); got != nil {
		t.Fatalf("nonVisionArms() = %v, want nil", got)
	}
}
