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
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// include_tool_names changes what the selector is asked on every ROUTED turn,
// not only on a route lookup, so the config key has to reach the projection
// the request pipeline actually builds. Testing ProjectTurns directly proves
// the projection honours an option; it cannot notice the wiring from config
// to that option being dropped, which would silently change every routed
// decision back and void the encoder measurement behind the flag.
func toolNamesRouter() *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{}}
}

func toolNamesAlgorithm(include bool) *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type:    config.RaylineARCAlgorithmType,
		OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			IncludeToolNames: include,
			Episode:          config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session"},
		},
	}
}

func toolNamesRequestContext() *RequestContext {
	return &RequestContext{
		Headers: map[string]string{},
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Tools: []llmprotocol.Tool{
				{
					Name:        "Edit",
					Description: "Edit a file in place, replacing an exact string.",
					InputSchema: []byte(`{"type":"object","properties":{"path":{"type":"string"}}}`),
				},
				{Type: "web_search_20250305"},
			},
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "rename the helper"}},
			}},
		},
	}
}

func TestIncludeToolNamesReachesTheProjectionFromConfig(t *testing.T) {
	t.Parallel()
	selectionContext := toolNamesRouter().buildRaylineARCSelectionContext(
		toolNamesAlgorithm(true),
		toolNamesRequestContext(),
		nil,
		raylineARCEpisodeEphemeral,
	)
	if selectionContext == nil || selectionContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %+v", selectionContext)
	}
	if len(selectionContext.Turns) == 0 {
		t.Fatal("no turns were projected")
	}
	text := selectionContext.Turns[0].Text

	// The names reach the encoder, in declaration order, behind the exact
	// prefix the probe measured -- including the server tool, which has a
	// type and no name.
	// The prefix is spelled out rather than imported: it is the exact string
	// the encoder probe measured, so a change to it has to break a test and be
	// thought about, not follow silently from the constant moving.
	if want := "[available tools] Edit, web_search_20250305"; !strings.Contains(text, want) {
		t.Fatalf("turn text = %q, want it to carry %q", text, want)
	}
	// The schemas do not. That is the whole finding the probe rested on: the
	// contract is a shared prefix that drowns the per-episode signal, and
	// sending it would destroy the routing it was meant to improve.
	for _, leaked := range []string{"input_schema", "properties", "Edit a file in place"} {
		if strings.Contains(text, leaked) {
			t.Fatalf("turn text = %q, want no %q", text, leaked)
		}
	}
}

// Off by default, and off means byte-identical to what the encoder saw before
// tool names were readable at all. One byte of drift voids the calibration.
func TestToolNamesStayOutUnlessTheConfigAsksForThem(t *testing.T) {
	t.Parallel()
	selectionContext := toolNamesRouter().buildRaylineARCSelectionContext(
		toolNamesAlgorithm(false),
		toolNamesRequestContext(),
		nil,
		raylineARCEpisodeEphemeral,
	)
	if selectionContext == nil || selectionContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %+v", selectionContext)
	}
	if text := selectionContext.Turns[0].Text; text != "rename the helper" {
		t.Fatalf("turn text = %q, want the turn alone", text)
	}
}
