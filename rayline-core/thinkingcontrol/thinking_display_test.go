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

package thinkingcontrol

import (
	"encoding/json"
	"strings"
	"testing"
)

// D-haiku-thinking-fix (2026-10-28): Claude 5.x thinks adaptively by default
// with its display omitted, so a hard turn streamed nothing until its text.
// The Claude 5.5 Messages cells ask for the summarized display on every base:
// every admitted control renders exactly one thinking member, {type:
// adaptive, display: summarized}, in place of whatever the client sent, and
// an effort base keeps its output_config.effort beside it.
func TestClaudeMessagesCellsRenderTheSummarizedThinkingDisplayAtEveryLevel(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	// A Claude Code body states its own adaptive thinking and effort; the
	// cell's base replaces both.
	client := []byte(`{"model":"claude-x","max_tokens":32000,"thinking":{"type":"adaptive","display":"omitted"},` +
		`"output_config":{"effort":"xhigh","format":{"type":"text"}},"stream":true,` +
		`"messages":[{"role":"user","content":"List the files."}]}`)
	for _, model := range []string{"anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5"} {
		for _, provider := range []string{"openrouter", "anthropic"} {
			cell, err := reg.Cell(model, provider, FormatMessages)
			if err != nil {
				t.Fatal(err)
			}
			ids := cell.ControlIDs()
			bases := map[string]bool{}
			for _, id := range ids {
				control, _ := reg.Control(id)
				if _, err := reg.Admit(model, provider, FormatMessages, control, true); err != nil {
					// An effort-item or every-turn control VSR does not place; its
					// base still renders through the controls VSR does.
					continue
				}
				placer, err := NewPlacer(FormatMessages)
				if err != nil {
					t.Fatal(err)
				}
				body, _, err := placer.Render(client, &control, cell, model)
				if err != nil {
					t.Fatalf("%s on %s, control %s: %v", model, provider, prefix12(id), err)
				}
				if count := strings.Count(string(body), `"thinking":`); count != 1 {
					t.Fatalf("%s on %s, base %s: %d thinking members in %s", model, provider, control.Native, count, body)
				}
				var rendered struct {
					Thinking     map[string]any `json:"thinking"`
					OutputConfig map[string]any `json:"output_config"`
				}
				if err := json.Unmarshal(body, &rendered); err != nil {
					t.Fatal(err)
				}
				if len(rendered.Thinking) != 2 || rendered.Thinking["type"] != "adaptive" ||
					rendered.Thinking["display"] != "summarized" {
					t.Fatalf("%s on %s, base %s: thinking %v", model, provider, control.Native, rendered.Thinking)
				}
				effort, stated := rendered.OutputConfig["effort"]
				switch control.Native {
				case "default":
					if stated {
						t.Fatalf("%s on %s: the default base kept an effort %v", model, provider, effort)
					}
				default:
					if effort != control.Native {
						t.Fatalf("%s on %s, base %s: effort %v", model, provider, control.Native, effort)
					}
				}
				bases[control.Native] = true
				if rendered.OutputConfig["format"] == nil {
					t.Fatalf("%s on %s: the client's other output_config keys were dropped: %s", model, provider, body)
				}
			}
			// Every base the cell admits rendered through at least one control.
			for base := range cell.baseWire {
				if !bases[base] {
					t.Fatalf("%s on %s: base %s rendered through no control VSR places", model, provider, base)
				}
			}
		}
	}
}
