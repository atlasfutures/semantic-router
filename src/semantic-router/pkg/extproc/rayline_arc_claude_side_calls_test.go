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
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// claudeCodeFixtureMessages reads a captured Claude Code body's messages and
// their roles.
func claudeCodeFixtureMessages(t *testing.T, name string) ([]json.RawMessage, []string) {
	t.Helper()
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(wireSignalFixture(t, name), &body); err != nil {
		t.Fatal(err)
	}
	roles := make([]string, len(body.Messages))
	for i, message := range body.Messages {
		var m struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(message, &m); err != nil {
			t.Fatal(err)
		}
		roles[i] = m.Role
	}
	return body.Messages, roles
}

// claudeCodeFixtureRequest is the captured body as the client sent it to the
// cell, with the request fields the scrub dropped put back.
func claudeCodeFixtureRequest(t *testing.T, name string) []byte {
	t.Helper()
	messages, _ := claudeCodeFixtureMessages(t, name)
	body, err := json.Marshal(map[string]any{"model": "auto", "max_tokens": 1024, "messages": messages})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// Claude Code 2.1.309's prompt-suggestion call and its auto-compaction call
// (whose directive is followed by a trailing role:system message) are side
// calls; the main turns around them are not.
func TestClaudeCode2309SideCallsAreClassifiedFromTheBody(t *testing.T) {
	cases := []struct {
		fixture string
		kind    raylinearc.PolicyCallKind
		source  string
	}{
		{"claude_code_2.1.309_main_turn_before_suggestion.json", raylinearc.PolicyCallUnknown, ""},
		{"claude_code_2.1.309_prompt_suggestion.json", raylinearc.PolicyCallSide, callKindSourceClaudeSuggestion},
		{"claude_code_2.1.309_main_turn_after_suggestion.json", raylinearc.PolicyCallUnknown, ""},
		{"claude_code_2.1.309_main_turn_before_compaction.json", raylinearc.PolicyCallUnknown, ""},
		{"claude_code_2.1.309_compaction_request.json", raylinearc.PolicyCallSide, callKindSourceClaudeCompactionRq},
		{"claude_code_2.1.309_post_compaction.json", raylinearc.PolicyCallUnknown, ""},
	}
	for _, c := range cases {
		body := claudeCodeFixtureRequest(t, c.fixture)
		if kind, source := raylineARCPolicyCallKindOfBody(nil, policyFormatAnthropic, body, "e"); kind != c.kind || source != c.source {
			t.Errorf("%s: %s/%q, want %s/%q", c.fixture, kind, source, c.kind, c.source)
		}
	}
	// A user who types the words is having a main turn.
	typed := policyTestRequest(t, map[string]any{"role": "user",
		"content": "the log says SUGGESTION MODE and [SUGGESTION MODE: twice; what is it?"})
	if kind, source := raylineARCPolicyCallKindOfBody(nil, policyFormatAnthropic, typed, "e"); kind != raylinearc.PolicyCallUnknown {
		t.Fatalf("a typed message mentioning suggestion mode was classified %s/%s", kind, source)
	}
}

// Through the policy-turn rule with the cell's own signals, each side call
// leaves the episode's prefix where it was: the next typed turn extends it
// (no prefix break), and a compaction moves the epoch once, not twice.
func TestClaudeCode2309SideCallsDoNotBreakThePrefix(t *testing.T) {
	sequences := []struct {
		name  string
		files []string
		want  []string
		epoch int
	}{
		{"prompt suggestion", []string{
			"claude_code_2.1.309_main_turn_before_suggestion.json",
			"claude_code_2.1.309_prompt_suggestion.json",
			"claude_code_2.1.309_main_turn_after_suggestion.json",
		}, []string{raylinearc.PolicyTransitionStart, raylinearc.PolicyTransitionSideCall, raylinearc.PolicyTransitionExtends}, 0},
		{"auto-compaction", []string{
			"claude_code_2.1.309_main_turn_before_compaction.json",
			"claude_code_2.1.309_compaction_request.json",
			"claude_code_2.1.309_post_compaction.json",
		}, []string{raylinearc.PolicyTransitionStart, raylinearc.PolicyTransitionSideCall, raylinearc.PolicyTransitionCompaction}, 1},
	}
	for _, sequence := range sequences {
		t.Run(sequence.name, func(t *testing.T) {
			var state *raylinearc.PolicyEpisodeState
			for turn, name := range sequence.files {
				messages, roles := claudeCodeFixtureMessages(t, name)
				signals := raylineARCPolicyTurnSignals(nil, policyFormatAnthropic, raylinearc.PolicyClientRequest{}, messages, "e")
				next, _, transition := raylinearc.PolicyTurn(state, messages, roles, uint64(turn), signals)
				if transition != sequence.want[turn] {
					t.Fatalf("%s: transition %s, want %s", name, transition, sequence.want[turn])
				}
				if transition == raylinearc.PolicyTransitionSideCall {
					if !reflect.DeepEqual(next, state) {
						t.Fatalf("%s: the side call changed the episode: %+v -> %+v", name, state, next)
					}
					continue // a side call commits nothing
				}
				state = next.Next(messages, strings.Repeat("a", 64), "arm-a")
			}
			if state.Epoch != sequence.epoch {
				t.Fatalf("epoch = %d, want %d", state.Epoch, sequence.epoch)
			}
		})
	}
}

// End to end on the policy selector: each captured side call is decided on
// the current epoch, keeps the held arm and stages nothing, and the next
// typed turn still extends the stored prefix.
func TestClaudeCode2309SideCallsKeepTheEpisodeAndTheHeldArm(t *testing.T) {
	cases := []struct {
		name               string
		before, side, next string
		nextEpoch          string
	}{
		{"prompt suggestion", "claude_code_2.1.309_main_turn_before_suggestion.json",
			"claude_code_2.1.309_prompt_suggestion.json", "claude_code_2.1.309_main_turn_after_suggestion.json", "0"},
		{"auto-compaction", "claude_code_2.1.309_main_turn_before_compaction.json",
			"claude_code_2.1.309_compaction_request.json", "claude_code_2.1.309_post_compaction.json", "1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
			fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
				return request.Selection.AvailableActionIDs[len(request.Selection.AvailableActionIDs)-1]
			})
			state, _ := raylinearc.NewEpisodeState(2)
			held := 0
			state.PreviousArm, state.TurnIndex = &held, 5
			before, _ := claudeCodeFixtureMessages(t, c.before)
			state.Policy = (&raylinearc.PolicyEpisodeState{}).Next(before, strings.Repeat("a", 64), "arm-a")
			stored := cloneARCState(state)

			result, err := fixture.selectWithHeaders(t, state, claudeCodeFixtureRequest(t, c.side), nil)
			if err != nil {
				t.Fatal(err)
			}
			if sent := fixture.fake.received()[0]; sent.ContextEpoch != "0" {
				t.Fatalf("side call decided on epoch %q, want 0", sent.ContextEpoch)
			}
			if !result.RaylineARC.PolicySideCall || result.RaylineARC.PolicyNextState != nil || !reflect.DeepEqual(state, stored) {
				t.Fatalf("side call staged state: side=%v next=%+v", result.RaylineARC.PolicySideCall, result.RaylineARC.PolicyNextState)
			}
			// The fake picks the last action offered: only the held model's
			// levels are, so the held arm serves.
			if result.RaylineARC.SelectedArm != held || len(offeredActions(fixture, 0)) != 2 {
				t.Fatalf("side call chose arm %d from %v, want the held arm %d", result.RaylineARC.SelectedArm, offeredActions(fixture, 0), held)
			}

			result, err = fixture.selectWithHeaders(t, state, claudeCodeFixtureRequest(t, c.next), nil)
			if err != nil {
				t.Fatal(err)
			}
			if sent := fixture.fake.received()[1]; sent.ContextEpoch != c.nextEpoch {
				t.Fatalf("next typed turn decided on epoch %q, want %s", sent.ContextEpoch, c.nextEpoch)
			}
			if result.RaylineARC.PolicySideCall || result.RaylineARC.PolicyNextState == nil {
				t.Fatalf("next typed turn: side=%v next=%v", result.RaylineARC.PolicySideCall, result.RaylineARC.PolicyNextState)
			}
		})
	}
}
