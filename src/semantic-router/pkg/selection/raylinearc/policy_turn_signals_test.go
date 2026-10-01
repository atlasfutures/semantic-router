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
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"testing"
)

// scheduleScript is one scripted session: compactions maps a completed-turn
// count to the ordinal of the compaction the next request opens; side calls
// and prefix breaks happen before the main request at their turn.
type scheduleScript struct {
	compactions  map[int]int
	sideCalls    map[int]bool
	prefixBreaks map[int]bool
	// marker signals compaction through Claude Code's continuation summary
	// rather than through the explicit ordinal.
	marker bool
}

func jsonString(text string) string {
	encoded, _ := json.Marshal(text)
	return string(encoded)
}

// decisionPoints drives the policy episode the way the router does and
// returns the completed-turn counts at which the model could change. A main
// turn resolves against the stored state and commits; a side call resolves
// against it and commits nothing.
func (script scheduleScript) decisionPoints(t *testing.T) []int {
	t.Helper()
	var policy *PolicyEpisodeState
	var previousArm *int
	transcript := []string{`{"role":"user","content":"task"}`}
	var points []int
	for turnIndex := uint64(0); turnIndex < 18; turnIndex++ {
		completed := int(turnIndex)
		if script.sideCalls[completed] {
			title, roles := rawMessages(t, `{"role":"user","content":"write a five word title"}`)
			turn, _, transition := PolicyTurn(policy, title, roles, turnIndex,
				PolicyTurnSignals{CallKind: PolicyCallSide})
			if transition != PolicyTransitionSideCall || policy != nil && !reflect.DeepEqual(turn, policy) {
				t.Fatalf("side call at %d changed the episode: %+v", completed, turn)
			}
		}
		signals := PolicyTurnSignals{}
		if ordinal, ok := script.compactions[completed]; ok {
			summary := fmt.Sprintf(`{"role":"user","content":%s}`,
				jsonString(ClaudeCodeContinuationMarker+" Summary "+strconv.Itoa(ordinal)+"."))
			transcript = []string{summary}
			if !script.marker {
				signals.Compaction = &PolicyCompactionSignal{Ordinal: ordinal}
			}
		} else if script.prefixBreaks[completed] {
			transcript = []string{fmt.Sprintf(`{"role":"user","content":"rewritten at %d"}`, completed)}
		}
		messages, roles := rawMessages(t, transcript...)
		if script.marker {
			if digest := ClaudeCodeCompactionSummaryDigest(messages); digest != "" {
				signals.Compaction = &PolicyCompactionSignal{SummaryDigest: digest}
			}
		}
		turn, _, _ := PolicyTurn(policy, messages, roles, turnIndex, signals)
		if previousArm == nil || ModelChangeAllowed(turnIndex, turn.EpochStartTurn, turn.CompactionCount) {
			points = append(points, completed)
		}
		arm := completed
		previousArm = &arm
		policy = turn.Next(messages, fmt.Sprintf("action-%d", completed), "arm")
		transcript = append(transcript,
			`{"role":"assistant","content":"reply"}`,
			fmt.Sprintf(`{"role":"user","content":"next %d"}`, completed))
	}
	return points
}

// The decision points of pathfinder's own schedule tests
// (tests/test_model_routing_state.py,
// test_sparse_model_clock_survives_every_event_round_trip), with a side call
// and a prefix break interleaved that must change none of them.
func TestPolicyScheduleMatchesPathfinderDecisionPoints(t *testing.T) {
	cases := []struct {
		compactions map[int]int
		expected    []int
	}{
		{map[int]int{}, []int{0, 5}},
		{map[int]int{9: 1}, []int{0, 5, 9, 14}},
		{map[int]int{3: 1}, []int{0, 3, 5, 8}},
		{map[int]int{5: 1}, []int{0, 5, 10}},
		{map[int]int{3: 1, 6: 2}, []int{0, 3, 5, 6, 11}},
	}
	for _, marker := range []bool{false, true} {
		for _, c := range cases {
			script := scheduleScript{
				compactions:  c.compactions,
				sideCalls:    map[int]bool{1: true, 5: true, 9: true, 12: true},
				prefixBreaks: map[int]bool{2: true, 7: true, 12: true, 16: true},
				marker:       marker,
			}
			if got := script.decisionPoints(t); !slices.Equal(got, c.expected) {
				t.Errorf("marker=%v compactions=%v: decision points %v, want %v", marker, c.compactions, got, c.expected)
			}
		}
	}
}

// pathfinder's test_compaction_before_first_completed_turn_is_not_inferred_from_zero:
// a session whose first request is already a compacted context has its
// next boundaries at five, for both reasons, and none at six.
func TestPolicyScheduleCompactionBeforeTheFirstTurn(t *testing.T) {
	summary, roles := rawMessages(t, fmt.Sprintf(`{"role":"user","content":%s}`,
		jsonString(ClaudeCodeContinuationMarker+" Resumed.")))
	turn, _, transition := PolicyTurn(nil, summary, roles, 0, PolicyTurnSignals{
		Compaction: &PolicyCompactionSignal{SummaryDigest: ClaudeCodeCompactionSummaryDigest(summary)},
	})
	if transition != PolicyTransitionCompaction || turn.CompactionCount != 1 || turn.EpochStartTurn != 0 {
		t.Fatalf("resumed session: %+v (%s)", turn, transition)
	}
	for completed, want := range map[uint64]bool{0: true, 1: false, 5: true, 6: false, 10: false} {
		if got := ModelChangeAllowed(completed, turn.EpochStartTurn, turn.CompactionCount); got != want {
			t.Errorf("completed=%d: allowed %v, want %v", completed, got, want)
		}
	}
}

func TestClaudeCodeCompactionSummaryDigest(t *testing.T) {
	marker := jsonString(ClaudeCodeContinuationMarker + " The summary.")
	cases := []struct {
		name     string
		messages []string
		want     bool
	}{
		{"string content", []string{`{"role":"user","content":` + marker + `}`}, true},
		{"after reminders", []string{`{"role":"user","content":[{"type":"text","text":"<system-reminder>a</system-reminder>"},{"type":"text","text":` + marker + `}]}`}, true},
		{"leading whitespace", []string{`{"role":"user","content":` + jsonString("\n "+ClaudeCodeContinuationMarker) + `}`}, true},
		{"not the first real block", []string{`{"role":"user","content":[{"type":"text","text":"task"},{"type":"text","text":` + marker + `}]}`}, false},
		{"quoted mid-text", []string{`{"role":"user","content":` + jsonString("about: "+ClaudeCodeContinuationMarker) + `}`}, false},
		{"a later message", []string{`{"role":"user","content":"task"}`, `{"role":"user","content":` + marker + `}`}, false},
		{"an assistant message", []string{`{"role":"assistant","content":` + marker + `}`}, false},
		{"no messages", nil, false},
	}
	for _, c := range cases {
		messages, _ := rawMessages(t, c.messages...)
		if got := ClaudeCodeCompactionSummaryDigest(messages) != ""; got != c.want {
			t.Errorf("%s: detected %v, want %v", c.name, got, c.want)
		}
	}
}

func TestClaudeCodeSubagentClaim(t *testing.T) {
	cases := []struct {
		system string
		want   bool
	}{
		{`[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1; cc_entrypoint=sdk-cli; cc_is_subagent=true;"},{"type":"text","text":"You are..."}]`, true},
		{`"x-anthropic-billing-header: cc_version=2.1; cc_is_subagent=true"`, true},
		{`[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1; cc_entrypoint=sdk-cli;"}]`, false},
		{`[{"type":"text","text":"x-anthropic-billing-header: cc_is_subagent=false"}]`, false},
		{`[{"type":"text","text":"x-anthropic-billing-header: cc_is_subagent=maybe"}]`, false},
		{`[{"type":"text","text":"You are a subagent. cc_is_subagent=true"}]`, false},
		{``, false},
	}
	for _, c := range cases {
		if got := ClaudeCodeSubagentClaim(json.RawMessage(c.system)); got != c.want {
			t.Errorf("%s: claim %v, want %v", c.system, got, c.want)
		}
	}
}

func TestIsCodexCompactionRequest(t *testing.T) {
	directive := jsonString(codexCompactionPrompt)
	user := func(content string) json.RawMessage {
		return json.RawMessage(`{"type":"message","role":"user","content":` + content + `}`)
	}
	task := user(`"task"`)
	cases := []struct {
		name  string
		input []json.RawMessage
		want  bool
	}{
		{"input_text part", []json.RawMessage{task, user(`[{"type":"input_text","text":` + directive + `}]`)}, true},
		{"string content", []json.RawMessage{task, user(directive)}, true},
		{"not last", []json.RawMessage{user(directive), task}, false},
		{"a prefix only", []json.RawMessage{user(jsonString(codexCompactionPrompt[:40]))}, false},
		{"two parts", []json.RawMessage{user(`[{"type":"input_text","text":` + directive + `},{"type":"input_text","text":"x"}]`)}, false},
		{"empty", nil, false},
	}
	for _, c := range cases {
		if got := IsCodexCompactionRequest(c.input); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}
