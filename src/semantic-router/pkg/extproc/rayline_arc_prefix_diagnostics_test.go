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
	"strings"
	"testing"

	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// prefixDiagnosticTurn reads one request against previous as the policy
// selector does and returns the transition and the state it would commit.
func prefixDiagnosticTurn(
	t *testing.T,
	previous *raylinearc.PolicyEpisodeState,
	body map[string]any,
	kind raylinearc.PolicyCallKind,
) (string, *raylinearc.PolicyEpisodeState) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	arcContext := &selection.RaylineARCSelectionContext{
		EpisodeIDHash: "episode", RequestID: "request", RawRequest: raw, RequestFormat: policyFormatAnthropic,
	}
	clientRequest, messages, roles, err := policyClientRequestOf(arcContext)
	if err != nil {
		t.Fatal(err)
	}
	signals := raylinearc.PolicyTurnSignals{CallKind: kind}
	turn, _, transition := raylinearc.PolicyTurn(previous, messages, roles, 0, signals)
	rememberPolicyPrefix(arcContext.EpisodeIDHash,
		observePolicyPrefix(arcContext, clientRequest, raylinearc.NewPrefixDigests(messages), previous, signals, transition))
	if kind == raylinearc.PolicyCallSide {
		return transition, previous
	}
	return transition, turn.Next(messages, "action", "arm")
}

func freshPolicyPrefixShapes(t *testing.T) {
	t.Helper()
	saved := policyPrefixShapes
	policyPrefixShapes = &policyPrefixShapeCache{episodes: map[string][]*raylinearc.PolicyPrefixShape{}}
	t.Cleanup(func() { policyPrefixShapes = saved })
}

func prefixDiagnostics(logs *observer.ObservedLogs) []map[string]interface{} {
	var events []map[string]interface{}
	for _, entry := range logs.FilterMessage("rayline_arc_policy_prefix_diagnostic").All() {
		events = append(events, entry.ContextMap())
	}
	return events
}

var (
	diagSystem = []any{
		map[string]any{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.309.abc; cc_entrypoint=cli;"},
		map[string]any{"type": "text", "text": "You are Claude Code."},
	}
	diagTools = []any{map[string]any{"name": "Read", "input_schema": map[string]any{"type": "object"}}}
)

func diagMainTurn(messages ...any) map[string]any {
	return map[string]any{
		"model": "auto", "stream": true, "max_tokens": 32000,
		"system": diagSystem, "tools": diagTools, "messages": messages,
	}
}

// A small helper request served as a main turn breaks the prefix at its
// first message, and the log says so with the helper's shape; the next main
// turn breaks against the helper's prefix.
func TestPrefixBreakLogsWhereAHelperDiverged(t *testing.T) {
	freshPolicyPrefixShapes(t)
	logs := captureLogs(t)
	first := []any{
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "done"}}},
		map[string]any{"role": "user", "content": "now the tests"},
	}
	transition, state := prefixDiagnosticTurn(t, nil, diagMainTurn(first...), raylinearc.PolicyCallUnknown)
	if transition != raylinearc.PolicyTransitionStart {
		t.Fatalf("first turn = %s", transition)
	}
	helper := map[string]any{
		"model": "auto", "stream": false, "max_tokens": 64,
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "quota"}}}},
	}
	transition, _ = prefixDiagnosticTurn(t, state, helper, raylinearc.PolicyCallUnknown)
	if transition != raylinearc.PolicyTransitionPrefixBreak {
		t.Fatalf("helper = %s, want prefix_break", transition)
	}
	events := prefixDiagnostics(logs)
	if len(events) != 1 {
		t.Fatalf("%d diagnostics, want 1", len(events))
	}
	want := map[string]interface{}{
		"transition": "prefix_break", "call_kind": "unknown", "request_id": "request",
		"previous_message_count": int64(3), "message_count": int64(1), "previous_shape_known": true,
		"first_divergent_index": int64(0), "divergent_role": "user", "divergent_block_type": "text",
		"system_changed": true, "tools_changed": true,
		"stream": false, "max_tokens": int64(64), "has_tools": false, "tool_count": int64(0),
		"claude_code_billing_header": false,
	}
	for key, value := range want {
		if events[0][key] != value {
			t.Errorf("%s = %#v, want %#v", key, events[0][key], value)
		}
	}
}

// A fork that replaced the last user message diverges there; an extending
// turn whose tools changed is logged as extends; one that changed only
// Claude Code's billing header or cache_control is not logged.
func TestPrefixDiagnosticForksAndShellChanges(t *testing.T) {
	freshPolicyPrefixShapes(t)
	logs := captureLogs(t)
	history := []any{
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "done"},
	}
	_, state := prefixDiagnosticTurn(t, nil,
		diagMainTurn(append(history, map[string]any{"role": "user", "content": "[a fork's own question]"})...),
		raylinearc.PolicyCallUnknown)
	transition, state := prefixDiagnosticTurn(t, state,
		diagMainTurn(append(history, map[string]any{"role": "user", "content": "now the tests"})...),
		raylinearc.PolicyCallUnknown)
	events := prefixDiagnostics(logs)
	if transition != raylinearc.PolicyTransitionPrefixBreak || len(events) != 1 ||
		events[0]["first_divergent_index"] != int64(2) || events[0]["system_changed"] != false ||
		events[0]["claude_code_billing_header"] != true || events[0]["has_tools"] != true {
		t.Fatalf("fork: %s %v", transition, events)
	}

	reheadered := diagMainTurn(append(history,
		map[string]any{"role": "user", "content": "now the tests"},
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "and lint"})...)
	reheadered["system"] = []any{
		map[string]any{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.309.f00; cc_entrypoint=cli;"},
		map[string]any{"type": "text", "text": "You are Claude Code.", "cache_control": map[string]any{"type": "ephemeral"}},
	}
	transition, state = prefixDiagnosticTurn(t, state, reheadered, raylinearc.PolicyCallUnknown)
	if transition != raylinearc.PolicyTransitionExtends || len(prefixDiagnostics(logs)) != 1 {
		t.Fatalf("billing header change: %s, %d diagnostics", transition, len(prefixDiagnostics(logs)))
	}

	retooled := diagMainTurn(append(history,
		map[string]any{"role": "user", "content": "now the tests"},
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "and lint"},
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "ship it"})...)
	retooled["tools"] = []any{map[string]any{"name": "Write", "input_schema": map[string]any{"type": "object"}}}
	transition, _ = prefixDiagnosticTurn(t, state, retooled, raylinearc.PolicyCallUnknown)
	events = prefixDiagnostics(logs)
	if transition != raylinearc.PolicyTransitionExtends || len(events) != 2 ||
		events[1]["transition"] != "extends" || events[1]["tools_changed"] != true ||
		events[1]["system_changed"] != false || events[1]["first_divergent_index"] != int64(5) {
		t.Fatalf("tools change: %s %v", transition, events)
	}
}

// A side call is neither logged nor remembered, so the main turn after it is
// compared with the main turn before it. A previous request this process did
// not see leaves the divergence unknown.
func TestPrefixDiagnosticSkipsSideCallsAndUnknownPrefixes(t *testing.T) {
	freshPolicyPrefixShapes(t)
	logs := captureLogs(t)
	first := []any{map[string]any{"role": "user", "content": "fix the bug"}}
	_, state := prefixDiagnosticTurn(t, nil, diagMainTurn(first...), raylinearc.PolicyCallUnknown)
	side := diagMainTurn(map[string]any{"role": "user", "content": "title this"})
	if _, after := prefixDiagnosticTurn(t, state, side, raylinearc.PolicyCallSide); after != state {
		t.Fatal("a side call changed the episode")
	}
	if events := prefixDiagnostics(logs); len(events) != 0 {
		t.Fatalf("side call logged %v", events)
	}
	policyPrefixShapes = &policyPrefixShapeCache{episodes: map[string][]*raylinearc.PolicyPrefixShape{}}
	transition, _ := prefixDiagnosticTurn(t, state,
		diagMainTurn(map[string]any{"role": "user", "content": "a rewritten history"}), raylinearc.PolicyCallMain)
	events := prefixDiagnostics(logs)
	if transition != raylinearc.PolicyTransitionPrefixBreak || len(events) != 1 ||
		events[0]["previous_shape_known"] != false || events[0]["first_divergent_index"] != nil ||
		events[0]["tools_changed"] != nil || events[0]["max_tokens"] != int64(32000) {
		t.Fatalf("unknown previous: %s %v", transition, events)
	}
}

func TestPolicyTokenSplit(t *testing.T) {
	for _, c := range []struct {
		action             string
		unknown            bool
		retained, appended interface{}
	}{
		{"created", false, 0, 900},
		{"rebuilt", false, 0, 900},
		{"reused", false, 900, 0},
		{"appended", false, nil, nil},
		{"rebuilt", true, nil, nil},
	} {
		retained, appended := policyTokenSplit(&selection.RaylineARCTrace{
			SessionAction: c.action, SerializedTokens: 900, TokenCountsUnknown: c.unknown,
		})
		if retained != c.retained || appended != c.appended {
			t.Errorf("%s (unknown %v) = %v/%v, want %v/%v", c.action, c.unknown, retained, appended, c.retained, c.appended)
		}
	}
}

// The cache forgets the oldest episode past its bound and keeps the last two
// shapes of each.
func TestPolicyPrefixShapeCacheBounds(t *testing.T) {
	cache := &policyPrefixShapeCache{episodes: map[string][]*raylinearc.PolicyPrefixShape{}}
	for index := range policyPrefixShapeEpisodes + 1 {
		cache.remember(string(rune('a'+index%26))+string(rune(index)), &raylinearc.PolicyPrefixShape{})
	}
	if len(cache.episodes) != policyPrefixShapeEpisodes || len(cache.order) != policyPrefixShapeEpisodes {
		t.Fatalf("episodes = %d, order = %d", len(cache.episodes), len(cache.order))
	}
	for index := range 3 {
		cache.remember("one", &raylinearc.PolicyPrefixShape{Len: index})
	}
	if shapes := cache.episodes["one"]; len(shapes) != 2 || shapes[0].Len != 1 || shapes[1].Len != 2 {
		t.Fatalf("shapes = %v", shapes)
	}
}

// Two remembered requests with the same transcript and different tools: the
// newer one is compared, so a later turn with the newer tools is unchanged.
func TestPrefixDiagnosticComparesTheNewestMatchingShape(t *testing.T) {
	freshPolicyPrefixShapes(t)
	logs := captureLogs(t)
	first := []any{map[string]any{"role": "user", "content": "fix the bug"}}
	_, state := prefixDiagnosticTurn(t, nil, diagMainTurn(first...), raylinearc.PolicyCallUnknown)
	retried := diagMainTurn(first...)
	retried["tools"] = []any{map[string]any{"name": "Write", "input_schema": map[string]any{"type": "object"}}}
	raw, _ := json.Marshal(retried)
	arcContext := &selection.RaylineARCSelectionContext{EpisodeIDHash: "episode", RawRequest: raw, RequestFormat: policyFormatAnthropic}
	clientRequest, messages, _, err := policyClientRequestOf(arcContext)
	if err != nil {
		t.Fatal(err)
	}
	policyPrefixShapes.remember("episode", raylinearc.NewPolicyPrefixShape(messages, clientRequest.System, clientRequest.Tools))
	next := diagMainTurn(append(first,
		map[string]any{"role": "assistant", "content": "done"},
		map[string]any{"role": "user", "content": "now the tests"})...)
	next["tools"] = retried["tools"]
	if transition, _ := prefixDiagnosticTurn(t, state, next, raylinearc.PolicyCallUnknown); transition != raylinearc.PolicyTransitionExtends {
		t.Fatalf("transition = %s", transition)
	}
	if events := prefixDiagnostics(logs); len(events) != 0 {
		t.Fatalf("compared with the older tools: %v", events)
	}
}

// Through the selector: a request whose decide failed was never encoded, so
// it is not remembered; one the service decided is.
func TestPolicySelectorRemembersOnlyDecidedShapes(t *testing.T) {
	freshPolicyPrefixShapes(t)
	fixture := newSideCallFixture(t, config.RaylineARCConsistencyStrict)
	fixture.fake.failNext("not_ready")
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallMain); err == nil {
		t.Fatal("a failed decide selected")
	}
	if len(policyPrefixShapes.episodes) != 0 {
		t.Fatalf("a failed decide was remembered: %v", policyPrefixShapes.episodes)
	}
	fixture.fake.failNext("")
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallMain); err != nil {
		t.Fatal(err)
	}
	if shapes := policyPrefixShapes.episodes[strings.Repeat("e", 64)]; len(shapes) != 1 {
		t.Fatalf("decided request remembered %d shapes, want 1", len(shapes))
	}
}

// A rewritten earlier user message is described on both sides by layout
// only: the form, block types, lengths, cache breakpoints and short text
// digests show what changed, and no text reaches the log.
func TestPrefixBreakLogsTheDivergentMessageLayouts(t *testing.T) {
	freshPolicyPrefixShapes(t)
	logs := captureLogs(t)
	const prompt = "fix the secret bug"
	reminder := map[string]any{"type": "text", "text": "<system-reminder>ide context</system-reminder>"}
	history := []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": "hi"},
	}
	asSent := map[string]any{"role": "user", "content": []any{
		reminder,
		map[string]any{"type": "text", "text": prompt, "cache_control": map[string]any{"type": "ephemeral"}},
	}}
	_, state := prefixDiagnosticTurn(t, nil, diagMainTurn(append(history, asSent,
		map[string]any{"role": "system", "content": "<system-reminder>r</system-reminder>"})...), raylinearc.PolicyCallUnknown)
	transition, _ := prefixDiagnosticTurn(t, state, diagMainTurn(append(history,
		map[string]any{"role": "user", "content": prompt},
		map[string]any{"role": "system", "content": "<system-reminder>r</system-reminder>"},
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "go on"})...), raylinearc.PolicyCallUnknown)
	events := prefixDiagnostics(logs)
	if transition != raylinearc.PolicyTransitionPrefixBreak || len(events) != 1 || events[0]["first_divergent_index"] != int64(2) {
		t.Fatalf("%s %v", transition, events)
	}
	raw, err := json.Marshal(events[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "secret") || strings.Contains(string(raw), "ide context") {
		t.Fatalf("the diagnostic carries text: %s", raw)
	}
	var logged struct {
		Current  layoutFields `json:"divergent_layout"`
		Previous layoutFields `json:"previous_divergent_layout"`
	}
	if err := json.Unmarshal(raw, &logged); err != nil {
		t.Fatal(err)
	}
	if logged.Previous.Form != "list" || logged.Previous.Count != 2 || len(logged.Previous.Blocks) != 2 ||
		!logged.Previous.Blocks[0].Reminder || logged.Previous.Blocks[0].CacheControl ||
		!logged.Previous.Blocks[1].CacheControl || logged.Previous.Blocks[1].TextLen != len(prompt) {
		t.Fatalf("previous layout = %+v", logged.Previous)
	}
	if logged.Current.Role != "user" || logged.Current.Form != "string" || logged.Current.Count != 1 ||
		logged.Current.Blocks[0].TextLen != len(prompt) || logged.Current.Blocks[0].Reminder ||
		logged.Current.Blocks[0].SHA != logged.Previous.Blocks[1].SHA || len(logged.Current.Blocks[0].SHA) != 8 {
		t.Fatalf("current layout = %+v", logged.Current)
	}
}

type layoutFields struct {
	Role   string `json:"role"`
	Form   string `json:"content_form"`
	Count  int    `json:"block_count"`
	Blocks []struct {
		Type         string   `json:"type"`
		Keys         []string `json:"keys"`
		TextLen      int      `json:"text_len"`
		SHA          string   `json:"text_sha256_8"`
		CacheControl bool     `json:"cache_control"`
		Reminder     bool     `json:"system_reminder"`
	} `json:"blocks"`
}
