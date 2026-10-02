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
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func (fixture *policySelectorFixture) selectWithHeaders(
	t *testing.T,
	state *raylinearc.EpisodeState,
	body []byte,
	headers map[string]string,
) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash:       strings.Repeat("e", 64),
			State:               state,
			RawRequest:          body,
			RequestFormat:       policyFormatAnthropic,
			PolicySignalHeaders: headers,
		},
	})
}

// heldEpisode is an episode on the think worker two turns in, between the
// schedule's boundaries, whose stored prefix is the one-message task.
func heldEpisode(t *testing.T) *raylinearc.EpisodeState {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	task := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	var body struct{ Messages []json.RawMessage }
	if err := json.Unmarshal(task, &body); err != nil {
		t.Fatal(err)
	}
	state.Policy = (&raylinearc.PolicyEpisodeState{}).Next(body.Messages, strings.Repeat("a", 64), "arm-a")
	return state
}

func offeredActions(fixture *policySelectorFixture, call int) []string {
	return fixture.fake.received()[call].Selection.AvailableActionIDs
}

// A transcript rewrite starts a new context epoch but is not a compaction:
// between boundaries the model stays held.
func TestPolicySelectorPrefixBreakKeepsTheModelHeld(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state := heldEpisode(t)
	rewritten := policyTestRequest(t, map[string]any{"role": "user", "content": "summary of earlier work"})
	result, err := fixture.selectWithHeaders(t, state, rewritten, nil)
	if err != nil {
		t.Fatal(err)
	}
	sent := fixture.fake.received()[0]
	if sent.ContextEpoch != "1" || len(sent.Attribution) != 0 || len(offeredActions(fixture, 0)) != 2 {
		t.Fatalf("prefix break: epoch %q attribution %v offered %v", sent.ContextEpoch, sent.Attribution, offeredActions(fixture, 0))
	}
	if next := result.RaylineARC.PolicyNextState; next.EpochStartTurn != 0 || next.CompactionCount != 0 {
		t.Fatalf("prefix break moved the compaction clock: %+v", next)
	}
}

// An explicit compaction -- the header, or Claude Code's continuation summary
// -- opens a boundary at this turn.
func TestPolicySelectorExplicitCompactionOpensABoundary(t *testing.T) {
	cases := []struct {
		name    string
		content string
		headers map[string]string
	}{
		{"header", "compacted context", map[string]string{raylineARCCompactionHeader: "1"}},
		{"claude code summary", raylinearc.ClaudeCodeContinuationMarker + " Summary.", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
			fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
				return request.Selection.AvailableActionIDs[0]
			})
			body := policyTestRequest(t, map[string]any{"role": "user", "content": c.content})
			result, err := fixture.selectWithHeaders(t, heldEpisode(t), body, c.headers)
			if err != nil {
				t.Fatal(err)
			}
			if sent := fixture.fake.received()[0]; sent.ContextEpoch != "1" || len(offeredActions(fixture, 0)) != 3 {
				t.Fatalf("compaction: epoch %q offered %v", sent.ContextEpoch, offeredActions(fixture, 0))
			}
			if next := result.RaylineARC.PolicyNextState; next.EpochStartTurn != 2 || next.CompactionCount != 1 {
				t.Fatalf("compaction clock: %+v", next)
			}
		})
	}
}

// A side call at a schedule boundary still keeps the held model, sees the
// current epoch, and stages nothing to commit.
func TestPolicySelectorSideCallKeepsTheHeldArm(t *testing.T) {
	subagentSystem := []map[string]any{{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cc_is_subagent=true;"}}
	cases := []struct {
		name    string
		body    func(*testing.T) []byte
		headers map[string]string
	}{
		{"header", func(t *testing.T) []byte {
			return policyTestRequest(t, map[string]any{"role": "user", "content": "write a title"})
		}, map[string]string{raylineARCCallKindHeader: "side"}},
		{"claude code subagent", func(t *testing.T) []byte {
			body, err := json.Marshal(map[string]any{
				"model": "auto", "max_tokens": 1024, "system": subagentSystem,
				"messages": []map[string]any{{"role": "user", "content": "search the repo"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			return body
		}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
			fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
				return request.Selection.AvailableActionIDs[0]
			})
			state := heldEpisode(t)
			state.TurnIndex = 5
			before := cloneARCState(state)
			result, err := fixture.selectWithHeaders(t, state, c.body(t), c.headers)
			if err != nil {
				t.Fatal(err)
			}
			if sent := fixture.fake.received()[0]; sent.ContextEpoch != "0" || len(offeredActions(fixture, 0)) != 2 {
				t.Fatalf("side call: epoch %q offered %v", sent.ContextEpoch, offeredActions(fixture, 0))
			}
			if !result.RaylineARC.PolicySideCall || result.RaylineARC.PolicyNextState != nil ||
				!reflect.DeepEqual(state, before) {
				t.Fatalf("side call staged state: side=%v next=%+v", result.RaylineARC.PolicySideCall, result.RaylineARC.PolicyNextState)
			}
		})
	}
}

// A completed side call leaves the stored episode exactly as it was: no turn
// counted, no ledger entry, the previous arm unchanged, and the lease free.
func TestRaylineARCSideCallCommitsNothing(t *testing.T) {
	store, episode := newLedgerTestStore(t)
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
		completeStream(router, ctx)
	})
	before := readLedgerTestState(t, store, episode)

	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		Headers:               map[string]string{},
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
	}
	ctx.RaylineARCTransaction.markSelection(0, 50)
	ctx.RaylineARCTransaction.markPolicyState(nil, true)
	bindRaylineARCSelectionTransaction(ctx)
	router := &OpenAIRouter{}
	sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
	completeStream(router, ctx)
	finalizeSelectionProcessTerminal(ctx)

	after := readLedgerTestStateNoWait(store, episode)
	if after == nil {
		t.Fatal("the side call kept the episode lease")
	}
	if after.TurnIndex != before.TurnIndex || *after.PreviousArm != *before.PreviousArm ||
		!reflect.DeepEqual(after.Policy, before.Policy) {
		t.Fatalf("side call changed the episode: before turn=%d arm=%d after turn=%d arm=%d",
			before.TurnIndex, *before.PreviousArm, after.TurnIndex, *after.PreviousArm)
	}
}

func TestRaylineARCPolicyTurnSignalsReadHeadersAndWire(t *testing.T) {
	marker := json.RawMessage(`{"role":"user","content":` +
		mustJSON(t, raylinearc.ClaudeCodeContinuationMarker+" S.") + `}`)
	plain := json.RawMessage(`{"role":"user","content":"go"}`)
	subagent := json.RawMessage(`[{"type":"text","text":"x-anthropic-billing-header: cc_is_subagent=true"}]`)
	cases := []struct {
		name        string
		headers     map[string]string
		system      json.RawMessage
		messages    []json.RawMessage
		kind        raylinearc.PolicyCallKind
		source      string
		ordinal     int
		summaryRead bool
	}{
		{"nothing", nil, nil, []json.RawMessage{plain}, raylinearc.PolicyCallUnknown, "", 0, false},
		{"main header", map[string]string{raylineARCCallKindHeader: "Main"}, nil, []json.RawMessage{plain}, raylinearc.PolicyCallMain, "header", 0, false},
		{"side header", map[string]string{raylineARCCallKindHeader: "side"}, nil, []json.RawMessage{plain}, raylinearc.PolicyCallSide, "header", 0, false},
		{"bad header is unknown", map[string]string{raylineARCCallKindHeader: "helper"}, nil, []json.RawMessage{plain}, raylinearc.PolicyCallUnknown, "", 0, false},
		{"header overrides subagent claim", map[string]string{raylineARCCallKindHeader: "main"}, subagent, []json.RawMessage{plain}, raylinearc.PolicyCallMain, "header", 0, false},
		{"subagent claim", nil, subagent, []json.RawMessage{plain}, raylinearc.PolicyCallSide, callKindSourceClaudeSubagent, 0, false},
		{"compaction ordinal", map[string]string{raylineARCCompactionHeader: "2"}, nil, []json.RawMessage{plain}, raylinearc.PolicyCallUnknown, "", 2, false},
		{"bad compaction ordinal", map[string]string{raylineARCCompactionHeader: "0"}, nil, []json.RawMessage{plain}, raylinearc.PolicyCallUnknown, "", 0, false},
		{"continuation summary", nil, nil, []json.RawMessage{marker}, raylinearc.PolicyCallUnknown, "", 0, true},
	}
	for _, c := range cases {
		signals := raylineARCPolicyTurnSignals(c.headers, policyFormatAnthropic,
			raylinearc.PolicyClientRequest{System: c.system}, c.messages, "episode")
		ordinal, summaryRead := 0, false
		if signals.Compaction != nil {
			ordinal, summaryRead = signals.Compaction.Ordinal, signals.Compaction.SummaryDigest != ""
		}
		if signals.CallKind != c.kind || signals.CallKindSource != c.source ||
			ordinal != c.ordinal || summaryRead != c.summaryRead {
			t.Errorf("%s: %+v (ordinal %d, summary %v)", c.name, signals, ordinal, summaryRead)
		}
	}
	// The continuation literal is Claude Code's; another format never reads it.
	if signals := raylineARCPolicyTurnSignals(nil, policyFormatOpenAI, raylinearc.PolicyClientRequest{},
		[]json.RawMessage{marker}, "episode"); signals.Compaction != nil {
		t.Fatalf("openai_chat read the Claude Code summary: %+v", signals)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
