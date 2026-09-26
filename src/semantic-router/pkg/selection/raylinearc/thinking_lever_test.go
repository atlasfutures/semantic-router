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
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const (
	steerDown = "Until the next steering instruction, use minimal deliberation."
	steerUp   = "Until the next steering instruction, reason more thoroughly."
)

func suffixBinding(emit ThinkingEmitMode, neutral string) ThinkingBinding {
	binding := ThinkingBinding{
		Lever:      ThinkingLeverSteeringSuffix,
		Emit:       emit,
		Neutral:    neutral,
		Placements: []ThinkingPlacement{ThinkingPlaceAppendTailUserText, ThinkingPlaceUserAfterToolRun},
		Levels: []ThinkingLevel{
			{Name: "none", Rank: 0},
			{Name: "down", Rank: -1, Suffix: steerDown},
			{Name: "up", Rank: 1, Suffix: steerUp},
		},
	}
	if neutral == "neutral" {
		binding.Levels = append(binding.Levels, ThinkingLevel{
			Name: "neutral", Rank: 0, Suffix: "Until the next steering instruction, use your normal judgement.",
		})
	}
	return binding
}

func effortBinding() ThinkingBinding {
	return ThinkingBinding{
		Lever:      ThinkingLeverPerTurnEffort,
		Emit:       ThinkingEmitEveryTurn,
		Neutral:    "base",
		Placements: []ThinkingPlacement{ThinkingPlaceSystemBeforeTurn, ThinkingPlaceSystemAfterToolRun},
		Levels: []ThinkingLevel{
			{Name: "low", Rank: -1, Effort: "low"},
			{Name: "base", Rank: 0, Effort: "high"},
			{Name: "max", Rank: 1, Effort: "max"},
		},
	}
}

func text(role llmprotocol.Role, value string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
}

func toolCall(id string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: "bash", Arguments: `{"cmd":"ls"}`},
	}}}
}

func toolResult(id string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID:  id,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ok"}},
		},
	}}}
}

// episode drives one conversation the way the router does: plan, apply,
// and commit the staged ledger only for a turn that "succeeded".
type episode struct {
	t       *testing.T
	binding ThinkingBinding
	ledger  *ThinkingLedger
	turn    uint64
	spacing uint64
}

func (e *episode) serve(messages []llmprotocol.Message, level string) (ThinkingPlan, []llmprotocol.Message) {
	e.t.Helper()
	plan, err := PlanThinkingTurn(ThinkingTurn{
		Binding: e.binding, Ledger: e.ledger, Messages: ThinkingMessages(messages),
		TurnIndex: e.turn, Requested: level, MinTurnsBetweenChanges: e.spacing,
	})
	if err != nil {
		e.t.Fatalf("plan turn %d: %v", e.turn, err)
	}
	provider, err := ApplyThinkingLedger(messages, e.binding, plan.Next)
	if err != nil {
		e.t.Fatalf("apply turn %d: %v", e.turn, err)
	}
	return plan, provider
}

func (e *episode) commit(plan ThinkingPlan) {
	next := plan.Next
	e.ledger = &next
	e.turn++
}

func chatMessages(t *testing.T, messages []llmprotocol.Message) []json.RawMessage {
	t.Helper()
	request := llmprotocol.Request{Generation: 1, Model: "m", Messages: messages}
	body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode chat: %v", err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode chat: %v", err)
	}
	return wire.Messages
}

// requireExtension is the invariant the whole lever rests on: what the
// provider saw last turn is a byte prefix of what it sees this turn.
func requireExtension(t *testing.T, previous, current []llmprotocol.Message) {
	t.Helper()
	before, after := chatMessages(t, previous), chatMessages(t, current)
	if len(after) < len(before) {
		t.Fatalf("provider transcript shrank from %d to %d messages", len(before), len(after))
	}
	for index := range before {
		if !bytes.Equal(before[index], after[index]) {
			t.Fatalf("provider message %d changed:\nbefore %s\nafter  %s", index, before[index], after[index])
		}
	}
}

func TestEffortControlDigestsMatchTheExportContract(t *testing.T) {
	// Vectors from the pathfinder export: sha256 of the RFC 8785 bytes of
	// the configuration_update value.
	want := map[string]string{
		"low":    "7e33466b523b791e8c21d36aecdc09a8815a733b810f6bd202ff2601e3c97463",
		"medium": "49e511e263eb52447fda59ae97133853113a92a78448d9d071ffdc0bfc789f1f",
		"high":   "b9e2def14575ff37b827abcb049dc5e45321c25f54d1fad1676585a6e324fcaf",
		"xhigh":  "1d4cb4a9413193fb2f90619dc4c01acf4cac6ebb7c0811852b314d11de0af260",
		"max":    "dd1bcd0bcc5b77573de1a2d9bf2996847c298c28b82f0f95fc6e60f638343b37",
	}
	binding := effortBinding()
	for effort, digest := range want {
		if got := binding.ControlSHA256(ThinkingLevel{Name: effort, Effort: effort}); got != digest {
			t.Fatalf("control digest for %s = %s, want %s", effort, got, digest)
		}
	}
}

func TestThinkingBindingValidation(t *testing.T) {
	valid := suffixBinding(ThinkingEmitOnChange, "")
	if err := valid.Validate(); err != nil {
		t.Fatalf("a ladder without a neutral rung is valid: %v", err)
	}
	cases := map[string]func(*ThinkingBinding){
		"placement for the other lever": func(b *ThinkingBinding) {
			b.Placements = []ThinkingPlacement{ThinkingPlaceSystemBeforeTurn}
		},
		"no placement":        func(b *ThinkingBinding) { b.Placements = nil },
		"undeclared neutral":  func(b *ThinkingBinding) { b.Neutral = "missing" },
		"duplicate level":     func(b *ThinkingBinding) { b.Levels = append(b.Levels, b.Levels[1]) },
		"blank suffix":        func(b *ThinkingBinding) { b.Levels[1].Suffix = "  " },
		"effort on a suffix":  func(b *ThinkingBinding) { b.Levels[1].Effort = "low" },
		"single level ladder": func(b *ThinkingBinding) { b.Levels = b.Levels[:1] },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			binding := suffixBinding(ThinkingEmitOnChange, "")
			mutate(&binding)
			if binding.Validate() == nil {
				t.Fatal("binding validated")
			}
		})
	}
	effort := effortBinding()
	effort.Levels[0].Effort = "Low"
	if effort.Validate() == nil {
		t.Fatal("a non-canonical effort name validated")
	}
}

func TestSuffixAppendsToUserTailAndReplaysAsAnExtension(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	turn0 := []llmprotocol.Message{text(llmprotocol.RoleUser, "fix the bug")}
	plan, provider0 := e.serve(turn0, "down")
	if !plan.Emitted || plan.Placement != ThinkingPlaceAppendTailUserText || plan.LevelInForce != "down" {
		t.Fatalf("turn 0 plan = %+v", plan)
	}
	if got := provider0[0].Content[1].Text; got != steerDown {
		t.Fatalf("appended text = %q", got)
	}
	if len(turn0[0].Content) != 1 {
		t.Fatal("the client transcript was modified")
	}
	e.commit(plan)

	turn1 := append(append([]llmprotocol.Message(nil), turn0...), toolCall("c1"), toolResult("c1"))
	plan, provider1 := e.serve(turn1, "down")
	if plan.Emitted || plan.Replayed != 1 || plan.LevelInForce != "down" {
		t.Fatalf("turn 1 plan = %+v", plan)
	}
	requireExtension(t, provider0, provider1)
}

func TestSuffixAfterToolRunInsertsAUserMessage(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go"), toolCall("c1"), toolResult("c1")}
	plan, provider := e.serve(messages, "up")
	if plan.Placement != ThinkingPlaceUserAfterToolRun || len(provider) != 4 {
		t.Fatalf("plan = %+v, provider messages = %d", plan, len(provider))
	}
	if provider[3].Role != llmprotocol.RoleUser || provider[3].Content[0].Text != steerUp {
		t.Fatalf("inserted message = %+v", provider[3])
	}
}

func TestPerTurnEffortStatesTheLevelEveryTurn(t *testing.T) {
	e := &episode{t: t, binding: effortBinding()}
	turn0 := []llmprotocol.Message{text(llmprotocol.RoleUser, "plan it")}
	plan, provider0 := e.serve(turn0, "max")
	if plan.Placement != ThinkingPlaceSystemBeforeTurn || provider0[0].Configuration == nil ||
		provider0[0].Configuration.ReasoningEffort != "max" {
		t.Fatalf("turn 0 plan = %+v, first message = %+v", plan, provider0[0])
	}
	e.commit(plan)

	turn1 := append(append([]llmprotocol.Message(nil), turn0...), toolCall("c1"), toolResult("c1"))
	plan, provider1 := e.serve(turn1, "max")
	if !plan.Emitted || plan.Placement != ThinkingPlaceSystemAfterToolRun {
		t.Fatalf("turn 1 plan = %+v", plan)
	}
	requireExtension(t, provider0, provider1)
	last := provider1[len(provider1)-1]
	if last.Configuration == nil || last.Configuration.ReasoningEffort != "max" {
		t.Fatalf("turn 1 tail = %+v", last)
	}
	e.commit(plan)

	turn2 := append(append([]llmprotocol.Message(nil), turn1...), text(llmprotocol.RoleAssistant, "done"), text(llmprotocol.RoleUser, "next"))
	plan, provider2 := e.serve(turn2, "low")
	requireExtension(t, provider1, provider2)
	if plan.LevelInForce != "low" || len(plan.Next.Entries) != 3 {
		t.Fatalf("turn 2 plan = %+v", plan)
	}
}

func TestRetryOfACommittedTurnReusesItsItem(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
	plan, first := e.serve(messages, "down")
	e.commit(plan)
	// The client never saw the response and sends the same turn again, now
	// with the policy asking for another level.
	plan, second := e.serve(messages, "up")
	if !plan.Retry || plan.Emitted || plan.LevelInForce != "down" || len(plan.Next.Entries) != 1 {
		t.Fatalf("retry plan = %+v", plan)
	}
	requireExtension(t, first, second)
	requireExtension(t, second, first)
}

func TestHistoryNoiseDoesNotResetButARewriteDoes(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	turn0 := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
	plan, _ := e.serve(turn0, "down")
	e.commit(plan)

	noisy := []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentText, Text: "go", Cache: &llmprotocol.CacheDirective{}},
		{Kind: llmprotocol.ContentText, Text: "<system-reminder>todo list changed</system-reminder>"},
	}}, text(llmprotocol.RoleAssistant, "ok"), text(llmprotocol.RoleUser, "more")}
	plan, _ = e.serve(noisy, "down")
	if plan.ResetReason != "" || plan.Replayed != 1 {
		t.Fatalf("noise reset the ledger: %+v", plan)
	}
	e.commit(plan)

	compacted := []llmprotocol.Message{text(llmprotocol.RoleUser, "summary of earlier work"), text(llmprotocol.RoleUser, "continue")}
	plan, provider := e.serve(compacted, "down")
	if plan.ResetReason != ThinkingResetTranscriptRewrite || plan.Next.Epoch != 1 {
		t.Fatalf("rewrite plan = %+v", plan)
	}
	// The level in force is re-asserted at the new tail.
	if !plan.Emitted || provider[1].Content[len(provider[1].Content)-1].Text != steerDown {
		t.Fatalf("level not re-asserted after the reset: %+v", plan)
	}
}

func TestBindingChangeResetsTheLedger(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
	plan, _ := e.serve(messages, "down")
	e.commit(plan)
	e.binding.Levels[1].Suffix = "A different instruction."
	next := append(append([]llmprotocol.Message(nil), messages...), text(llmprotocol.RoleAssistant, "a"), text(llmprotocol.RoleUser, "b"))
	plan, _ = e.serve(next, "down")
	if plan.ResetReason != ThinkingResetBindingChanged {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestReturnToNeutralNeedsANeutralRung(t *testing.T) {
	grow := func(messages []llmprotocol.Message) []llmprotocol.Message {
		return append(append([]llmprotocol.Message(nil), messages...), text(llmprotocol.RoleAssistant, "a"), text(llmprotocol.RoleUser, "b"))
	}
	t.Run("without one the instruction stays in force", func(t *testing.T) {
		e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
		messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
		plan, _ := e.serve(messages, "down")
		e.commit(plan)
		plan, _ = e.serve(grow(messages), "none")
		if plan.Emitted || plan.Skipped != ThinkingSkipNeutralInexpressible || plan.LevelInForce != "down" {
			t.Fatalf("plan = %+v", plan)
		}
	})
	t.Run("with one it is written", func(t *testing.T) {
		e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "neutral")}
		messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
		plan, _ := e.serve(messages, "down")
		e.commit(plan)
		plan, _ = e.serve(grow(messages), "neutral")
		if !plan.Emitted || plan.LevelInForce != "neutral" {
			t.Fatalf("plan = %+v", plan)
		}
	})
}

func TestChangeSpacingHoldsTheLevel(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, ""), spacing: 3}
	messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "go")}
	plan, _ := e.serve(messages, "down")
	e.commit(plan)
	messages = append(messages, text(llmprotocol.RoleAssistant, "a"), text(llmprotocol.RoleUser, "b"))
	plan, _ = e.serve(messages, "up")
	if plan.Emitted || plan.Skipped != ThinkingSkipChangeTooSoon || plan.LevelInForce != "down" {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestUnsteerableTailsWriteNothing(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(ThinkingEmitOnChange, "")}
	plan, provider := e.serve([]llmprotocol.Message{text(llmprotocol.RoleUser, "go"), text(llmprotocol.RoleAssistant, "prefill")}, "up")
	if plan.Emitted || plan.Skipped != ThinkingSkipTailNotSteerable || len(provider) != 2 {
		t.Fatalf("plan = %+v", plan)
	}
	e.binding.Placements = []ThinkingPlacement{ThinkingPlaceAppendTailUserText}
	plan, _ = e.serve([]llmprotocol.Message{text(llmprotocol.RoleUser, "go"), toolCall("c1"), toolResult("c1")}, "up")
	if plan.Emitted || plan.Skipped != ThinkingSkipPlacementRefused {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestLedgerOverflowStartsANewEpoch(t *testing.T) {
	binding := effortBinding()
	messages := []llmprotocol.Message{text(llmprotocol.RoleUser, "q0")}
	ledger := &ThinkingLedger{BindingSHA256: binding.SHA256(), LevelInForce: "base"}
	for turn := 0; turn < maxThinkingLedgerLength; turn++ {
		plan, err := PlanThinkingTurn(ThinkingTurn{
			Binding: binding, Ledger: ledger, Messages: ThinkingMessages(messages),
			TurnIndex: uint64(turn), Requested: "max",
		})
		if err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		next := plan.Next
		ledger = &next
		messages = append(messages, text(llmprotocol.RoleAssistant, "a"), text(llmprotocol.RoleUser, "q"))
	}
	plan, err := PlanThinkingTurn(ThinkingTurn{
		Binding: binding, Ledger: ledger, Messages: ThinkingMessages(messages),
		TurnIndex: maxThinkingLedgerLength, Requested: "max",
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.ResetReason != ThinkingResetOverflow || len(plan.Next.Entries) != 1 || plan.Next.Epoch != 1 {
		t.Fatalf("overflow plan: reset %q entries %d epoch %d", plan.ResetReason, len(plan.Next.Entries), plan.Next.Epoch)
	}
}
