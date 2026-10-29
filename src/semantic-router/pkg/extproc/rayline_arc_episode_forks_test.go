package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const forkTestEpisode = "user-1:conversation-1"

type forkCell struct {
	t         *testing.T
	router    *OpenAIRouter
	algorithm *config.AlgorithmConfig
}

func newForkCell(t *testing.T) *forkCell {
	t.Helper()
	router, _, algorithm := missingSessionRequestContext(t, "")
	algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "http://policy.invalid"}
	// Long enough that a request waiting out a parallel turn is visible.
	algorithm.RaylineARC.Episode.AcquireTimeoutSeconds = 3
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 64, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	router.RaylineARCEpisodeStore = store
	return &forkCell{t: t, router: router, algorithm: algorithm}
}

// chatBody is an OpenAI chat body of alternating user/assistant texts.
// A "user:" prefix forces a user message.
func chatBody(t *testing.T, turns ...string) []byte {
	t.Helper()
	messages := []map[string]string{{"role": "system", "content": "you are helpful"}}
	for index, text := range turns {
		role := "user"
		if index%2 == 1 {
			role = "assistant"
		}
		if len(text) > 5 && text[:5] == "user:" {
			role, text = "user", text[5:]
		}
		messages = append(messages, map[string]string{"role": role, "content": text})
	}
	body, err := json.Marshal(map[string]any{"model": "auto", "messages": messages})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func (cell *forkCell) request(body []byte, format llmprotocol.WireFormat) *RequestContext {
	return &RequestContext{
		Headers:      map[string]string{testEpisodeIDHeader: forkTestEpisode},
		SourceFormat: format,
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "public test turn"}},
			}},
		},
		RaylineARCRawBody: body,
		TraceContext:      context.Background(),
	}
}

type forkTurn struct {
	arc     *selection.RaylineARCSelectionContext
	ctx     *RequestContext
	elapsed time.Duration
}

// start prepares a turn and leaves it in flight.
func (cell *forkCell) start(turns ...string) forkTurn {
	cell.t.Helper()
	turn := cell.startBody(chatBody(cell.t, turns...), llmprotocol.OpenAIChatV1)
	if turn.arc.PreparationFailure != "" {
		cell.t.Fatalf("turn %v: failure %q", turns, turn.arc.PreparationFailure)
	}
	return turn
}

func (cell *forkCell) startBody(body []byte, format llmprotocol.WireFormat) forkTurn {
	ctx := cell.request(body, format)
	began := time.Now()
	arc := cell.router.buildRaylineARCSelectionContext(cell.algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
	return forkTurn{arc: arc, ctx: ctx, elapsed: time.Since(began)}
}

// finish ends the turn and releases its lease.
func (turn forkTurn) finish(t *testing.T) {
	t.Helper()
	if turn.ctx.RaylineARCTransaction != nil {
		_ = turn.ctx.RaylineARCTransaction.abort(context.Background(), "test")
	}
}

// run prepares and finishes a turn.
func (cell *forkCell) run(turns ...string) forkTurn {
	cell.t.Helper()
	turn := cell.start(turns...)
	turn.finish(cell.t)
	return turn
}

var forkTestRoot = raylinearc.HashEpisodeID(forkTestEpisode)

func TestForkContinuationStaysOnTheEpisode(t *testing.T) {
	cell := newForkCell(t)
	first := cell.run("u1")
	second := cell.run("u1", "a1", "u2")
	for _, turn := range []forkTurn{first, second} {
		if turn.arc.EpisodeIDHash != forkTestRoot || turn.arc.Fork == nil || turn.arc.Fork.Fork {
			t.Fatalf("turn on %s fork=%+v, want the root episode", turn.arc.EpisodeIDHash, turn.arc.Fork)
		}
	}
}

// The prod shape of episode f9b21a8b: run A's turn holds the episode; run B
// sends the conversation with a new user message and without A's reply. B
// goes to a sibling at once, its identical retry and its next turn find the
// same sibling, and A's next turn stays on the episode.
func TestForkWhileTheEpisodeIsMidTurnDoesNotWait(t *testing.T) {
	cell := newForkCell(t)
	cell.run("u1")
	a := cell.start("u1", "a1", "uA")
	b := cell.start("u1", "a1", "uA", "user:uB")
	if !b.arc.Fork.Fork || b.arc.EpisodeIDHash == forkTestRoot || b.arc.Fork.ForkOf != forkTestRoot {
		t.Fatalf("B on %s fork=%+v, want a sibling of the root", b.arc.EpisodeIDHash, b.arc.Fork)
	}
	if b.elapsed > time.Second {
		t.Fatalf("B waited %s on the parent's lease", b.elapsed)
	}
	sibling := b.arc.EpisodeIDHash
	b.finish(t)
	retry := cell.run("u1", "a1", "uA", "user:uB")
	if retry.arc.EpisodeIDHash != sibling {
		t.Fatalf("B's retry on %s, want %s", retry.arc.EpisodeIDHash, sibling)
	}
	next := cell.run("u1", "a1", "uA", "user:uB", "aB", "uB2")
	if next.arc.EpisodeIDHash != sibling {
		t.Fatalf("B's next turn on %s, want %s", next.arc.EpisodeIDHash, sibling)
	}
	a.finish(t)
	aNext := cell.run("u1", "a1", "uA", "aA", "uA2")
	if aNext.arc.EpisodeIDHash != forkTestRoot {
		t.Fatalf("A's next turn on %s, want the root", aNext.arc.EpisodeIDHash)
	}
}

// Two forks sent together while the parent is mid-turn each get their own
// sibling and both prepare at once: no session_busy, no wait.
func TestConcurrentForksRunConcurrently(t *testing.T) {
	cell := newForkCell(t)
	cell.run("u1")
	parent := cell.start("u1", "a1", "u2")
	defer parent.finish(t)
	var wait sync.WaitGroup
	turns := make([]forkTurn, 2)
	for index := range turns {
		wait.Add(1)
		go func() {
			defer wait.Done()
			turns[index] = cell.startBody(chatBody(t, "u1", "a1", fmt.Sprintf("fork %d", index)), llmprotocol.OpenAIChatV1)
		}()
	}
	wait.Wait()
	for index, turn := range turns {
		if turn.arc.PreparationFailure != "" {
			t.Fatalf("fork %d failed %q", index, turn.arc.PreparationFailure)
		}
		if turn.elapsed > time.Second {
			t.Fatalf("fork %d waited %s", index, turn.elapsed)
		}
		if !turn.arc.Fork.Fork {
			t.Fatalf("fork %d not on a sibling: %+v", index, turn.arc.Fork)
		}
		defer turn.finish(t)
	}
	if turns[0].arc.EpisodeIDHash == turns[1].arc.EpisodeIDHash {
		t.Fatal("two different forks share a sibling")
	}
}

// Requests that all look like the idle episode's next turn race for its
// lease. The first admitted keeps the episode; each other one, on its next
// probe, is no longer a continuation and moves to a sibling instead of
// waiting out the winner's turn.
func TestRacingContinuationsSplit(t *testing.T) {
	cell := newForkCell(t)
	cell.run("u1")
	var wait sync.WaitGroup
	turns := make([]forkTurn, 4)
	for index := range turns {
		wait.Add(1)
		go func() {
			defer wait.Done()
			turns[index] = cell.startBody(chatBody(t, "u1", "a1", fmt.Sprintf("user:racer %d", index)), llmprotocol.OpenAIChatV1)
		}()
	}
	wait.Wait()
	onRoot, seen := 0, map[string]bool{}
	for index, turn := range turns {
		defer turn.finish(t)
		if turn.arc.PreparationFailure != "" || turn.elapsed > 2*time.Second {
			t.Fatalf("racer %d: failure %q after %s", index, turn.arc.PreparationFailure, turn.elapsed)
		}
		if turn.arc.EpisodeIDHash == forkTestRoot {
			onRoot++
		}
		if seen[turn.arc.EpisodeIDHash] {
			t.Fatalf("two racers share episode %s", turn.arc.EpisodeIDHash)
		}
		seen[turn.arc.EpisodeIDHash] = true
	}
	if onRoot != 1 {
		t.Fatalf("%d racers on the root episode, want 1", onRoot)
	}
}

func TestSiblingCapIsSixteen(t *testing.T) {
	cell := newForkCell(t)
	cell.run("u1")
	parent := cell.start("u1", "a1", "u2")
	defer parent.finish(t)
	for index := range raylinearc.MaxEpisodeForkSiblings + 2 {
		// The forks run in parallel: each holds its own lease.
		turn := cell.start("u1", "a1", fmt.Sprintf("fork %d", index))
		defer turn.finish(t)
		if !turn.arc.Fork.Fork || turn.arc.Fork.SiblingCount > raylinearc.MaxEpisodeForkSiblings {
			t.Fatalf("fork %d: %+v", index, turn.arc.Fork)
		}
	}
	store := cell.router.RaylineARCEpisodeStore.(raylinearc.EpisodeForkRegistry)
	view, err := store.ForkView(context.Background(), forkTestRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Siblings) != raylinearc.MaxEpisodeForkSiblings {
		t.Fatalf("siblings = %d, want %d", len(view.Siblings), raylinearc.MaxEpisodeForkSiblings)
	}
}

// A compaction of the idle main stream (the same system prompt and first
// message, a shorter history) stays on the episode. So does an edited retry
// after a turn that failed (its lease released by the abort), and a
// "continue" after an interrupted one: single-stream safety comes first.
func TestIdleCompactionEditedRetryAndContinueStay(t *testing.T) {
	cell := newForkCell(t)
	cell.run("u1")
	cell.run("u1", "a1", "u2")
	cell.run("u1", "a1", "u2", "a2", "u3") // fails upstream: aborted
	edited := cell.run("u1", "a1", "u2", "a2", "u3 edited")
	if edited.arc.EpisodeIDHash != forkTestRoot || edited.arc.Fork.Reason != raylinearc.ForkReasonRetryEdit {
		t.Fatalf("edited retry on %s %+v, want the root as retry_edit", edited.arc.EpisodeIDHash, edited.arc.Fork)
	}
	resumed := cell.run("u1", "a1", "u2", "a2", "u3 edited", "user:continue")
	if resumed.arc.EpisodeIDHash != forkTestRoot {
		t.Fatalf("continue after an interrupted turn on %s %+v, want the root", resumed.arc.EpisodeIDHash, resumed.arc.Fork)
	}
	compacted := cell.run("u1", "summary of the conversation", "u4")
	if compacted.arc.EpisodeIDHash != forkTestRoot {
		t.Fatalf("compaction on %s %+v, want the root", compacted.arc.EpisodeIDHash, compacted.arc.Fork)
	}
}

// A side call is classified before the episode is read: fork detection
// does not run on it, and it waits on no lease.
func TestSideCallIsUnchangedByForkDetection(t *testing.T) {
	cell := newForkCell(t)
	parent := cell.startBody(claudeCodeFixtureRequest(t, "claude_code_2.1.309_main_turn_before_suggestion.json"), llmprotocol.AnthropicMessagesV1)
	if parent.arc.PreparationFailure != "" {
		t.Fatalf("main turn failure %q", parent.arc.PreparationFailure)
	}
	defer parent.finish(t)
	side := cell.startBody(claudeCodeFixtureRequest(t, "claude_code_2.1.309_prompt_suggestion.json"), llmprotocol.AnthropicMessagesV1)
	defer side.finish(t)
	if side.arc.PreparationFailure != "" || side.arc.PolicyCallKind != raylinearc.PolicyCallSide {
		t.Fatalf("side call failure=%q kind=%s", side.arc.PreparationFailure, side.arc.PolicyCallKind)
	}
	if side.arc.Fork != nil || side.arc.EpisodeIDHash != forkTestRoot || side.elapsed > time.Second {
		t.Fatalf("side call fork=%+v episode=%s elapsed=%s", side.arc.Fork, side.arc.EpisodeIDHash, side.elapsed)
	}
}

// With fork detection off, a parallel request waits on the lease as before.
func TestForkDetectionOffKeepsTodaysBehaviour(t *testing.T) {
	cell := newForkCell(t)
	off := false
	cell.algorithm.RaylineARC.Episode.ForkDetection = &off
	cell.algorithm.RaylineARC.Episode.AcquireTimeoutSeconds = 1
	cell.run("u1")
	parent := cell.start("u1", "a1", "u2")
	defer parent.finish(t)
	other := cell.startBody(chatBody(t, "u1", "a1", "user:other"), llmprotocol.OpenAIChatV1)
	if other.arc.PreparationFailure != "episode_timeout" || other.arc.Fork != nil {
		t.Fatalf("fork detection off: failure=%q fork=%+v", other.arc.PreparationFailure, other.arc.Fork)
	}
}

// subagentCell trusts the gateway's turn-signal headers, as the serving
// cells do.
func subagentCell(t *testing.T, forks bool) *forkCell {
	cell := newForkCell(t)
	cell.algorithm.RaylineARC.PolicyService.TrustTurnSignalHeaders = true
	cell.algorithm.RaylineARC.Episode.ForkDetection = &forks
	return cell
}

func (cell *forkCell) subagentTurn(format llmprotocol.WireFormat, body []byte) forkTurn {
	ctx := cell.request(body, format)
	ctx.Headers[testEpisodeIDHeader] = forkTestEpisode + ":sub-explore"
	ctx.Headers[raylineARCAgentKeySourceHeader] = "role"
	began := time.Now()
	arc := cell.router.buildRaylineARCSelectionContext(cell.algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
	return forkTurn{arc: arc, ctx: ctx, elapsed: time.Since(began)}
}

func forkAnthropicBody(t *testing.T, system string, user string) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model": "auto", "max_tokens": 1024, "system": system,
		"messages": []map[string]string{{"role": "user", "content": user}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// D-arc-subagent-own-decision: with fork detection on, a role-keyed subagent
// decides as a main turn of its own episode, and same-role subagents with
// different tasks run concurrently on separate siblings.
func TestRoleKeyedSubagentsDecideOnTheirOwnEpisodes(t *testing.T) {
	cell := subagentCell(t, true)
	var wait sync.WaitGroup
	turns := make([]forkTurn, 2)
	for index := range turns {
		wait.Add(1)
		go func() {
			defer wait.Done()
			turns[index] = cell.subagentTurn(llmprotocol.OpenAIChatV1, chatBody(t, fmt.Sprintf("task %d", index)))
		}()
	}
	wait.Wait()
	for index, turn := range turns {
		defer turn.finish(t)
		if turn.arc.PreparationFailure != "" || turn.elapsed > time.Second {
			t.Fatalf("subagent %d: failure %q after %s", index, turn.arc.PreparationFailure, turn.elapsed)
		}
		if turn.arc.PolicyCallKind != raylinearc.PolicyCallMain ||
			turn.arc.PolicyCallKindSource != callKindSourceGatewaySubagentForked || turn.arc.Fork == nil {
			t.Fatalf("subagent %d: %s/%s fork=%+v, want its own main turn", index,
				turn.arc.PolicyCallKind, turn.arc.PolicyCallKindSource, turn.arc.Fork)
		}
	}
	if turns[0].arc.EpisodeIDHash == turns[1].arc.EpisodeIDHash {
		t.Fatal("two same-role subagents with different tasks share an episode")
	}
}

// Its title helper stays a side call; with fork detection off a role-keyed
// subagent is a side call as before.
func TestRoleKeyedSubagentHelpersAndFlagOffStaySideCalls(t *testing.T) {
	cell := subagentCell(t, true)
	title := cell.subagentTurn(llmprotocol.AnthropicMessagesV1,
		forkAnthropicBody(t, raylinearc.ClaudeCodeTitlePrompt+" the session", "name it"))
	defer title.finish(t)
	if title.arc.PolicyCallKind != raylinearc.PolicyCallSide || title.arc.Fork != nil {
		t.Fatalf("title helper: %s/%s fork=%+v, want a side call", title.arc.PolicyCallKind, title.arc.PolicyCallKindSource, title.arc.Fork)
	}
	off := subagentCell(t, false)
	turn := off.subagentTurn(llmprotocol.OpenAIChatV1, chatBody(t, "task"))
	defer turn.finish(t)
	if turn.arc.PolicyCallKind != raylinearc.PolicyCallSide || turn.arc.PolicyCallKindSource != callKindSourceGatewaySubagent {
		t.Fatalf("fork detection off: %s/%s, want a side call", turn.arc.PolicyCallKind, turn.arc.PolicyCallKindSource)
	}
}

// The Claude Code claim path: a subagent the billing block declares, keyed
// on a role, is a main turn with fork detection on.
func TestClaudeCodeRoleKeyedSubagentClaimIsMainWhenForked(t *testing.T) {
	// Without a role or task key source the claim stays a side call.
	if kind, _ := raylineARCPolicyCallKindForked(nil, policyFormatAnthropic,
		json.RawMessage(`[{"type":"text","text":"x-anthropic-billing-header: cc_is_subagent=true"}]`), nil, nil, "e", true); kind != raylinearc.PolicyCallSide {
		t.Fatalf("unkeyed subagent claim = %s, want side", kind)
	}
	for _, forks := range []bool{true, false} {
		kind, source := raylineARCPolicyCallKindForked(
			map[string]string{raylineARCAgentKeySourceHeader: "role"}, policyFormatAnthropic,
			json.RawMessage(`[{"type":"text","text":"x-anthropic-billing-header: cc_is_subagent=true"}]`), nil, nil, "e", forks)
		want, wantSource := raylinearc.PolicyCallSide, callKindSourceClaudeSubagent
		if forks {
			want, wantSource = raylinearc.PolicyCallMain, callKindSourceClaudeSubagentForked
		}
		if kind != want || source != wantSource {
			t.Fatalf("forks=%v: %s/%s, want %s/%s", forks, kind, source, want, wantSource)
		}
	}
}
