//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A refused turn is delivered but recorded nowhere: no ledger entry, no
// completed turn, the held arm unchanged, as collection ended a refused query
// with no delivered turn. Each case runs beside its positive control, the same
// turn answered normally, which commits.

func TestRaylineARCRefusedStreamCommitsNothing(t *testing.T) {
	for _, stop := range []llmprotocol.StopReason{llmprotocol.StopContentFilter, llmprotocol.StopEndTurn} {
		store, episode := newLedgerTestStore(t)
		runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
			sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
			ctx.SemanticStreamState = &semanticResponseStreamState{terminal: true, stop: stop, items: map[int]*semanticStreamItem{}}
			router.finalizeSemanticStreamingResponse(ctx, nil)
			runPendingSelectionCompletion(ctx)
		})
		state := readLedgerTestState(t, store, episode)
		committed := ledgerLength(state) == 1 && state.TurnIndex == 1
		if refused := stop == llmprotocol.StopContentFilter; committed == refused || (refused && (ledgerLength(state) != 0 || state.TurnIndex != 0 || state.PreviousArm != nil)) {
			t.Fatalf("stop %s: turn=%d ledger=%d previous=%v", stop, state.TurnIndex, ledgerLength(state), state.PreviousArm)
		}
	}
}

const arcRefusedCompletion = `{
  "id": "gen-refused",
  "object": "chat.completion",
  "created": 1757030000,
  "model": "test",
  "choices": [{"index": 0, "finish_reason": "content_filter", "native_finish_reason": "refusal",
    "message": {"role": "assistant", "content": null, "refusal": "This request triggered restrictions."}}]
}`

func TestRaylineARCRefusedNonStreamCommitsNothingAndIsNotCached(t *testing.T) {
	for _, body := range []string{arcRefusedCompletion, arcCacheTestCompletion} {
		refused := body == arcRefusedCompletion
		logs := captureLogs(t)
		mockCache, router, decision := statusCacheRouter()
		store, episode := newLedgerTestStore(t)
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-arc-refusal", RequestModel: "test", RequestQuery: "hello",
			SemanticRequest: testNeutralRequest("test", "hello"),
			SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: context.Background(), UpstreamStatusCode: 200,
			RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
		}, decision)
		ctx.RaylineARCTransaction.commitOnCompletion = true
		ctx.RaylineARCTransaction.markSelection(0, 10)
		bindRaylineARCSelectionTransaction(ctx)
		response := router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
		runPendingSelectionCompletion(ctx) // the reply was sent
		finalizeSelectionProcessTerminal(ctx)
		if response.GetImmediateResponse() != nil {
			t.Fatalf("refused=%v: the reply was not delivered", refused)
		}
		after := readLedgerTestState(t, store, episode)
		if (after.TurnIndex == 1) == refused || mockCache.addEntryCalled == refused {
			t.Fatalf("refused=%v: turn=%d cached=%v", refused, after.TurnIndex, mockCache.addEntryCalled)
		}
		// A refusal states no usage: its counts are unknown, not an
		// authoritative zero.
		want := "authoritative"
		if refused {
			want = "unknown"
		}
		if source := findLogEvent(t, logs, "llm_usage")["usage_source"]; source != want {
			t.Fatalf("refused=%v: usage_source = %v, want %s", refused, source, want)
		}
	}
}

// A refusal clears a retained boundary decision that chose the refused arm,
// so the client's retry decides again. Its control, the same attempt failing
// with a 5xx, keeps the decision and the retry reuses it.
func TestRaylineARCRefusalClearsTheRetainedBoundaryArm(t *testing.T) {
	for _, refusal := range []bool{true, false} {
		fixture, store, episode := boundaryFixture(t)
		body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
		first, decided := boundaryAttempt(t, fixture, store, episode, body)
		if decided.RaylineARC.PolicyBoundary == nil {
			t.Fatal("no boundary decision was stored")
		}
		first.VSRRaylineARC = decided.RaylineARC
		if refusal {
			declineRefusedTurn(first)
		}
		if err := first.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
			t.Fatal(err)
		}
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		_, retried := boundaryAttempt(t, fixture, store, episode, body)
		offered := offeredActions(fixture, 1)
		if redecided := len(offered) > 1 && retried.RaylineARC.PolicyBoundary != nil; redecided != refusal {
			t.Fatalf("refusal=%v: retry offered %v, boundary %+v", refusal, offered, retried.RaylineARC.PolicyBoundary)
		}
	}
}

// A refusal the response jailbreak plugin blocks is still declined: the
// plugin's early return must not leave the boundary decision that chose the
// refusing arm in place. Control: the same blocked reply without a refusal
// keeps the decision, as any failed attempt does.
func TestRaylineARCBlockedRefusalStillClearsTheBoundaryArm(t *testing.T) {
	for _, body := range []string{arcRefusedCompletion, arcCacheTestCompletion} {
		refused := body == arcRefusedCompletion
		fixture, store, episode := boundaryFixture(t)
		attempt, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		router, ctx := newResponseStageRouter(t, newJailbreakFailingServer(t), config.OnErrorBlock, "block")
		ctx.RequestModel, ctx.UpstreamStatusCode = "test", 200
		ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
		ctx.RaylineARCTransaction, ctx.VSRRaylineARC = attempt.RaylineARCTransaction, decided.RaylineARC
		ctx.RaylineARCTransaction.commitOnCompletion = true
		bindRaylineARCSelectionTransaction(ctx)
		response := router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
		if response.GetImmediateResponse() == nil {
			t.Fatalf("refused=%v: the plugin did not block", refused)
		}
		finalizeSelectionProcessTerminal(ctx)
		after := readLedgerTestState(t, store, episode)
		if cleared := after.PolicyBoundary == nil; cleared != refused {
			t.Fatalf("refused=%v: boundary after the blocked reply = %+v", refused, after.PolicyBoundary)
		}
	}
}

// Clearing a relaxed boundary is bounded like staging one: a relaxed turn
// never waits on episode state.
func TestRaylineARCRelaxedRefusalClearIsBounded(t *testing.T) {
	_, store, episode := boundaryFixture(t)
	state, read, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	state.PolicyBoundary = raylinearc.NewPolicyBoundaryDecision(1, 0, &raylinearc.PolicyEpisodeState{}, messages)
	transaction := newRelaxedRaylineARCEpisodeTransaction(blockingSnapshotStore{store}, state, read, episode, false)
	started := time.Now()
	transaction.clearRefusedBoundary(context.Background(), 1)
	if elapsed := time.Since(started); elapsed < relaxedBoundaryStageTimeout/2 || elapsed > time.Second {
		t.Fatalf("relaxed refusal clear took %v, want about %v", elapsed, relaxedBoundaryStageTimeout)
	}
}

// A refusal part with empty text, ended by an ordinary stop, is still a
// refusal: the reduced stream keeps the refusal, and the turn commits
// nothing. Control: the same stream carrying text commits.
func TestRaylineARCEmptyStreamedRefusalCommitsNothing(t *testing.T) {
	for _, kind := range []llmprotocol.ContentKind{llmprotocol.ContentRefusal, llmprotocol.ContentText} {
		store, episode := newLedgerTestStore(t)
		runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
			sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
			delta := ""
			if kind == llmprotocol.ContentText {
				delta = "hi"
			}
			state := &semanticResponseStreamState{items: map[int]*semanticStreamItem{}}
			state.observe([]llmprotocol.Event{
				{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, Role: llmprotocol.RoleAssistant, Content: &llmprotocol.Content{Kind: kind}},
				{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, Delta: delta, Content: &llmprotocol.Content{Kind: kind}},
				{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0},
				{Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn},
			})
			if _, err := state.response(); err != nil {
				t.Fatalf("%s: the stream does not reconstruct: %v", kind, err)
			}
			ctx.SemanticStreamState = state
			router.finalizeSemanticStreamingResponse(ctx, nil)
			runPendingSelectionCompletion(ctx)
		})
		state := readLedgerTestState(t, store, episode)
		if committed := state.TurnIndex == 1; committed == (kind == llmprotocol.ContentRefusal) {
			t.Fatalf("%s: turn=%d", kind, state.TurnIndex)
		}
	}
}

// A coalesced resend that was refused clears the boundary decision even
// though the request that decided failed and kept it. Control: a resend
// that was not refused leaves it. And while the deciding request still holds
// the lease, the resend gives up within the relaxed bound.
func TestRaylineARCRefusedCoalescedResendClearsTheBoundaryArm(t *testing.T) {
	for _, refusal := range []bool{true, false} {
		fixture, store, episode := boundaryFixture(t)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		if err := leader.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
			t.Fatal(err)
		}
		staged := readLedgerTestState(t, store, episode)
		follower := &RequestContext{
			RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(store, staged, episode, nil),
			VSRRaylineARC:         decided.RaylineARC,
		}
		if refusal {
			declineRefusedTurn(follower)
		}
		if cleared := readLedgerTestState(t, store, episode).PolicyBoundary == nil; cleared != refusal {
			t.Fatalf("refusal=%v: boundary cleared=%v", refusal, cleared)
		}
	}
	// While the leader holds the lease, the resend does not wait on it: it
	// hands its refusal to the leader, whose abort clears the decision.
	// Control: with no refusal handed over, the leader's abort keeps it.
	for _, refusal := range []bool{true, false} {
		fixture, store, episode := boundaryFixture(t)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		leader.RaylineARCTransaction.inflight = entry
		follower := &RequestContext{
			RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(store, leader.RaylineARCTransaction.state, episode, entry),
			VSRRaylineARC:         decided.RaylineARC,
		}
		started := time.Now()
		if refusal {
			declineRefusedTurn(follower)
		}
		if elapsed := time.Since(started); elapsed > time.Second {
			t.Fatalf("a refused resend waited %v on the leader's lease", elapsed)
		}
		if err := leader.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
			t.Fatal(err)
		}
		if cleared := readLedgerTestState(t, store, episode).PolicyBoundary == nil; cleared != refusal {
			t.Fatalf("leader held the lease, refusal=%v: boundary cleared=%v", refusal, cleared)
		}
	}
}

// A late refused resend clears only the decision it was dispatched under: a
// newer request's decision on the same arm survives it.
func TestRaylineARCRefusedResendKeepsANewerBoundaryOnTheSameArm(t *testing.T) {
	fixture, store, episode := boundaryFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	leader, decided := boundaryAttempt(t, fixture, store, episode, body)
	if err := leader.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	borrowed := readLedgerTestState(t, store, episode)
	// Another request stages its own decision, on the same arm.
	lease, current, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	newer := cloneARCState(current)
	boundary := *current.PolicyBoundary
	boundary.PrefixDigest = strings.Repeat("f", 64)
	newer.PolicyBoundary = &boundary
	if err := store.Stage(context.Background(), lease, newer); err != nil {
		t.Fatal(err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	declineRefusedTurn(&RequestContext{
		RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(store, borrowed, episode, nil),
		VSRRaylineARC:         decided.RaylineARC,
	})
	if after := readLedgerTestState(t, store, episode).PolicyBoundary; after == nil || after.PrefixDigest != strings.Repeat("f", 64) {
		t.Fatalf("the newer request's decision was cleared: %+v", after)
	}
}

// deadlineStore stages only while its context is live, as a networked store
// does; the in-memory store ignores the context.
type deadlineStore struct {
	*raylinearc.MemoryEpisodeStore
}

func (store deadlineStore) Stage(ctx context.Context, lease raylinearc.Lease, state *raylinearc.EpisodeState) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return store.MemoryEpisodeStore.Stage(ctx, lease, state)
}

// failingCommitStore fails every commit, as a store outage at commit does.
type failingCommitStore struct {
	*raylinearc.MemoryEpisodeStore
}

func (failingCommitStore) Commit(context.Context, raylinearc.Lease, uint64, *raylinearc.EpisodeState) error {
	return errors.New("episode store unavailable")
}

// A resend whose first wait ran out, and whose leader finished meanwhile,
// takes the lease on its retry and stages the clear under that retry's live
// context. Control: with no leader finished, it hands the decision over and
// leaves the store alone.
func TestRaylineARCRefusedResendRetryStagesUnderItsLiveContext(t *testing.T) {
	for _, leaderFinished := range []bool{true, false} {
		fixture, store, episode := boundaryFixture(t)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		if leaderFinished {
			close(entry.finished)
		}
		follower := &RequestContext{
			RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(deadlineStore{store}, leader.RaylineARCTransaction.state, episode, entry),
			VSRRaylineARC:         decided.RaylineARC,
		}
		// The holder releases after the first wait has run out, within the retry.
		go func() {
			time.Sleep(relaxedBoundaryStageTimeout + relaxedBoundaryStageTimeout/5)
			_ = leader.RaylineARCTransaction.abort(context.Background(), "upstream_status")
		}()
		declineRefusedTurn(follower)
		time.Sleep(relaxedBoundaryStageTimeout)
		if cleared := readLedgerTestState(t, store, episode).PolicyBoundary == nil; cleared != leaderFinished {
			t.Fatalf("leader finished=%v: boundary cleared=%v", leaderFinished, cleared)
		}
	}
}

// A leader whose commit fails leaves the turn unrecorded, as an abort does,
// and still clears a refusal a resend handed over. Control: with nothing
// handed over, the failed commit keeps the decision.
func TestRaylineARCLeaderCommitFailureClearsAHandedOverRefusal(t *testing.T) {
	for _, refusal := range []bool{true, false} {
		fixture, store, episode := boundaryFixture(t)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		leader.RaylineARCTransaction.inflight = entry
		leader.RaylineARCTransaction.store = failingCommitStore{store}
		if refusal {
			entry.noteRefusedBoundary(*decided.RaylineARC.PolicyBoundary)
		}
		leader.RaylineARCTransaction.markPolicyState(decided.RaylineARC.PolicyNextState, false)
		if err := leader.RaylineARCTransaction.commit(context.Background(), leader); err == nil {
			t.Fatal("the commit did not fail")
		}
		if cleared := readLedgerTestState(t, store, episode).PolicyBoundary == nil; cleared != refusal {
			t.Fatalf("refusal=%v: boundary cleared=%v after the failed commit", refusal, cleared)
		}
	}
}
