//go:build !windows && cgo

package extproc

import (
	"context"
	"testing"
	"time"

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
