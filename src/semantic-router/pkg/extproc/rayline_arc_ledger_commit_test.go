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
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The attribution ledger records which action produced an assistant message.
// It may only record a reply the client actually received in full: a turn
// whose stream broke is retried by the client on the same prefix, and a
// ledger written at the headers would then hold two entries for one message
// and count the turn twice.

var ledgerTestMessages = []json.RawMessage{
	json.RawMessage(`{"role":"user","content":"first"}`),
	json.RawMessage(`{"role":"assistant","content":"reply"}`),
	json.RawMessage(`{"role":"user","content":"second"}`),
}

var ledgerTestRoles = []string{"user", "assistant", "user"}

func newLedgerTestStore(t *testing.T) (*raylinearc.MemoryEpisodeStore, string) {
	t.Helper()
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 8,
		IdleTTL:     time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	return store, raylinearc.HashEpisodeID(t.Name())
}

// runLedgerTestTurn prepares one policy-service turn on the episode, stages the
// ledger entry the turn would write, and drives it through the response seams
// with respond. The process-terminal finalizer runs last, as it does when the
// ext_proc stream ends.
func runLedgerTestTurn(
	t *testing.T,
	store *raylinearc.MemoryEpisodeStore,
	episode string,
	respond func(*OpenAIRouter, *RequestContext),
) {
	t.Helper()
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	turn, _, _ := raylinearc.PolicyTurn(state.Policy, ledgerTestMessages, ledgerTestRoles, state.TurnIndex, raylinearc.PolicyTurnSignals{})
	ctx := &RequestContext{
		Headers: make(map[string]string),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(
			store, lease, state, episode, time.Minute, nil,
		),
	}
	ctx.RaylineARCTransaction.markSelection(1, 123)
	ctx.RaylineARCTransaction.markPolicyState(
		turn.Next(ledgerTestMessages, strings.Repeat("a", 64), "arm-b"), false,
	)
	bindRaylineARCSelectionTransaction(ctx)
	router := &OpenAIRouter{}
	respond(router, ctx)
	finalizeSelectionProcessTerminal(ctx)
}

func streamingResponseHeaders(statusCode string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", Value: statusCode},
				{Key: "content-type", Value: "text/event-stream"},
			}},
		},
	}
}

func sendHeaders(t *testing.T, router *OpenAIRouter, ctx *RequestContext, headers *ext_proc.ProcessingRequest_ResponseHeaders) {
	t.Helper()
	response, err := router.handleResponseHeaders(headers, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if response.GetImmediateResponse() != nil {
		t.Fatalf("response headers were refused: %#v", response.GetImmediateResponse())
	}
}

// completeStream ends the stream the way a received terminal event does.
func completeStream(router *OpenAIRouter, ctx *RequestContext) {
	ctx.SemanticStreamState = &semanticResponseStreamState{
		terminal: true,
		items:    map[int]*semanticStreamItem{},
	}
	router.finalizeSemanticStreamingResponse(ctx, nil)
}

// breakStream ends the stream the way a cut connection does: before the
// terminal event, through the receive-error path.
func breakStream(router *OpenAIRouter, ctx *RequestContext) {
	ctx.SemanticStreamState = &semanticResponseStreamState{items: map[int]*semanticStreamItem{}}
	_ = router.handleProcessReceiveError(ctx, status.Error(codes.Canceled, "stream cut mid-response"))
}

func readLedgerTestState(t *testing.T, store *raylinearc.MemoryEpisodeStore, episode string) *raylinearc.EpisodeState {
	t.Helper()
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	return state
}

func ledgerLength(state *raylinearc.EpisodeState) int {
	if state.Policy == nil {
		return 0
	}
	return len(state.Policy.Ledger)
}

func TestRaylineARCLedgerNon2xxCommitsNothing(t *testing.T) {
	store, episode := newLedgerTestStore(t)
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		if _, err := router.handleResponseHeaders(streamingResponseHeaders("529"), ctx); err != nil {
			t.Fatal(err)
		}
	})
	state := readLedgerTestState(t, store, episode)
	if ledgerLength(state) != 0 || state.TurnIndex != 0 || state.PreviousArm != nil {
		t.Fatalf("a non-2xx response advanced the episode: turn=%d ledger=%d", state.TurnIndex, ledgerLength(state))
	}
}

func TestRaylineARCLedgerMidStreamFailureThenRetryCommitsOnce(t *testing.T) {
	store, episode := newLedgerTestStore(t)
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
		breakStream(router, ctx)
	})
	if state := readLedgerTestState(t, store, episode); ledgerLength(state) != 0 || state.TurnIndex != 0 {
		t.Fatalf("a broken stream advanced the episode: turn=%d ledger=%d", state.TurnIndex, ledgerLength(state))
	}
	// The client retries the same prefix, and this time the reply arrives.
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
		completeStream(router, ctx)
	})
	state := readLedgerTestState(t, store, episode)
	if ledgerLength(state) != 1 || state.TurnIndex != 1 {
		t.Fatalf("broken stream + retry: turn=%d ledger=%#v, want one turn and one entry",
			state.TurnIndex, state.Policy)
	}
	if entry := state.Policy.Ledger[0]; entry.Message != len(ledgerTestMessages) || entry.ArmID != "arm-b" {
		t.Fatalf("ledger entry = %#v", entry)
	}
}

func TestRaylineARCLedgerCompletedStreamCommitsOnce(t *testing.T) {
	store, episode := newLedgerTestStore(t)
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
		if state := readLedgerTestStateNoWait(store, episode); state != nil {
			t.Fatal("the episode committed at the response headers")
		}
		completeStream(router, ctx)
		// A trailer-driven second end must not commit again.
		completeStream(router, ctx)
	})
	state := readLedgerTestState(t, store, episode)
	if ledgerLength(state) != 1 || state.TurnIndex != 1 {
		t.Fatalf("completed stream: turn=%d ledger=%d, want 1/1", state.TurnIndex, ledgerLength(state))
	}
}

func TestRaylineARCLedgerNonStreamCommitsOnceAfterBody(t *testing.T) {
	store, episode := newLedgerTestStore(t)
	runLedgerTestTurn(t, store, episode, func(router *OpenAIRouter, ctx *RequestContext) {
		sendHeaders(t, router, ctx, arcResponseHeaders("200"))
		if state := readLedgerTestStateNoWait(store, episode); state != nil {
			t.Fatal("the episode committed at the response headers")
		}
		if err := finalizeSelectionCompletion(ctx); err != nil {
			t.Fatal(err)
		}
		if err := finalizeSelectionCompletion(ctx); err != nil {
			t.Fatal(err)
		}
	})
	state := readLedgerTestState(t, store, episode)
	if ledgerLength(state) != 1 || state.TurnIndex != 1 {
		t.Fatalf("non-stream: turn=%d ledger=%d, want 1/1", state.TurnIndex, ledgerLength(state))
	}
}

// readLedgerTestStateNoWait reads the committed episode if no lease is held
// on it, and returns nil while one is: a turn that has not committed still
// holds its lease.
func readLedgerTestStateNoWait(store *raylinearc.MemoryEpisodeStore, episode string) *raylinearc.EpisodeState {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	lease, state, err := store.Prepare(ctx, episode, 2)
	if err != nil {
		return nil
	}
	_ = store.Abort(context.Background(), lease)
	return state
}

// completeTestResponse stands for the client receiving the whole response
// body, which is where a turn commits. It is a no-op after a non-2xx status.
func completeTestResponse(t *testing.T, ctx *RequestContext) {
	t.Helper()
	if err := finalizeSelectionCompletion(ctx); err != nil {
		t.Fatalf("response completion: %v", err)
	}
}
