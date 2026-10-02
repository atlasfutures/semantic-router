package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const coalesceTestEpisode = "episode-coalesce"

func coalesceRequestContext(body string) *RequestContext {
	return &RequestContext{
		Headers:      map[string]string{testEpisodeIDHeader: coalesceTestEpisode},
		SourceFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "public test turn"}},
			}},
		},
		RaylineARCRawBody: []byte(body),
		TraceContext:      context.Background(),
	}
}

func coalesceDecision(model string, arm int) *selection.SelectionResult {
	return &selection.SelectionResult{
		SelectedModel: model,
		Method:        selection.MethodRaylineARC,
		RaylineARC: &selection.RaylineARCTrace{
			SelectedArm:              arm,
			SerializedTokens:         11,
			EncoderVisitedReplicaIDs: []string{"replica-a"},
		},
	}
}

// A resend of a turn that is still being decided joins that decision: it
// takes no lease, so it cannot time out on the first copy's lease, and it
// dispatches with the same decision. The episode advances once.
func TestIdenticalResendJoinsTheInFlightDecision(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	refs := missingSessionModelRefs()

	leaderCtx := coalesceRequestContext(`{"turn":1}`)
	leader := router.buildRaylineARCSelectionContext(algorithm, leaderCtx, refs, raylineARCEpisodeRequired)
	if leader.PreparationFailure != "" || leader.Coalesced != nil {
		t.Fatalf("leader: failure=%q coalesced=%v", leader.PreparationFailure, leader.Coalesced)
	}
	if leaderCtx.RaylineARCInflight == nil || leaderCtx.RaylineARCTransaction.borrowed {
		t.Fatal("the first copy must lead with a real transaction")
	}

	type joined struct {
		arc *selection.RaylineARCSelectionContext
		ctx *RequestContext
	}
	followers := make(chan joined, 2)
	for range 2 {
		go func() {
			ctx := coalesceRequestContext(`{"turn":1}`)
			followers <- joined{router.buildRaylineARCSelectionContext(algorithm, ctx, refs, raylineARCEpisodeRequired), ctx}
		}()
	}
	// The followers are waiting on the leader, not on the lease: longer than
	// acquire_timeout (1 s here) must not fail them.
	time.Sleep(1200 * time.Millisecond)
	select {
	case early := <-followers:
		t.Fatalf("a resend finished before the decision existed: failure=%q", early.arc.PreparationFailure)
	default:
	}

	decision := coalesceDecision("arm-2", 2)
	leaderCtx.RaylineARCTransaction.markSelection(2, 11)
	router.publishRaylineARCDecision(leaderCtx, decision)

	collected := []joined{<-followers, <-followers}
	for _, follower := range collected {
		if follower.arc.PreparationFailure != "" {
			t.Fatalf("follower failure = %q", follower.arc.PreparationFailure)
		}
		if follower.arc.Coalesced == nil || follower.arc.Coalesced.SelectedModel != "arm-2" {
			t.Fatalf("follower decision = %+v", follower.arc.Coalesced)
		}
		if follower.arc.Coalesced.RaylineARC == decision.RaylineARC {
			t.Fatal("a follower shares the leader's trace instead of a copy")
		}
		if follower.ctx.RaylineARCTransaction == nil || !follower.ctx.RaylineARCTransaction.borrowed {
			t.Fatal("a follower must dispatch under a borrowed transaction")
		}
		if err := follower.ctx.RaylineARCTransaction.commit(context.Background(), follower.ctx); err != nil {
			t.Fatalf("borrowed commit = %v", err)
		}
	}
	if err := leaderCtx.RaylineARCTransaction.commit(context.Background(), leaderCtx); err != nil {
		t.Fatalf("leader commit = %v", err)
	}

	// One turn was committed, by the leader alone.
	after := coalesceRequestContext(`{"turn":2}`)
	next := router.buildRaylineARCSelectionContext(algorithm, after, refs, raylineARCEpisodeRequired)
	if next.PreparationFailure != "" {
		t.Fatalf("next turn failure = %q", next.PreparationFailure)
	}
	if got := next.State.TurnIndex; got != 1 {
		t.Fatalf("turn index = %d, want 1: the resends must not advance the episode", got)
	}
	_ = after.RaylineARCTransaction.abort(context.Background(), "test")
}

// A different body on the same episode is a different turn and keeps today's
// behaviour: it contends for the lease.
func TestDifferentBodyStillContends(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	refs := missingSessionModelRefs()
	leaderCtx := coalesceRequestContext(`{"turn":1}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, leaderCtx, refs, raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("leader failure = %q", failure)
	}
	other := router.buildRaylineARCSelectionContext(algorithm, coalesceRequestContext(`{"turn":"other"}`), refs, raylineARCEpisodeRequired)
	if other.PreparationFailure != "episode_timeout" {
		t.Fatalf("different body failure = %q, want episode_timeout", other.PreparationFailure)
	}
	_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "test")
}

// A leader that gives up without deciding releases its followers to decide
// for themselves; the first to arrive leads the next attempt.
func TestResendLeadsWhenTheFirstCopyGivesUp(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	refs := missingSessionModelRefs()
	leaderCtx := coalesceRequestContext(`{"turn":1}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, leaderCtx, refs, raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("leader failure = %q", failure)
	}
	result := make(chan *RequestContext, 1)
	go func() {
		ctx := coalesceRequestContext(`{"turn":1}`)
		arc := router.buildRaylineARCSelectionContext(algorithm, ctx, refs, raylineARCEpisodeRequired)
		if arc.PreparationFailure != "" || arc.Coalesced != nil {
			t.Errorf("resend after an abandoned leader: failure=%q coalesced=%v", arc.PreparationFailure, arc.Coalesced)
		}
		result <- ctx
	}()
	time.Sleep(50 * time.Millisecond)
	_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "test")
	resend := <-result
	if resend.RaylineARCTransaction == nil || resend.RaylineARCTransaction.borrowed || resend.RaylineARCInflight == nil {
		t.Fatal("the resend must lead its own attempt with a real transaction")
	}
	_ = resend.RaylineARCTransaction.abort(context.Background(), "test")
}

// A follower whose own request ends while it waits fails as that request
// ending, not as contention.
func TestResendWaitEndsWithItsOwnRequest(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	refs := missingSessionModelRefs()
	leaderCtx := coalesceRequestContext(`{"turn":1}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, leaderCtx, refs, raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("leader failure = %q", failure)
	}
	ctx := coalesceRequestContext(`{"turn":1}`)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	ctx.TraceContext = cancelled
	arc := router.buildRaylineARCSelectionContext(algorithm, ctx, refs, raylineARCEpisodeRequired)
	if arc.PreparationFailure != "episode_canceled" {
		t.Fatalf("failure = %q, want episode_canceled", arc.PreparationFailure)
	}
	_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "test")
}

// The selector returns a joined decision as a copy and decides nothing.
func TestSelectorReturnsTheJoinedDecision(t *testing.T) {
	decision := coalesceDecision("arm-4", 4)
	got, err := (&raylineARCSelector{}).Select(context.Background(), &selection.SelectionContext{
		RaylineARC: &selection.RaylineARCSelectionContext{Coalesced: decision},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got == decision || got.SelectedModel != "arm-4" || got.RaylineARC.SelectedArm != 4 {
		t.Fatalf("selector result = %+v", got)
	}
}

func leadTurn(t *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig) *RequestContext {
	t.Helper()
	ctx := coalesceRequestContext(`{"turn":1}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("leader failure = %q", failure)
	}
	return ctx
}

// joinFollower starts an identical resend and returns it once it has joined
// the leader's published decision.
func joinFollower(t *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig, leaderCtx *RequestContext) *RequestContext {
	t.Helper()
	joined := make(chan *RequestContext, 1)
	go func() {
		ctx := coalesceRequestContext(`{"turn":1}`)
		arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
		if arc.PreparationFailure != "" || arc.Coalesced == nil {
			t.Errorf("follower: failure=%q coalesced=%v", arc.PreparationFailure, arc.Coalesced)
		}
		joined <- ctx
	}()
	time.Sleep(50 * time.Millisecond)
	leaderCtx.RaylineARCTransaction.markSelection(2, 11)
	router.publishRaylineARCDecision(leaderCtx, coalesceDecision("arm-2", 2))
	return <-joined
}

func turnIndexAfter(t *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig) uint64 {
	t.Helper()
	ctx := coalesceRequestContext(`{"turn":"probe"}`)
	arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
	if arc.PreparationFailure != "" {
		t.Fatalf("probe failure = %q: the lease was not released", arc.PreparationFailure)
	}
	defer func() { _ = ctx.RaylineARCTransaction.abort(context.Background(), "test") }()
	return arc.State.TurnIndex
}

// Only the first copy records the turn. When its own upstream call fails, the
// turn goes unrecorded even if a resend's call succeeded: a deliberate
// trade, documented on raylineARCInflightRegistry.
func TestOnlyTheFirstCopyRecordsTheTurn(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	leaderCtx := leadTurn(t, router, algorithm)
	follower := joinFollower(t, router, algorithm, leaderCtx)

	_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "upstream_non_2xx")
	if err := follower.RaylineARCTransaction.commit(context.Background(), follower); err != nil {
		t.Fatalf("resend commit = %v, want a no-op", err)
	}
	if got := turnIndexAfter(t, router, algorithm); got != 0 {
		t.Fatalf("turn index = %d, want 0: a resend never records the turn", got)
	}
}

// A resend released by the decision after the first copy has already
// committed and left the registry is still the same turn: it dispatches under
// its borrowed transaction and the episode advances once.
func TestResendAfterTheFirstCopyCommittedAdvancesNothing(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	leaderCtx := leadTurn(t, router, algorithm)
	follower := joinFollower(t, router, algorithm, leaderCtx)

	if err := leaderCtx.RaylineARCTransaction.commit(context.Background(), leaderCtx); err != nil {
		t.Fatalf("leader commit = %v", err)
	}
	if !follower.RaylineARCTransaction.borrowed {
		t.Fatal("the resend must stay a borrowed copy of the committed turn")
	}
	if err := follower.RaylineARCTransaction.commit(context.Background(), follower); err != nil {
		t.Fatalf("resend commit = %v", err)
	}
	if got := turnIndexAfter(t, router, algorithm); got != 1 {
		t.Fatalf("turn index = %d, want 1", got)
	}
}

// Equal bodies that differ in a request input ARC reads outside the body are
// different turns and must not coalesce.
func TestEqualBodiesWithDifferentTurnInputsDoNotCoalesce(t *testing.T) {
	for name, vary := range map[string]func(*RequestContext, *config.AlgorithmConfig) *config.AlgorithmConfig{
		"close request": func(ctx *RequestContext, algorithm *config.AlgorithmConfig) *config.AlgorithmConfig {
			ctx.Headers["x-rayline-episode-close"] = "true"
			return algorithm
		},
		"wire format": func(ctx *RequestContext, algorithm *config.AlgorithmConfig) *config.AlgorithmConfig {
			ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
			return algorithm
		},
		"decision": func(_ *RequestContext, algorithm *config.AlgorithmConfig) *config.AlgorithmConfig {
			otherARC := *algorithm.RaylineARC
			other := *algorithm
			other.RaylineARC = &otherARC
			return &other
		},
	} {
		t.Run(name, func(t *testing.T) {
			router, _, algorithm := missingSessionRequestContext(t, "")
			leaderCtx := leadTurn(t, router, algorithm)
			ctx := coalesceRequestContext(`{"turn":1}`)
			decision := vary(ctx, algorithm)
			// A wrong join would wait on a decision that never comes; bound it
			// so that shows up as a failure rather than a hang.
			bounded, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			ctx.TraceContext = bounded
			arc := router.buildRaylineARCSelectionContext(decision, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
			if arc.Coalesced != nil || arc.PreparationFailure != "episode_timeout" {
				t.Fatalf("coalesced=%v failure=%q, want a contended, separate turn", arc.Coalesced, arc.PreparationFailure)
			}
			_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "test")
		})
	}
}

// A caller's provider credential decides who issues encrypted reasoning, so
// equal bodies sent with different credentials are different turns. Equal
// credentials still coalesce.
func TestEqualBodiesWithDifferentCredentialsDoNotCoalesce(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	router.CredentialResolver = authz.NewCredentialResolver(
		authz.NewHeaderInjectionProvider(map[string]string{"openai": "x-user-openai-key"}),
	)
	withKey := func(key string) *RequestContext {
		ctx := coalesceRequestContext(`{"turn":1}`)
		ctx.Headers["x-user-openai-key"] = key
		return ctx
	}
	leaderCtx := withKey("key-a")
	if failure := router.buildRaylineARCSelectionContext(algorithm, leaderCtx, missingSessionModelRefs(), raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("leader failure = %q", failure)
	}

	other := withKey("key-b")
	bounded, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	other.TraceContext = bounded
	arc := router.buildRaylineARCSelectionContext(algorithm, other, missingSessionModelRefs(), raylineARCEpisodeRequired)
	if arc.Coalesced != nil || arc.PreparationFailure != "episode_timeout" {
		t.Fatalf("different credential: coalesced=%v failure=%q, want a contended, separate turn", arc.Coalesced, arc.PreparationFailure)
	}

	same := make(chan *selection.RaylineARCSelectionContext, 1)
	go func() {
		same <- router.buildRaylineARCSelectionContext(algorithm, withKey("key-a"), missingSessionModelRefs(), raylineARCEpisodeRequired)
	}()
	time.Sleep(50 * time.Millisecond)
	leaderCtx.RaylineARCTransaction.markSelection(2, 11)
	router.publishRaylineARCDecision(leaderCtx, coalesceDecision("arm-2", 2))
	if joined := <-same; joined.Coalesced == nil {
		t.Fatalf("same credential did not coalesce: failure=%q", joined.PreparationFailure)
	}
	_ = leaderCtx.RaylineARCTransaction.abort(context.Background(), "test")
}

// The key carries a digest of the credential, never the credential.
func TestCoalescingKeyHoldsNoCredential(t *testing.T) {
	ctx := coalesceRequestContext(`{"turn":1}`)
	ctx.Headers["x-user-openai-key"] = "sk-secret-value"
	inputs := raylineARCTurnInputs(raylineARCAlgorithmConfigForTest().RaylineARC, ctx, []string{"x-user-openai-key"})
	if strings.Contains(inputs, "sk-secret-value") {
		t.Fatalf("coalescing key leaks the credential: %s", inputs)
	}
}

// Two otherwise identical requests that the gateway keyed differently are
// different turns: one may be a subagent's own main turn, the other a side
// call, so neither joins the other's decision.
func TestCoalescingKeySeparatesSubagentKeySources(t *testing.T) {
	config := raylineARCAlgorithmConfigForTest().RaylineARC
	agent := coalesceRequestContext(`{"turn":1}`)
	agent.Headers[raylineARCAgentKeySourceHeader] = "agent"
	task := coalesceRequestContext(`{"turn":1}`)
	task.Headers[raylineARCAgentKeySourceHeader] = "task"
	if raylineARCTurnInputs(config, agent, nil) == raylineARCTurnInputs(config, task, nil) {
		t.Fatal("requests keyed by agent and by task share a coalescing identity")
	}
}
