//go:build !windows && cgo

package extproc

import (
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A Chat stream states its stop and its usage on separate chunks: the
// re-serve waits for the usage, so the refused call is settled with it.
func TestRefusalReserveWaitsForTheRefusedCallsUsage(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	stop := `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
		`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}` + "\n\n"
	usage := `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
		`"model":"anthropic/claude-opus-5.5","choices":[],` +
		`"usage":{"prompt_tokens":120,"completion_tokens":7,"total_tokens":127,"cost":0.0009}}` + "\n\n"

	for _, chunk := range []string{reserveRoleChunk, reserveThinkingChunk, stop} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	assert.False(t, ctx.HostRetry.started(), "re-served before the refused call stated its usage")
	assert.True(t, router.holdingForRefusalReserve(ctx), "the stop released the hold")
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(usage, false), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served once its usage arrived")
	drainSpliced(t, router, ctx, stream)

	sent, _ := clientBytes(stream)
	refused := attemptsRecord(t, sent)["attempts"].([]interface{})[0].(map[string]interface{})
	require.NotNil(t, refused["usage"], "the refused attempt's usage was lost")
	assert.Equal(t, float64(120), refused["usage"].(map[string]interface{})["input_tokens"])
	assert.Equal(t, float64(7), refused["usage"].(map[string]interface{})["output_tokens"])
	assert.Equal(t, 0.0009, refused["cost_usd"])
}

// A re-decision that outlasts the budget starts no paid call.
func TestRefusalReserveRechecksTheBudgetBeforeDispatch(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RefusalReserve.budget = 20 * time.Millisecond
	ctx.RefusalReserve.redecide = func(*RequestContext) (*config.ModelRef, string, error) {
		time.Sleep(40 * time.Millisecond)
		return &config.ModelRef{Model: reserveGLMArm}, "test", nil
	}

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started(), "a re-serve started past its budget")
	assert.Empty(t, upstream.calls())
	assert.Equal(t, reserveClaudeArm, ctx.VSRSelectedModel, "the refused arm's turn was not restored")
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
}

// A wake past the hold, with no keepalive owed, is still a streamed turn
// under a re-serve plan: the loop takes it as the keepalive clock's, never
// as a held body's deadline that would end the stream.
func TestExpiredHoldWakeIsNotAStall(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))

	ctx.StartTime = time.Now().Add(-2 * ctx.RefusalReserve.hold)
	require.NoError(t, router.sendLapsedHoldFrames(stream, ctx))
	assert.Zero(t, router.refusalReserveKeepAliveWait(ctx))
	assert.True(t, refusalReserveStreaming(ctx))
	assert.False(t, router.refusalReserveKeepAliveDue(ctx))
}

// Every Responses turn has object state; only one that depends on retained
// history is not held.
func TestResponseHistoryRetained(t *testing.T) {
	assert.False(t, responseHistoryRetained(nil))
	assert.False(t, responseHistoryRetained(&ResponseObjectState{GeneratedResponseID: "resp_1"}))
	assert.True(t, responseHistoryRetained(&ResponseObjectState{PreviousResponseID: "resp_0"}))
	assert.True(t, responseHistoryRetained(&ResponseObjectState{ConversationID: "conv_1"}))
}

// A re-serve that does not start puts the refused arm's attempt-local
// dispatch attribution back: its worker, thinking and endpoint.
func TestReserveSnapshotRestoresDispatchAttribution(t *testing.T) {
	worker := &raylinearc.WorkerManifest{ID: "claude-worker"}
	trace := &raylineARCThinkingTrace{Source: "policy"}
	ctx := &RequestContext{
		RaylineARCDispatch: worker, RaylineARCThinking: trace,
		RaylineARCEndpointSource: "feed", RaylineARCDispatchedEndpoint: "anthropic",
		RaylineARCEndpointOrder: []string{"anthropic"}, RaylineARCEndpointsTried: []string{"anthropic"},
		DispatchedProviderOrder: []string{"anthropic"},
	}
	snapshot := snapshotReserveState(ctx, nil)
	ctx.DispatchedProviderOrder = []string{"z-ai"}
	ctx.RaylineARCDispatch = &raylinearc.WorkerManifest{ID: "glm-worker"}
	ctx.RaylineARCThinking, ctx.RaylineARCEndpointSource, ctx.RaylineARCDispatchedEndpoint = nil, "", "z-ai"
	ctx.RaylineARCEndpointOrder, ctx.RaylineARCEndpointsTried = []string{"z-ai"}, []string{"z-ai"}

	snapshot.restore(ctx, nil)
	assert.Same(t, worker, ctx.RaylineARCDispatch)
	assert.Same(t, trace, ctx.RaylineARCThinking)
	assert.Equal(t, "feed", ctx.RaylineARCEndpointSource)
	assert.Equal(t, "anthropic", ctx.RaylineARCDispatchedEndpoint)
	assert.Equal(t, []string{"anthropic"}, ctx.RaylineARCEndpointOrder)
	assert.Equal(t, []string{"anthropic"}, ctx.RaylineARCEndpointsTried)
	assert.Equal(t, []string{"anthropic"}, ctx.DispatchedProviderOrder)
}

// A held turn whose call refused waits for its re-serve: a host retry would
// take the refusal for an upstream error and ask the refusing model again.
func TestAwaitingRefusalReserve(t *testing.T) {
	refused := &semanticResponseStreamState{stop: llmprotocol.StopContentFilter}
	assert.False(t, awaitingRefusalReserve(&RequestContext{SemanticStreamState: refused}))
	assert.False(t, awaitingRefusalReserve(&RequestContext{RefusalReserve: &refusalReservePlan{},
		SemanticStreamState: &semanticResponseStreamState{}}))
	assert.False(t, awaitingRefusalReserve(&RequestContext{RefusalReserve: &refusalReservePlan{released: true},
		SemanticStreamState: refused}))
	assert.True(t, awaitingRefusalReserve(&RequestContext{RefusalReserve: &refusalReservePlan{},
		SemanticStreamState: refused}))
}

// A re-send's own anthropic-beta loses the server-side fallback betas and
// keeps the rest; with none left the header goes.
func TestStripServerSideFallbackBetaHeader(t *testing.T) {
	headers := http.Header{}
	headers.Set(anthropicBetaHeader, "prompt-caching-2024-07-31, server-side-fallback-2026-06-01")
	stripServerSideFallbackBetaHeader(headers)
	assert.Equal(t, "prompt-caching-2024-07-31", headers.Get(anthropicBetaHeader))

	headers.Set(anthropicBetaHeader, "server-side-fallback-2026-06-01")
	stripServerSideFallbackBetaHeader(headers)
	assert.Empty(t, headers.Values(anthropicBetaHeader))

	headers.Set(anthropicBetaHeader, "prompt-caching-2024-07-31")
	stripServerSideFallbackBetaHeader(headers)
	assert.Equal(t, "prompt-caching-2024-07-31", headers.Get(anthropicBetaHeader))
}

// limitByModel denies the models it names and records each usage report.
type limitByModel struct {
	denied  map[string]bool
	checks  []string
	reports []string
}

func (l *limitByModel) Name() string { return "test-limiter" }

func (l *limitByModel) Check(ctx ratelimit.Context) (*ratelimit.Decision, error) {
	l.checks = append(l.checks, ctx.Model)
	return &ratelimit.Decision{Allowed: !l.denied[ctx.Model]}, nil
}

func (l *limitByModel) Report(ctx ratelimit.Context, _ ratelimit.TokenUsage) error {
	l.reports = append(l.reports, ctx.Model)
	return nil
}

// The next arm answers to its own rate limits: one out of quota is not
// dispatched, and the turn keeps the refused arm's limit context.
func TestRefusalReserveRespectsTheNextArmsRateLimit(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	limiter := &limitByModel{denied: map[string]bool{reserveGLMArm: true}}
	router.RateLimiter = ratelimit.NewRateLimitResolver(limiter)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RateLimitCtx = &ratelimit.Context{Model: reserveClaudeArm}

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started(), "an arm out of quota was dispatched")
	assert.Empty(t, upstream.calls())
	require.NotNil(t, ctx.RateLimitCtx)
	assert.Equal(t, reserveClaudeArm, ctx.RateLimitCtx.Model)
}

// A turn the request path added to or compressed is not rebuilt.
func TestRequestContentInjected(t *testing.T) {
	assert.False(t, requestContentInjected(&RequestContext{}))
	assert.True(t, requestContentInjected(&RequestContext{RAGRetrievedContext: "doc"}))
	assert.True(t, requestContentInjected(&RequestContext{MemoryContext: "memory"}))
	assert.True(t, requestContentInjected(&RequestContext{ContextCompressionApplied: true}))
}

// A re-serve abandoned at its last bound never checks, so never consumes,
// the next arm's quota.
func TestAbandonedRefusalReserveConsumesNoQuota(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	limiter := &limitByModel{}
	router.RateLimiter = ratelimit.NewRateLimitResolver(limiter)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RefusalReserve.budget = 20 * time.Millisecond
	ctx.RefusalReserve.redecide = func(*RequestContext) (*config.ModelRef, string, error) {
		time.Sleep(40 * time.Millisecond)
		return &config.ModelRef{Model: reserveGLMArm}, "test", nil
	}

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, limiter.checks, "an abandoned re-serve consumed the next arm's quota")
}

// With a limiter, the bounds must leave the check its whole timeout: a
// re-serve with less budget left neither checks nor dispatches, so a slow
// check can never start a paid call past the budget. Without one, the same
// budget re-serves.
func TestRefusalReserveLeavesTheRateLimitCheckItsTimeout(t *testing.T) {
	for _, limited := range []bool{true, false} {
		upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
		router := reserveRouter(upstream.server.URL)
		limiter := &limitByModel{}
		if limited {
			router.RateLimiter = ratelimit.NewRateLimitResolver(limiter)
		}
		stream := NewMockStream(nil)
		ctx, _ := startReserveTurn(t, router)
		ctx.RefusalReserve.budget = ratelimit.ProviderCheckTimeout / 2

		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
		assert.Equal(t, !limited, ctx.HostRetry.started(), "limited=%v", limited)
		assert.Empty(t, limiter.checks, "limited=%v", limited)
	}
}

// A re-served arm whose own Messages stream refuses before any content block
// is a refusal, found when its stream is finalized: one re-serve is the
// most, so it reaches the client as a refusal, never cut as a host's
// failure and never re-served again.
func TestReservedArmsBareMessagesRefusalPassesThrough(t *testing.T) {
	const kimiArm, kimiHost = "kimi", "moonshotai"
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "messages_refusal", kimiHost: "content"})
	router := reserveRouter(upstream.server.URL)
	models := router.Config.ModelConfig
	kimi := models[reserveGLMArm]
	fallbacks := true
	kimi.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: []string{kimiHost}, AllowFallbacks: &fallbacks}
	kimi.ExternalModelIDs = map[string]string{"vllm": "moonshotai/kimi-k3"}
	models[kimiArm] = kimi
	glm := models[reserveGLMArm]
	glm.APIFormat = config.APIFormatAnthropic
	models[reserveGLMArm] = glm
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RefusalReserve.families = []string{"claude", "glm"}
	next := []string{reserveGLMArm, kimiArm}
	ctx.RefusalReserve.redecide = func(*RequestContext) (*config.ModelRef, string, error) {
		arm := next[0]
		next = next[1:]
		return &config.ModelRef{Model: arm}, "test", nil
	}

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the first refusal was not re-served")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Len(t, upstream.calls(), 1, "the re-served arm's refusal was re-served again")
	assert.NotContains(t, sent, "served on "+kimiHost)
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	attempts := attemptsRecord(t, sent)["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	assert.Equal(t, "refused", attempts[1].(map[string]interface{})["outcome"])
}

// A re-served arm's Chat stream that refuses before any content and closes
// with no usage line is passed through as a refusal: the stream stated its
// stop, so its host did not fail.
func TestReservedArmsChatRefusalAtCloseIsNotATimeout(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "chat_refusal"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Len(t, upstream.calls(), 1)
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	assert.NotContains(t, sent, "stream_incomplete")
	assert.Contains(t, sent, "gen-served-refusal")
}

// The re-served arm is not held: its first bytes go to the client as they
// arrive, and the client is owed keepalives only until then.
func TestReservedArmStreamsUnheld(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	plan := ctx.RefusalReserve
	assert.True(t, plan.reserved)
	assert.False(t, router.holdingForRefusalReserve(ctx), "the re-served arm is held")
	plan.lastKeepAlive = time.Now().Add(-refusalReserveKeepAlive)
	assert.True(t, router.refusalReserveKeepAliveDue(ctx), "no keepalive while the re-serve waits for its first bytes")
	assert.Positive(t, router.refusalReserveKeepAliveWait(ctx))

	for !plan.released && !ctx.StreamingComplete {
		serveNextSpliced(t, router, ctx, stream)
	}
	assert.True(t, plan.released, "the re-served arm's first bytes did not release it")
	assert.False(t, router.refusalReserveKeepAliveDue(ctx))
	assert.Zero(t, router.refusalReserveKeepAliveWait(ctx))
	drainSpliced(t, router, ctx, stream)
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "served on "+reserveGLMHost)
}

// A re-served arm's host that closes cleanly with nothing said is that host's
// failure: the arm's next host takes the turn, as without a re-serve plan.
func TestReservedArmsSilentCloseFailsOverToItsNextHost(t *testing.T) {
	const quietHost = "z-ai-quiet"
	upstream := newHostUpstream(t, map[string]string{quietHost: "keepalive_close", reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	router.Config.ResponseStreamHostRetry = true
	glm := router.Config.ModelConfig[reserveGLMArm]
	fallbacks := true
	glm.ProviderPreferences = &config.OpenRouterProviderPreferences{
		Order: []string{quietHost, reserveGLMHost}, AllowFallbacks: &fallbacks,
	}
	router.Config.ModelConfig[reserveGLMArm] = glm
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+reserveGLMHost, "the silent close did not fail over to the next host")
	assert.NotContains(t, sent, "stream_incomplete")
}

// The budget runs from the refusal itself, not from its usage: a usage line
// that arrives past the budget starts no paid call.
func TestRefusalReserveBudgetRunsFromTheRefusalSignal(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	stop := `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
		`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}` + "\n\n"
	usage := `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
		`"model":"anthropic/claude-opus-5.5","choices":[],` +
		`"usage":{"prompt_tokens":120,"completion_tokens":7,"total_tokens":127,"cost":0.0009}}` + "\n\n"
	for _, chunk := range []string{reserveRoleChunk, stop} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.False(t, ctx.RefusalReserve.firstRefusal.IsZero(), "the refusal did not start the budget")
	// The usage arrives once the budget, counted from the refusal, is spent.
	ctx.RefusalReserve.firstRefusal = time.Now().Add(-ctx.RefusalReserve.budget)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(usage, false), ctx))
	assert.False(t, ctx.HostRetry.started(), "a re-serve started past the budget")
	assert.Empty(t, upstream.calls())
}

// A hold that runs out with frames kept and the upstream quiet sends them on
// the loop's own wake, not on the next upstream chunk.
func TestLapsedHoldSendsItsFramesWithoutANewChunk(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	for _, chunk := range []string{reserveRoleChunk, reserveThinkingChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.NotEmpty(t, ctx.HostRetry.held, "the thinking was not held")
	assert.LessOrEqual(t, router.refusalReserveKeepAliveWait(ctx), ctx.RefusalReserve.hold,
		"the loop does not wake by the hold's end")

	ctx.StartTime = time.Now().Add(-2 * ctx.RefusalReserve.hold)
	require.True(t, router.refusalReserveHoldLapsed(ctx))
	assert.Positive(t, router.refusalReserveKeepAliveWait(ctx), "the lapsed hold schedules no wake")
	require.NoError(t, router.sendLapsedHoldFrames(stream, ctx))
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "weighing the request", "the kept frames were not sent")
	assert.Empty(t, ctx.HostRetry.held)
	assert.True(t, ctx.RefusalReserve.released)
	assert.False(t, router.refusalReserveHoldLapsed(ctx))
}
