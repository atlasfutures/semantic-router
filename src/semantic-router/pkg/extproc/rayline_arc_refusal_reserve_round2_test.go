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
	reports []string
}

func (l *limitByModel) Name() string { return "test-limiter" }

func (l *limitByModel) Check(ctx ratelimit.Context) (*ratelimit.Decision, error) {
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
