//go:build !windows && cgo

package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A re-decision that finds no arm leaves the turn as the refused arm's: the
// refusal reaches the client and is settled once, on the turn's own line.
func TestFailedRedecisionSettlesTheRefusalOnce(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RefusalReserve.redecide = func(*RequestContext) (*config.ModelRef, string, error) {
		return nil, "", assert.AnError
	}
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	assert.Equal(t, reserveClaudeArm, ctx.VSRSelectedModel, "the failed attempt left another arm's identity")
	lines := usageLines(logs)
	require.Len(t, lines, 1, "the refused call was settled twice: %v", lines)
	assert.Nil(t, lines[0]["refusal_reserve"])
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, refusalAttemptsCommentPrefix)
}

// A re-served arm with one host gets a plan of its own: none of the refused
// arm's hosts carry over.
func TestReservedArmGetsAFreshSplicePlan(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	before := ctx.HostRetry
	before.hosts = []string{"anthropic", "amazon-bedrock"}
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started())
	assert.NotSame(t, before, ctx.HostRetry)
	assert.Empty(t, ctx.HostRetry.hosts)
	drainSpliced(t, router, ctx, stream)
}

// Past the hold, a refusal is passed through and no keepalive is owed.
func TestExpiredHoldReservesNothing(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	ctx.StartTime = time.Now().Add(-61 * time.Second)
	assert.Zero(t, router.refusalReserveKeepAliveWait(ctx), "an expired hold still polls for keepalives")
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
}
