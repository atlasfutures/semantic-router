//go:build !windows && cgo

package extproc

import (
	"net/http"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// The prod shape, seen 3 times on claude-opus-5.5 (2026-10-29..31): a
// streamed Anthropic Messages response that sends thinking only and then
// stops with stop_reason refusal, stop_details {type: refusal, category:
// cyber}. The cell's neutral stop is content_filter. The client had been
// sent nothing but thinking, so a hold until the first text or tool call
// catches it: the turn is re-served, and the attempt record carries the
// signal content_filter and the category cyber.
var reserveMessagesCyberRefusal = []string{
	"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_refused\",\"type\":\"message\"," +
		"\"role\":\"assistant\",\"model\":\"claude-opus-5.5\",\"content\":[],\"stop_reason\":null," +
		"\"usage\":{\"input_tokens\":99000,\"cache_read_input_tokens\":98000,\"output_tokens\":1}}}\n\n",
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"considering the binary\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"c2lnbmF0dXJl\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_sequence\":null," +
		"\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\"}},\"usage\":{\"output_tokens\":160}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
}

func TestRefusalReserve_StopDetailsCyberAfterThinking(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	claude := router.Config.ModelConfig[reserveClaudeArm]
	claude.APIFormat = config.APIFormatAnthropic
	router.Config.ModelConfig[reserveClaudeArm] = claude
	stream := NewMockStream(nil)

	request := testNeutralRequest("auto", "route this turn")
	request.Stream = true
	limit := int64(1024)
	request.Sampling.MaxOutputTokens = &limit
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true
	ctx.StartTime = time.Now()
	_, err := router.handleEntrypointModelRouting(request, "auto", flexResendDecision, entropy.ReasoningDecision{}, reserveClaudeArm, ctx)
	require.NoError(t, err)
	ctx.ProcessingStartTime = ctx.StartTime
	ctx.RaylineARCRawBody = []byte(`{"model":"auto","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"route this turn"}]}`)
	ctx.VSRSelectedModel = reserveClaudeArm
	ctx.RefusalReserve = &refusalReservePlan{
		maxReserves: 2, budget: 45 * time.Second, hold: 60 * time.Second, families: []string{"claude"},
		family: "claude", attemptStarted: time.Now(),
		redecide: func(*RequestContext) (*config.ModelRef, string, error) {
			return &config.ModelRef{Model: reserveGLMArm}, "test", nil
		},
	}
	if ctx.HostRetry == nil {
		ctx.HostRetry = &hostRetryPlan{attempts: []hostAttempt{{started: ctx.StartTime}}}
	}
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)

	last := len(reserveMessagesCyberRefusal) - 1
	for index, chunk := range reserveMessagesCyberRefusal {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, index == last), ctx))
		if index < last {
			assert.True(t, router.holdingForRefusalReserve(ctx), "chunk %d released the hold", index)
		}
	}
	require.True(t, ctx.HostRetry.started(), "the cyber refusal after thinking was not re-served")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	frames := withoutAttemptsRecord(sent)
	assert.Contains(t, frames, "served on "+reserveGLMHost)
	assert.NotContains(t, frames, "considering the binary", "the refused call's thinking reached the client")
	assert.NotContains(t, frames, "msg_refused")
	assert.NotContains(t, frames, "refusal")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)

	record := attemptsRecord(t, sent)
	attempts := record["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	refused := attempts[0].(map[string]interface{})
	assert.Equal(t, "refused", refused["outcome"])
	assert.Equal(t, "content_filter", refused["signal"])
	assert.Equal(t, "cyber", refused["stop_category"])
	assert.Equal(t, "served", attempts[1].(map[string]interface{})["outcome"])
}
