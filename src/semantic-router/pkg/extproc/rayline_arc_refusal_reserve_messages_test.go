//go:build !windows && cgo

package extproc

import (
	"net/http"
	"os"
	"strings"
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
		maxReserves: 1, budget: 45 * time.Second, hold: 60 * time.Second, families: []string{"claude"},
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
	// Anthropic's own split: input_tokens is the uncached input, not the
	// stream's input total, so the gateway does not price the cache reads
	// twice.
	usage := refused["usage"].(map[string]interface{})
	assert.Equal(t, float64(99000), usage["input_tokens"])
	assert.Equal(t, float64(98000), usage["cache_read_input_tokens"])
	assert.Equal(t, float64(0), usage["cache_creation_input_tokens"])
	assert.Equal(t, float64(160), usage["output_tokens"])

	// The gateway's parser is tested against this exact line
	// (memex-desktop firebase/router/tests/fixtures/vsr_attempts_cell.txt).
	start := strings.Index(sent, refusalAttemptsCommentPrefix)
	line := sent[start : start+strings.Index(sent[start:], "\n")]
	assertGolden(t, "testdata/vsr_attempts_messages_cyber.golden", line+"\n")
}

// assertGolden compares got with the file at path; VSR_UPDATE_GOLDEN=1
// rewrites the file instead.
func assertGolden(t *testing.T, path, got string) {
	t.Helper()
	if os.Getenv("VSR_UPDATE_GOLDEN") == "1" {
		require.NoError(t, os.WriteFile(path, []byte(got), 0o644))
		return
	}
	want, err := os.ReadFile(path)
	require.NoError(t, err, "missing golden %s: run with VSR_UPDATE_GOLDEN=1", path)
	assert.Equal(t, string(want), got)
}

// A Messages refusal states its stop in message_delta and publishes it only
// once the stream is finalized; the re-serve budget starts at the
// message_delta, before message_stop arrives.
func TestRefusalReserveBudgetStartsAtTheMessagesHeldStop(t *testing.T) {
	router := reserveRouter("http://127.0.0.1:1")
	claude := router.Config.ModelConfig[reserveClaudeArm]
	claude.APIFormat = config.APIFormatAnthropic
	router.Config.ModelConfig[reserveClaudeArm] = claude
	request := testNeutralRequest("auto", "route this turn")
	request.Stream = true
	limit := int64(1024)
	request.Sampling.MaxOutputTokens = &limit
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true
	ctx.StartTime = time.Now()
	_, err := router.handleEntrypointModelRouting(request, "auto", flexResendDecision, entropy.ReasoningDecision{}, reserveClaudeArm, ctx)
	require.NoError(t, err)
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	stream := NewMockStream(nil)
	// The refusal's message_delta arrives; its message_stop has not yet.
	last := reserveMessagesCyberRefusal[len(reserveMessagesCyberRefusal)-1]
	delta := last[:strings.Index(last, "event: message_stop")]
	for _, chunk := range append(append([]string(nil), reserveMessagesCyberRefusal[:len(reserveMessagesCyberRefusal)-1]...), delta) {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.True(t, heldStopIsRefusal(ctx), "the message_delta's refusal stop is not seen")

	ctx.RefusalReserve = &refusalReservePlan{budget: 45 * time.Second, hold: 60 * time.Second}
	noteRefusalSignal(ctx)
	assert.False(t, ctx.RefusalReserve.firstRefusal.IsZero(), "the held stop did not start the budget")
}

// messagesReserveTurn routes a streamed Messages turn to the claude arm,
// both arms speaking Messages, refuses it as prod did, and streams the
// re-serve from a glm host that answers as behaviour says.
func messagesReserveTurn(t *testing.T, behaviour string) (*RequestContext, *MockStream, *hostUpstream) {
	t.Helper()
	return messagesReserveTurnWithFault(t, behaviour, "")
}

// messagesReserveTurnWithFault is messagesReserveTurn with a dev fault set.
func messagesReserveTurnWithFault(t *testing.T, behaviour, fault string) (*RequestContext, *MockStream, *hostUpstream) {
	t.Helper()
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: behaviour})
	router := reserveRouter(upstream.server.URL)
	for _, arm := range []string{reserveClaudeArm, reserveGLMArm} {
		model := router.Config.ModelConfig[arm]
		model.APIFormat = config.APIFormatAnthropic
		router.Config.ModelConfig[arm] = model
	}
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
		maxReserves: 1, budget: 45 * time.Second, hold: 60 * time.Second, families: []string{"claude"},
		family: "claude", attemptStarted: time.Now(),
		redecide: func(*RequestContext) (*config.ModelRef, string, error) {
			return &config.ModelRef{Model: reserveGLMArm}, "test", nil
		},
	}
	if ctx.HostRetry == nil {
		ctx.HostRetry = &hostRetryPlan{attempts: []hostAttempt{{started: ctx.StartTime}}}
	}
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, &recordingSelectionTransaction{})
	ctx.InjectedFault = fault

	last := len(reserveMessagesCyberRefusal) - 1
	for index, chunk := range reserveMessagesCyberRefusal {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, index == last), ctx))
	}
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	drainSpliced(t, router, ctx, stream)
	return ctx, stream, upstream
}

// On a re-serve whose client and served arm both speak Messages, the stream
// passes through and its message_stop goes out before the body ends: the
// attempt record goes out ahead of it, so a client that stops reading at
// message_stop still has it.
func TestRefusalAttemptsRecordPrecedesTheMessagesTerminal(t *testing.T) {
	_, stream, _ := messagesReserveTurn(t, "messages_content")

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	record := strings.Index(sent, refusalAttemptsCommentPrefix)
	stop := strings.Index(sent, "event: message_stop")
	require.GreaterOrEqual(t, record, 0, "no attempt record was sent: %s", sent)
	require.GreaterOrEqual(t, stop, 0)
	assert.Less(t, record, stop, "the attempt record followed message_stop")
	attempts := attemptsRecord(t, sent)["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	served := attempts[1].(map[string]interface{})
	assert.Equal(t, "served", served["outcome"])
	assert.NotNil(t, served["usage"], "the served attempt's usage was not in the record")
}

// The record goes out ahead of the terminal, before the stream is finalized:
// a re-served Messages arm that refuses is still recorded as refused, from
// the stop the engine holds.
func TestRefusalAttemptsRecordNamesTheServedArmsMessagesRefusal(t *testing.T) {
	ctx, stream, _ := messagesReserveTurn(t, "messages_refusal")

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	attempts := attemptsRecord(t, sent)["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	served := attempts[1].(map[string]interface{})
	assert.Equal(t, "refused", served["outcome"])
	assert.Equal(t, turnFailureRefusal, served["failure_class"])
}

// A re-served arm whose body breaks after it stated its stop is cut there, as
// any body the Router read is. A Chat client is sent a translated stream whose
// finish and [DONE] are written only at the end: it is not told the turn
// completed, and the attempt is not recorded as served.
func TestReservedArmsBodyBrokenAfterItsStopIsCut(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "messages_length_cut"})
	router := reserveRouter(upstream.server.URL)
	glm := router.Config.ModelConfig[reserveGLMArm]
	glm.APIFormat = config.APIFormatAnthropic
	router.Config.ModelConfig[reserveGLMArm] = glm
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Len(t, upstream.calls(), 1)
	assert.NotContains(t, sent, "data: [DONE]", "the broken body reached the client as a completion: %s", sent)
	assert.Contains(t, sent, `data: {"error"`)
	assert.NotEmpty(t, ctx.ResponseFailureClass)
	attempts := attemptsRecord(t, sent)["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	assert.Equal(t, "failed", attempts[1].(map[string]interface{})["outcome"])
}
