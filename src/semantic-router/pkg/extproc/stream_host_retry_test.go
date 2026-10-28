//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A streamed turn whose host answers 200 and then sends only keepalives is
// re-sent on the arm's next host, end to end through the extproc phases: the
// arm pins two OpenRouter hosts, the cell's provider hop stalls on the
// first, and a fake OpenRouter serves the re-send by the provider object it
// is sent.

const (
	hostRetryArm     = "haiku"
	hostRetryHostA   = "anthropic"
	hostRetryHostB   = "amazon-bedrock"
	hostRetryModelID = "anthropic/claude-haiku-5.5"
)

// hostRetryPreContentChunk is the stalled host's response-start frame: no
// content, so it is held from the client.
const hostRetryPreContentChunk = `data: {"id":"gen-stalled","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-haiku-5.5","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"

func hostRetryContentStream(text string) string {
	return `data: {"id":"gen-served","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-haiku-5.5",` +
		`"provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"role":"assistant","content":"` + text + `"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"gen-served","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-haiku-5.5",` +
		`"provider":"Amazon Bedrock","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}` + "\n\n" +
		"data: [DONE]\n\n"
}

// hostUpstream is a fake OpenRouter that answers by the first host the
// provider object orders: "content" streams a reply, "stall" sends a
// keepalive and nothing else, and a number answers with that status.
type hostUpstream struct {
	server    *httptest.Server
	mu        sync.Mutex
	providers []map[string]json.RawMessage
	behaviour map[string]string
}

func newHostUpstream(t *testing.T, behaviour map[string]string) *hostUpstream {
	t.Helper()
	upstream := &hostUpstream{behaviour: behaviour}
	upstream.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received, _ := io.ReadAll(r.Body)
		var body struct {
			Provider map[string]json.RawMessage `json:"provider"`
		}
		_ = json.Unmarshal(received, &body)
		var order []string
		_ = json.Unmarshal(body.Provider["order"], &order)
		upstream.mu.Lock()
		upstream.providers = append(upstream.providers, body.Provider)
		upstream.mu.Unlock()
		host := ""
		if len(order) > 0 {
			host = order[0]
		}
		switch upstream.behaviour[host] {
		case "content":
			w.Header().Set("content-type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, hostRetryContentStream("served on "+host))
		case "sse_error":
			w.Header().Set("content-type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `data: {"error":{"code":502,"message":"provider overloaded"}}`+"\n\n")
		case "stall":
			w.Header().Set("content-type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, openRouterKeepaliveChunk)
			w.(http.Flusher).Flush()
			<-r.Context().Done()
		default:
			w.Header().Set("content-type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"no capacity"}}`)
		}
	}))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func (upstream *hostUpstream) calls() []map[string]json.RawMessage {
	upstream.mu.Lock()
	defer upstream.mu.Unlock()
	return append([]map[string]json.RawMessage(nil), upstream.providers...)
}

// hostRetryRouter is a cell with one arm pinned to two hosts in order, with
// OpenRouter's own fallbacks inside the list, on one OpenRouter backend.
func hostRetryRouter(baseURL string, hostRetry bool, armFirstContentSec int, hosts ...string) *OpenAIRouter {
	if len(hosts) == 0 {
		hosts = []string{hostRetryHostA, hostRetryHostB}
	}
	fallbacks := true
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			ResponseStreamFirstContentSec: 90,
			ResponseStreamIdleSec:         60,
			ResponseStreamHostRetry:       hostRetry,
		},
		BackendModels: config.BackendModels{
			DefaultModel: hostRetryArm,
			ModelConfig: map[string]config.ModelParams{
				hostRetryArm: {
					PreferredEndpoints: []string{"openrouter"},
					APIFormat:          config.APIFormatOpenAI,
					ProviderPreferences: &config.OpenRouterProviderPreferences{
						Order: hosts, AllowFallbacks: &fallbacks,
						Ignore: []string{"anthropic/flex"},
					},
					FirstContentSec:  armFirstContentSec,
					AccessKeys:       map[string]string{"openrouter": "sk-test"},
					ExternalModelIDs: map[string]string{"vllm": hostRetryModelID},
					Pricing:          config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 5},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "openrouter", Address: "127.0.0.1", Port: 1,
				ProviderProfileName: "openrouter",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"openrouter": {Type: "openrouter", BaseURL: baseURL + "/api/v1"},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// startStalledTurn routes a streamed turn to the arm, answers its headers
// with a 200 stream, sends it chunks from the first host, and then lets its
// first-content limit pass with nothing more arriving, as the loop's timer
// finds it.
func startStalledTurn(
	t *testing.T,
	router *OpenAIRouter,
	stream *MockStream,
	chunks ...string,
) (*RequestContext, *recordingSelectionTransaction) {
	t.Helper()
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.True(t, ctx.IsStreamingResponse)
	// A policy turn's selection, still open once its headers are in.
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)
	for _, chunk := range chunks {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	ctx.StreamContentSince = time.Now().Add(-91 * time.Second)
	require.True(t, router.streamSilenceExceeded(ctx))
	require.NoError(t, router.endSilentStream(stream, ctx))
	return ctx, transaction
}

// serveNextSpliced serves the next thing the re-send in flight says.
func serveNextSpliced(t *testing.T, router *OpenAIRouter, ctx *RequestContext, stream *MockStream) hostRetryChunk {
	t.Helper()
	chunks := hostRetryChunks(ctx)
	require.NotNil(t, chunks, "no re-send is in flight")
	select {
	case chunk := <-chunks:
		require.NoError(t, router.serveHostRetryChunk(stream, ctx, chunk))
		return chunk
	case <-time.After(5 * time.Second):
		t.Fatal("the re-send said nothing")
	}
	return hostRetryChunk{}
}

func drainSpliced(t *testing.T, router *OpenAIRouter, ctx *RequestContext, stream *MockStream) {
	t.Helper()
	for !ctx.StreamingComplete {
		serveNextSpliced(t, router, ctx, stream)
	}
}

// clientBytes is everything the client was sent, and whether the last reply
// ended the response.
func clientBytes(stream *MockStream) (string, bool) {
	var sent strings.Builder
	ended := false
	for _, response := range stream.Responses {
		streamed := response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
		if streamed == nil {
			continue
		}
		sent.Write(streamed.GetBody())
		ended = streamed.GetEndOfStream()
	}
	return sent.String(), ended
}

func providerList(t *testing.T, provider map[string]json.RawMessage, field string) []string {
	t.Helper()
	var values []string
	if raw, ok := provider[field]; ok {
		require.NoError(t, json.Unmarshal(raw, &values))
	}
	return values
}

// The first host stalls behind keepalives; the second serves, and the client
// gets its stream on the same response, with nothing of the first host's.
// The arm and the lease stay; each call writes its own usage line.
func TestStalledHostIsRetriedOnTheNextHost(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk, openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started(), "the stall was not re-sent")
	assert.False(t, ctx.StreamingComplete)
	assert.Equal(t, 0, transaction.aborts, "the lease was released for a turn the next host is serving")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended, "the re-sent stream did not end the response")
	assert.Contains(t, sent, "served on "+hostRetryHostB)
	assert.NotContains(t, sent, "gen-stalled", "the stalled host's held frame reached the client")
	assert.Contains(t, sent, strings.TrimSpace(string(hostRetryHoldFrame)), "the held frame was not replaced by a comment")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)

	calls := upstream.calls()
	require.Len(t, calls, 1, "the re-send was not made exactly once")
	assert.Equal(t, []string{hostRetryHostB}, providerList(t, calls[0], "order"))
	assert.Equal(t, []string{"anthropic/flex", hostRetryHostA}, providerList(t, calls[0], "ignore"))
	assert.JSONEq(t, "true", string(calls[0]["allow_fallbacks"]), "the pin's other members were not kept")

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "stalled call", lines[0], map[string]interface{}{
		"host_retry": hostRetryAttempt, "host_attempt": 1, "upstream_host": hostRetryHostA,
		"upstream_host_source": hostSourceOrder, "failure_class": turnFailureTimeout,
		"failure_detail": streamFirstContentTimeoutCode, "content_sent_before_failure": false,
		"usage_source": usageSourceUnknown,
	})
	assert.Nil(t, lines[0]["first_content_ms"])
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"host_retry": hostRetryServed, "host_attempt": 2, "upstream_host": hostRetryHostB,
		"upstream_host_source": hostSourceResponse, "upstream_response_id": "gen-served",
		"upstream_attempts": 1, "provider_reported_cost": 0.00004,
	})
	assert.Nil(t, lines[1]["failure_class"])
	assert.NotNil(t, lines[1]["first_content_ms"], "the served call's first content was not logged")
	assert.NotNil(t, lines[1]["host_attempt_started_ms"])
	if tried := lines[1]["provider_endpoints_tried"]; !sameJSON(t, mustJSONRaw(t, tried), `["anthropic","amazon-bedrock"]`) {
		t.Fatalf("provider_endpoints_tried = %v", tried)
	}
	event := findLogEvent(t, logs, "stream_host_retry")
	assertFields(t, "retry event", event, map[string]interface{}{"stalled_host": hostRetryHostA, "attempt": 1})
}

// Every host stalls: the turn ends as it does without the retry, a timeout
// with nothing sent, and its lease is released.
func TestEveryHostStallingFailsTheTurnAsBefore(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "stall"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started())
	for {
		if chunk := serveNextSpliced(t, router, ctx, stream); len(chunk.body) > 0 {
			break
		}
	}
	// The original call's keepalives still arrive, and reach no one.
	before := len(stream.Responses)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	require.Len(t, stream.Responses, before+1)
	dropped := stream.Responses[before].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	assert.Empty(t, dropped.GetBody())

	ctx.StreamContentSince = time.Now().Add(-91 * time.Second)
	require.NoError(t, router.endSilentStream(stream, ctx))

	require.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, streamFirstContentTimeoutCode)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.False(t, *ctx.ContentSentBeforeFailure)
	assert.Equal(t, 1, transaction.aborts, "the lease must be released once every host has failed")
	assert.Len(t, upstream.calls(), 1)
	lines := usageLines(logs)
	require.Len(t, lines, 2)
	assertFields(t, "last call", lines[1], map[string]interface{}{
		"host_retry": hostRetryServed, "upstream_host": hostRetryHostB, "failure_class": turnFailureTimeout,
	})
}

// A re-send the next host refuses fails the turn the same way when no host
// is left.
func TestRefusedReSendWithNoHostLeftFailsTheTurn(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, openRouterKeepaliveChunk)
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "sent no content in time")
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
}

// Content already reached the client: the turn is never re-sent, and a
// later silence ends it at the idle limit as before.
func TestContentAlreadySentIsNeverRetried(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(endlessUpstreamChunk(1), false), ctx))
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "still thinking", "content on the only call was held")

	ctx.StreamContentSince = time.Now().Add(-61 * time.Second)
	require.NoError(t, router.endSilentStream(stream, ctx))

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Empty(t, upstream.calls())
}

// Flag off: nothing is staged or held, and the stall is cut as before.
func TestHostRetryOffLeavesTheStallCutAsBefore(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk)

	assert.Nil(t, ctx.HostRetry)
	sent, ended := clientBytes(stream)
	assert.Contains(t, sent, "gen-stalled", "a frame was held with the retry off")
	assert.True(t, ended)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
	assert.Empty(t, upstream.calls())
}

// An arm's own first-content limit replaces the global one; an arm without
// one keeps the global limit.
func TestArmFirstContentLimitOverridesTheGlobalOne(t *testing.T) {
	router := hostRetryRouter("http://127.0.0.1:1", false, 30)
	ctx, _ := silentStreamContext(t, 45*time.Second)
	ctx.RequestModel = hostRetryArm
	assert.Equal(t, 30*time.Second, router.streamSilenceLimit(ctx))
	assert.True(t, router.streamSilenceExceeded(ctx))

	ctx.StreamContentSeen = true
	assert.Equal(t, 60*time.Second, router.streamSilenceLimit(ctx), "the arm's limit applied after content")

	global := hostRetryRouter("http://127.0.0.1:1", false, 0)
	ctx.StreamContentSeen = false
	assert.Equal(t, 90*time.Second, global.streamSilenceLimit(ctx))
	assert.False(t, global.streamSilenceExceeded(ctx))
}

// The stalled host is the one the stream named, when it named one of the
// arm's hosts; otherwise the first host not yet tried.
func TestStalledHostIsTheOneTheStreamNamed(t *testing.T) {
	plan := &hostRetryPlan{hosts: []string{"anthropic", "amazon-bedrock", "google-vertex"}, attempts: []hostAttempt{{}}}
	ctx := &RequestContext{SemanticStreamState: &semanticResponseStreamState{provider: "Amazon Bedrock"}}
	host, source := plan.currentHost(ctx)
	assert.Equal(t, "amazon-bedrock", host)
	assert.Equal(t, hostSourceResponse, source)

	ctx.SemanticStreamState.provider = "Somewhere Else"
	host, source = plan.currentHost(ctx)
	assert.Equal(t, "anthropic", host)
	assert.Equal(t, hostSourceOrder, source)
}

func TestPinnedHostsNeverWidenTheOnlyList(t *testing.T) {
	body := []byte(`{"provider":{"order":["a","b"],"only":["b","c","d"],"ignore":["d"]}}`)
	assert.Equal(t, []string{"b", "c"}, pinnedHosts(body))
	assert.Equal(t, []string{"a", "b"}, pinnedHosts([]byte(`{"provider":{"order":["a","b"]}}`)))
	assert.Empty(t, pinnedHosts([]byte(`{"model":"m"}`)))
	// An order host outside only is no host the pin reaches.
	assert.Equal(t, []string{"a"}, pinnedHosts([]byte(`{"provider":{"order":["a","b"],"only":["a"]}}`)))

	narrowed, ok := withHostsRemoved(body, []string{"b"})
	require.True(t, ok)
	var wire struct {
		Provider map[string][]string `json:"provider"`
	}
	require.NoError(t, json.Unmarshal(narrowed, &wire))
	assert.Equal(t, []string{"a"}, wire.Provider["order"])
	assert.Equal(t, []string{"c", "d"}, wire.Provider["only"])
	assert.Equal(t, []string{"d", "b"}, wire.Provider["ignore"])
}

// With no first-content limit no retry can fire, so nothing is held.
func TestNothingIsHeldWithoutAFirstContentLimit(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	router.Config.ResponseStreamFirstContentSec = 0
	stream := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(hostRetryPreContentChunk, false), ctx))
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "gen-stalled")
}

// A codec error the stalled call raised does not fail the re-send that
// serves the turn.
func TestStalledCallsStreamErrorDoesNotFailTheReSend(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	ctx.StreamingAborted = true
	ctx.SemanticStreamErr = assert.AnError
	ctx.StreamContentSince = time.Now().Add(-91 * time.Second)
	require.NoError(t, router.endSilentStream(stream, ctx))
	require.True(t, ctx.HostRetry.started())
	drainSpliced(t, router, ctx, stream)

	assert.Empty(t, ctx.ResponseFailureClass)
	lines := usageLines(logs)
	require.Len(t, lines, 2)
	assert.Nil(t, lines[1]["failure_class"])
}

// A re-send that fails in-band (an SSE error) before any content moves on to
// the next host; its error never reaches the client.
func TestInBandErrorBeforeContentMovesToTheNextHost(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "sse_error", "google-vertex": "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0, hostRetryHostA, hostRetryHostB, "google-vertex")
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, openRouterKeepaliveChunk)
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on google-vertex")
	assert.NotContains(t, sent, "provider overloaded")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)
	calls := upstream.calls()
	require.Len(t, calls, 2)
	assert.Equal(t, []string{"google-vertex"}, providerList(t, calls[1], "order"))
	assert.Equal(t, []string{"anthropic/flex", hostRetryHostA, hostRetryHostB}, providerList(t, calls[1], "ignore"))
	lines := usageLines(logs)
	require.Len(t, lines, 3)
	assertFields(t, "failed re-send", lines[1], map[string]interface{}{
		"host_retry": hostRetryAttempt, "host_attempt": 2, "failure_class": turnFailureUpstreamError,
		"content_sent_before_failure": false,
	})
	assertFields(t, "served", lines[2], map[string]interface{}{"host_attempt": 3, "upstream_host": "google-vertex"})
}

// A stalled Messages call states its input at message_start and nothing
// after; its line keeps the input and leaves the output unknown.
func TestStalledCallKeepsTheInputItStated(t *testing.T) {
	input := int64(20)
	usage := partialStreamUsage(llmprotocol.Usage{
		State:      llmprotocol.UsageAvailable,
		InputTotal: llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageAuthoritative},
	})
	assert.True(t, usage.promptTokensReported)
	assert.Equal(t, 20, usage.promptTokens)
	assert.False(t, usage.completionTokensReported)
}

// Once the turn has spent its retry budget, no further host is tried and the
// stall is cut as before.
func TestSpentRetryBudgetCutsTheStall(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	router.Config.ResponseStreamHostRetryBudgetSec = 60
	stream := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	ctx.StartTime = time.Now().Add(-91 * time.Second)
	ctx.StreamContentSince = ctx.StartTime
	require.NoError(t, router.endSilentStream(stream, ctx))

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Empty(t, upstream.calls())
}

// The exchange ends while the call's opening frames are held: nothing but
// comments reached the client, so no content was sent before the failure.
func TestHeldFramesAreNotContentSentWhenTheExchangeEnds(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(hostRetryPreContentChunk, false), ctx))
	require.NotEmpty(t, ctx.HostRetry.held)

	router.finalizeEndedStream(ctx, io.EOF)

	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.False(t, *ctx.ContentSentBeforeFailure)
}

// A re-send in flight waits for its first content only as long as the
// turn's retry budget has left.
func TestRetryBudgetShortensTheReSendsWait(t *testing.T) {
	router := hostRetryRouter("http://127.0.0.1:1", true, 30)
	router.Config.ResponseStreamHostRetryBudgetSec = 45
	ctx, _ := silentStreamContext(t, 0)
	ctx.RequestModel = hostRetryArm
	ctx.StartTime = time.Now().Add(-30 * time.Second)
	ctx.StreamContentSince = time.Now()
	assert.Equal(t, 30*time.Second, router.streamSilenceLimit(ctx), "the first call was capped")

	ctx.HostRetry = &hostRetryPlan{splice: &hostRetrySplice{}}
	assert.InDelta(t, (15 * time.Second).Seconds(), router.streamSilenceLimit(ctx).Seconds(), 1)
	// 46 s into the turn, the re-send sent at 30 s has been silent 16 s.
	ctx.StartTime = time.Now().Add(-46 * time.Second)
	ctx.StreamContentSince = time.Now().Add(-16 * time.Second)
	assert.True(t, router.streamSilenceExceeded(ctx))
}

// A tool call's bare start is no content: an in-band error after it still
// moves the turn to the next host.
func TestABareToolCallStartIsNotContent(t *testing.T) {
	state := &semanticResponseStreamState{items: map[int]*semanticStreamItem{0: {toolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "read"}}}}
	assert.False(t, state.carriesContent())
	state.items[0].toolCall.Arguments = "{}"
	assert.False(t, state.carriesContent(), "a Messages tool_use start's input {} was taken as content")
	state.items[0].completed = true
	assert.True(t, state.carriesContent(), "a completed call without arguments carries content")
	state.items[0].completed = false
	state.items[0].toolCall.Arguments = `{"path":`
	assert.True(t, state.carriesContent())
}

// An arm that turns its own host retry off is cut as before while the global
// flag is on: a Claude arm's silence is its thinking.
func TestArmHostRetryOffOverridesTheGlobalFlag(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, true, 0)
	off := false
	params := router.Config.ModelConfig[hostRetryArm]
	params.HostRetry = &off
	router.Config.ModelConfig[hostRetryArm] = params
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk)

	assert.Nil(t, ctx.HostRetry)
	sent, ended := clientBytes(stream)
	assert.Contains(t, sent, "gen-stalled", "a frame was held for an arm with its retry off")
	assert.True(t, ended)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
	assert.Empty(t, upstream.calls())
}

// A Chat stream names a tool call in its first delta, with no arguments:
// that is the call's start, so the frames stay held and the turn can still
// be re-sent.
func TestABareChatToolCallDeltaIsNotContent(t *testing.T) {
	ctx := &RequestContext{}
	observeStreamContent(ctx, []llmprotocol.Event{{
		Type: llmprotocol.EventToolCallDelta, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "read"},
	}})
	assert.False(t, ctx.StreamContentSeen)
	observeStreamContent(ctx, []llmprotocol.Event{{
		Type: llmprotocol.EventToolCallDelta, ToolCall: &llmprotocol.ToolCall{Arguments: `{"path":`},
	}})
	assert.True(t, ctx.StreamContentSeen)
}
