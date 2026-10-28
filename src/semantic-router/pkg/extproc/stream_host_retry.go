package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A streamed turn's re-send on the arm's next provider host
// (global.router.response_stream.host_retry).
//
// An arm can pin an ordered list of equivalent OpenRouter hosts. OpenRouter
// moves down the list by itself on a connection error or a non-2xx before it
// commits, but not on a host that answers 200 and then sends only keepalives:
// claude-haiku-5.5 on the prod cell 2026-10-28. The first-content limit cuts
// such a turn, and the client's retry is offered the same arm, which stalls
// again, because the fallback that would exclude the arm is off.
//
// So when the first-content limit fires and no content has reached the
// client, the Router sends the same body again with the hosts tried so far
// taken out of the pin (provider order and only narrowed to the hosts left,
// the tried ones added to ignore), and the new call's stream becomes the rest
// of the response the client already has open. The arm, the decision and the
// episode lease stay as they are: nothing is excluded and nothing is decided
// again. Only when every host has stalled or failed does the turn end as a
// first-content timeout, exactly as without the retry.
//
// The splice needs FULL_DUPLEX_STREAMED. In that mode the Router's body
// replies are the response body and Envoy applies no message_timeout, so the
// Router can keep answering the original call's keepalives with nothing while
// it streams the new call's chunks itself, from the loop goroutine like every
// other send (processor_stream_receive.go). The original call is left to
// Envoy; it ends when the response does.
//
// Until the first content, the frames the client would get (a response-start
// frame, an item's start) are held and SSE comments are sent in their place,
// so a re-send can start a fresh stream that never repeats them. The first
// content releases what was held. A turn with no host left is not held.
//
// The stalled host is the one the stream named (the provider OpenRouter
// reports) when it named one of the arm's hosts; otherwise it is the first
// host of the list not yet tried, as OpenRouter would have tried it first.
//
// Each stalled call writes its own usage line (host_retry: host_attempt),
// settled as a timeout with nothing sent and no usage stated; the call that
// took over writes the turn's ordinary line (host_retry: host_retry).

const (
	hostRetryAttempt = "host_attempt"
	hostRetryServed  = "host_retry"

	// The sources of a call's host, as logged (upstream_host_source).
	hostSourceResponse = "response"
	hostSourceOrder    = "order"

	hostRetryReadSize = 32 << 10
	// hostRetryMaxHeld bounds the frames held from the client. Past it they
	// are sent, and the call can no longer be replaced.
	hostRetryMaxHeld = 1 << 20
)

// hostRetryHoldFrame is what the client is sent in place of held frames: an
// SSE comment, which every streamed wire format ignores.
var hostRetryHoldFrame = []byte(": ROUTER PROCESSING\n\n")

// hostRetryPlan is the re-send staged for a dispatch whose pin lists more
// than one host, and the state of the retries once one is under way.
type hostRetryPlan struct {
	dispatch   *providerDispatch
	dispatched []byte
	// hosts are the pin's hosts in the order OpenRouter tries them.
	hosts []string
	// attempts are the calls made so far; the last is the one in flight.
	attempts []hostAttempt
	// held is what the client has not yet been sent of the call in flight.
	held []byte
	// splice is the re-sent call being streamed to the client, nil before
	// the first re-send.
	splice *hostRetrySplice
	// exhausted says the last call failed before any content and no host is
	// left: the turn ends as a first-content timeout.
	exhausted bool
	// exposed says frames of the call in flight reached the client, so a
	// re-send would repeat them: it is not replaced.
	exposed bool
	// trailersPending says the original call's trailers arrived while a
	// re-send was serving; their reply goes out when the re-send ends.
	trailersPending bool
}

type hostAttempt struct {
	started time.Time
	// host and source are set once the call is settled or recorded.
	host   string
	source string
}

// hostRetryChunk is one thing the re-sent call said: its status, a body
// chunk, its end, or the error that ended it.
type hostRetryChunk struct {
	headers     bool
	status      int
	contentType string
	body        []byte
	end         bool
	err         error
}

type hostRetrySplice struct {
	chunks chan hostRetryChunk
	cancel context.CancelFunc
	// answered says the call's status was a 2xx stream.
	answered bool
}

// stageHostRetry keeps what a re-send needs when the dispatched body pins
// more than one OpenRouter host. Any other dispatch is left as it was.
func (r *OpenAIRouter) stageHostRetry(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.HostRetry = nil
	if r == nil || r.Config == nil || dispatch == nil || !r.Config.HostRetryForModel(dispatch.logicalModel) ||
		ctx.LooperRequest || ctx.FlexResend != nil || !providerIsOpenRouter(dispatch.profile) {
		return
	}
	hosts := pinnedHosts(body)
	if len(hosts) < 2 {
		return
	}
	ctx.HostRetry = &hostRetryPlan{
		dispatch:   dispatch,
		dispatched: body,
		hosts:      hosts,
		attempts:   []hostAttempt{{started: ctx.StartTime}},
	}
}

// pinnedHosts are the hosts a body's provider object lets OpenRouter try, in
// the order it tries them, without the ones ignore names: with an only list,
// its hosts, those order names first in order's order; without one, order.
func pinnedHosts(body []byte) []string {
	var wire struct {
		Provider struct {
			Order  []string `json:"order"`
			Only   []string `json:"only"`
			Ignore []string `json:"ignore"`
		} `json:"provider"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return nil
	}
	listed := wire.Provider.Order
	if len(wire.Provider.Only) > 0 {
		listed = []string{}
		for _, host := range wire.Provider.Order {
			if containsFold(wire.Provider.Only, strings.TrimSpace(host)) {
				listed = append(listed, host)
			}
		}
		listed = append(listed, wire.Provider.Only...)
	}
	hosts := []string{}
	for _, host := range listed {
		host = strings.TrimSpace(host)
		if host == "" || containsFold(hosts, host) || containsFold(wire.Provider.Ignore, host) {
			continue
		}
		hosts = append(hosts, host)
	}
	return hosts
}

func containsFold(values []string, value string) bool {
	return slices.ContainsFunc(values, func(candidate string) bool {
		return strings.EqualFold(strings.TrimSpace(candidate), value)
	})
}

// started reports that a re-send is under way.
func (plan *hostRetryPlan) started() bool {
	return plan != nil && plan.splice != nil
}

// tried are the hosts of the settled calls.
func (plan *hostRetryPlan) tried() []string {
	tried := []string{}
	for _, attempt := range plan.attempts[:len(plan.attempts)-1] {
		tried = append(tried, attempt.host)
	}
	return tried
}

// remaining are the hosts not yet tried, in order.
func (plan *hostRetryPlan) remaining() []string {
	tried := plan.tried()
	left := []string{}
	for _, host := range plan.hosts {
		if !containsFold(tried, host) {
			left = append(left, host)
		}
	}
	return left
}

// currentHost is the host the call in flight is attributed to: the one the
// stream named among the hosts it may reach, else the first of them.
func (plan *hostRetryPlan) currentHost(ctx *RequestContext) (string, string) {
	left := plan.remaining()
	if len(left) == 0 {
		return "", ""
	}
	named := ""
	if ctx.SemanticStreamState != nil {
		named = raylinearc.ProviderSlugKey(ctx.SemanticStreamState.provider)
	}
	if named != "" {
		for _, host := range left {
			base, _, _ := strings.Cut(host, "/")
			if raylinearc.ProviderSlugKey(base) == named {
				return host, hostSourceResponse
			}
		}
	}
	return left[0], hostSourceOrder
}

// holding reports that the call in flight's frames are held from the client:
// nothing has been sent of it yet, and another host could still replace it,
// which takes a first-content limit to fire.
func (r *OpenAIRouter) holdingForHostRetry(ctx *RequestContext) bool {
	plan := ctx.HostRetry
	return plan != nil && !plan.exposed && streamSilenceTimerArmed(ctx) && !ctx.StreamContentSeen &&
		!ctx.StreamingComplete && r.streamSilenceLimit(ctx) > 0 && len(plan.remaining()) > 1
}

// holdHostRetryFrames keeps a reply's frames from the client while the call
// could still be replaced, and sends what was held with the first reply
// that is not held.
func (r *OpenAIRouter) holdHostRetryFrames(ctx *RequestContext, response *ext_proc.ProcessingResponse) *ext_proc.ProcessingResponse {
	plan := ctx.HostRetry
	streamed := response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	if plan == nil || streamed == nil {
		return response
	}
	if r.holdingForHostRetry(ctx) && len(plan.held)+len(streamed.Body) > hostRetryMaxHeld {
		plan.exposed = true
	}
	if r.holdingForHostRetry(ctx) {
		if len(streamed.Body) > 0 {
			plan.held = append(plan.held, streamed.Body...)
			streamed.Body = bytes.Clone(hostRetryHoldFrame)
		}
		return response
	}
	if len(plan.held) > 0 {
		streamed.Body = append(plan.held, streamed.Body...)
		plan.held = nil
	}
	if len(streamed.Body) > 0 {
		plan.exposed = true
	}
	return response
}

// retryOnNextHost starts the re-send of a turn whose call in flight failed
// before any content (class and detail say how: a silence, or a re-send's
// own refusal), or reports that there is none: the turn then ends as it would
// without it.
func (r *OpenAIRouter) retryOnNextHost(ctx *RequestContext, class, detail string) bool {
	plan := ctx.HostRetry
	if plan == nil || plan.exposed || ctx.StreamContentSeen || ctx.StreamingComplete || !streamSilenceTimerArmed(ctx) {
		return false
	}
	host, source := plan.currentHost(ctx)
	if host == "" {
		return false
	}
	current := &plan.attempts[len(plan.attempts)-1]
	current.host, current.source = host, source
	if budget := r.Config.ResponseStreamHostRetryBudgetSec; budget > 0 && !ctx.StartTime.IsZero() &&
		time.Since(ctx.StartTime) >= time.Duration(budget)*time.Second {
		logging.ComponentWarnEvent("extproc", "stream_host_retry_unavailable", map[string]interface{}{
			"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel, "reason": "budget_spent",
			"attempt": len(plan.attempts), "budget_ms": int64(budget) * 1000,
		})
		return false
	}
	tried := append(plan.tried(), host)
	next := []string{}
	for _, candidate := range plan.hosts {
		if !containsFold(tried, candidate) {
			next = append(next, candidate)
		}
	}
	if len(next) == 0 {
		return false
	}
	body, ok := withHostsRemoved(plan.dispatched, tried)
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel,
		"stalled_host": host, "stalled_host_source": source, "next_hosts": next,
		"attempt": len(plan.attempts), "trigger": detail,
		"silent_ms": time.Since(ctx.StreamContentSince).Milliseconds(),
		"limit_ms":  r.streamSilenceLimit(ctx).Milliseconds(),
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	var call *flexTwinCall
	unavailable := "body_unpinned"
	if ok {
		call, unavailable = r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, body)
	}
	if call == nil {
		fields["reason"] = unavailable
		logging.ComponentWarnEvent("extproc", "stream_host_retry_unavailable", fields)
		return false
	}
	logging.ComponentEvent("extproc", "stream_host_retry", fields)

	r.settleStalledHost(ctx, class, detail)
	plan.stop()
	plan.attempts = append(plan.attempts, hostAttempt{started: time.Now()})
	plan.held, plan.exposed = nil, false
	plan.dispatched = body
	resetStreamForHostRetry(ctx, body, plan.dispatch)
	r.initializeSemanticResponseStream(ctx)
	plan.splice = startHostRetrySplice(ctx, call)
	return true
}

// failHost ends a re-send that failed before any content: the turn moves to
// the next host, or, with none left, ends as a first-content timeout. It
// reports whether a re-send took over.
func (r *OpenAIRouter) failHost(ctx *RequestContext, class, detail string) bool {
	plan := ctx.HostRetry
	logging.ComponentWarnEvent("extproc", "stream_host_retry_failed", map[string]interface{}{
		"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel,
		"attempt": len(plan.attempts), "detail": detail,
	})
	if r.retryOnNextHost(ctx, class, detail) {
		return true
	}
	plan.exhausted = true
	return false
}

// withHostsRemoved is the dispatched body with the tried hosts taken out of
// the provider object's order and only, each narrowed on its own so neither
// list ever admits a host it did not, and added to ignore. Every other member
// of the body and of the object stays as it was.
func withHostsRemoved(body []byte, tried []string) ([]byte, bool) {
	var members map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil {
		return nil, false
	}
	var provider map[string]json.RawMessage
	if json.Unmarshal(members[providerPreferencesField], &provider) != nil || provider == nil {
		return nil, false
	}
	for _, field := range []string{"order", "only"} {
		raw, set := provider[field]
		if !set {
			continue
		}
		var listed []string
		if json.Unmarshal(raw, &listed) != nil {
			return nil, false
		}
		kept := []string{}
		for _, host := range listed {
			if !containsFold(tried, strings.TrimSpace(host)) {
				kept = append(kept, host)
			}
		}
		provider[field] = mustMarshalStrings(kept)
	}
	var ignore []string
	if raw, set := provider["ignore"]; set && json.Unmarshal(raw, &ignore) != nil {
		return nil, false
	}
	for _, host := range tried {
		if !containsFold(ignore, host) {
			ignore = append(ignore, host)
		}
	}
	provider["ignore"] = mustMarshalStrings(ignore)
	encoded, err := json.Marshal(provider)
	if err != nil {
		return nil, false
	}
	members[providerPreferencesField] = encoded
	replaced, err := json.Marshal(members)
	if err != nil {
		return nil, false
	}
	return replaced, true
}

func mustMarshalStrings(values []string) json.RawMessage {
	encoded, _ := json.Marshal(values)
	return encoded
}

// settleStalledHost writes the usage line of a call that failed before any
// content. It states no usage, so its counts are null and its cost unknown,
// never zero; the turn has not failed, so the turn's own failure state is
// left for the call that takes over.
func (r *OpenAIRouter) settleStalledHost(ctx *RequestContext, class, detail string) {
	latency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		latency = time.Since(ctx.StartTime)
	}
	// What the call stated before it stalled (Messages states its input
	// usage at message_start) is kept; what it did not stays unknown.
	usage := responseUsageMetrics{}
	if ctx.SemanticStreamState != nil {
		usage = partialStreamUsage(ctx.SemanticStreamState.usage)
	}
	record := r.newLLMUsageRecord(ctx, usage)
	latencyMillis := latency.Milliseconds()
	record.CompletionLatencyMS = &latencyMillis
	record.FailureClass = nonEmpty(class)
	record.FailureDetail = nonEmpty(detail)
	contentSent := false
	record.ContentSentBeforeFailure = &contentSent
	record.HostRetry = nonEmpty(hostRetryAttempt)
	r.priceUsageRecord(&record, ctx, ctx.RequestModel, usage)
	emitLLMUsageRecord(record)
	metrics.RecordModelWindowedRequest(
		ctx.RequestModel, latency.Seconds(), 0, 0, true, class == turnFailureTimeout,
	)
}

// partialStreamUsage is each count a call stated before it ended, alone: a
// Messages call states its input at message_start and its output only later,
// so the whole-usage reading would drop the input it did state.
func partialStreamUsage(usage llmprotocol.Usage) responseUsageMetrics {
	metrics := responseUsageMetrics{providerCost: usage.ProviderCost}
	if value, reported, ok := authoritativeTerminalTokenCount(usage.InputTotal); reported && ok {
		metrics.promptTokens, metrics.promptTokensReported = value, true
	}
	if value, reported, ok := authoritativeTerminalTokenCount(usage.OutputTotal); reported && ok {
		metrics.completionTokens, metrics.completionTokensReported = value, true
	}
	if value, reported, ok := authoritativeTerminalTokenCount(usage.InputCacheRead); reported && ok {
		metrics.cachedPromptTokens, metrics.cachedPromptTokensReported = value, true
	}
	if value, reported, ok := authoritativeTerminalTokenCount(usage.InputCacheWrite); reported && ok {
		metrics.cacheWriteTokens, metrics.cacheWriteTokensReported = value, true
	}
	return metrics
}

// retryAfterStreamFailure moves a turn whose call failed in-band (an SSE
// error event) before any content to the arm's next host. The failure event
// counts as content for the silence limits; here it is no content the client
// needs, since a host is left to serve the turn.
func (r *OpenAIRouter) retryAfterStreamFailure(ctx *RequestContext) bool {
	state := ctx.SemanticStreamState
	if state == nil || state.upstreamFailed == nil && state.failed == nil || state.carriesContent() {
		return false
	}
	seen, first := ctx.StreamContentSeen, ctx.StreamFirstContentAt
	ctx.StreamContentSeen, ctx.StreamFirstContentAt = false, time.Time{}
	if r.retryOnNextHost(ctx, turnFailureUpstreamError, "stream_error") {
		return true
	}
	ctx.StreamContentSeen, ctx.StreamFirstContentAt = seen, first
	return false
}

// carriesContent reports that the stream delivered model output: text,
// reasoning, a refusal or a tool call.
func (state *semanticResponseStreamState) carriesContent() bool {
	for _, item := range state.items {
		// A tool call's bare start carries no output yet.
		calledTool := item.toolCall != nil && (item.toolCall.Arguments != "" || item.completed)
		if item.text != "" || item.reasoning != "" || item.refusal != "" || calledTool || item.carried != nil {
			return true
		}
	}
	return false
}

// resetStreamForHostRetry forgets what the stalled call's stream said: the
// re-send's stream starts afresh, and the silence clock with it.
func resetStreamForHostRetry(ctx *RequestContext, body []byte, dispatch *providerDispatch) {
	ctx.ProtocolResponseStream = nil
	ctx.PublicStreamFilter = nil
	ctx.SemanticStreamState = nil
	// A codec error the stalled call raised is that call's, not the re-send's.
	ctx.StreamingAborted = false
	ctx.SemanticStreamErr = nil
	ctx.UpstreamBodyCut = nil
	ctx.StreamContentSince = time.Now()
	// The re-send is the cell's own single call.
	ctx.UpstreamAttempts = 1
	ctx.DispatchedOutbound = readDispatchedOutboundParams(body, dispatch)
	recordDispatchedProviderPin(ctx, body)
}

// stop abandons the re-send in flight, if any.
func (plan *hostRetryPlan) stop() {
	if plan != nil && plan.splice != nil {
		plan.splice.cancel()
	}
}

// hostRetryChunks is the channel the loop reads the re-send from, nil (never
// ready) when none is in flight.
func hostRetryChunks(ctx *RequestContext) <-chan hostRetryChunk {
	if ctx == nil || !ctx.HostRetry.started() || ctx.StreamingComplete {
		return nil
	}
	return ctx.HostRetry.splice.chunks
}

// startHostRetrySplice makes the call and streams what it says to the loop.
// The call ends with the exchange, or when the loop abandons it.
func startHostRetrySplice(ctx *RequestContext, call *flexTwinCall) *hostRetrySplice {
	parent := ctx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	callCtx, cancel := context.WithCancel(parent)
	splice := &hostRetrySplice{chunks: make(chan hostRetryChunk), cancel: cancel}
	deliver := func(chunk hostRetryChunk) bool {
		select {
		case splice.chunks <- chunk:
			return true
		case <-callCtx.Done():
			return false
		}
	}
	goSafely("extproc_stream_host_retry", func() {
		request, err := http.NewRequestWithContext(callCtx, http.MethodPost, call.url, bytes.NewReader(call.body))
		if err != nil {
			deliver(hostRetryChunk{err: err})
			return
		}
		request.Header = call.headers.Clone()
		response, err := flexResendHTTPClient.Do(request)
		if err != nil {
			deliver(hostRetryChunk{err: err})
			return
		}
		defer response.Body.Close()
		if !deliver(hostRetryChunk{
			headers: true, status: response.StatusCode, contentType: response.Header.Get("content-type"),
		}) {
			return
		}
		buffer := make([]byte, hostRetryReadSize)
		for {
			n, err := response.Body.Read(buffer)
			if n > 0 && !deliver(hostRetryChunk{body: bytes.Clone(buffer[:n])}) {
				return
			}
			if err == io.EOF {
				deliver(hostRetryChunk{end: true})
				return
			}
			if err != nil {
				deliver(hostRetryChunk{err: err})
				return
			}
		}
	})
	return splice
}

// serveHostRetryChunk answers what the re-sent call said: its stream goes to
// the client through the ordinary streaming path; a call that fails before
// any content moves the turn on to the next host, or ends it.
func (r *OpenAIRouter) serveHostRetryChunk(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
	chunk hostRetryChunk,
) error {
	plan := ctx.HostRetry
	var response *ext_proc.ProcessingResponse
	switch {
	case chunk.headers:
		if chunk.status >= 200 && chunk.status < 300 &&
			strings.Contains(strings.ToLower(chunk.contentType), "text/event-stream") {
			plan.splice.answered = true
			return nil
		}
		class := upstreamFailureClass(chunk.status, nil)
		if chunk.status >= 200 && chunk.status < 300 {
			// A 2xx that is not a stream is no use to a streamed response.
			class = turnFailureUpstreamError
		}
		if r.failHost(ctx, class, "upstream_status_"+strconv.Itoa(chunk.status)) {
			return nil
		}
		response = r.handleSemanticStreamingResponseBody(nil, false, ctx)
	case len(chunk.body) > 0:
		response = r.handleSemanticStreamingResponseBody(chunk.body, false, ctx)
	case !ctx.StreamContentSeen:
		// It ended, or broke, before any content.
		detail := "ended_before_content"
		if chunk.err != nil {
			detail = "transport_error"
		}
		if r.failHost(ctx, turnFailureUpstreamError, detail) {
			return nil
		}
		response = r.handleSemanticStreamingResponseBody(nil, false, ctx)
	case chunk.err != nil:
		ctx.UpstreamBodyCut = flexTwinCutError(chunk.err)
		response = r.handleSemanticStreamingResponseBody(nil, true, ctx)
	default:
		response = r.handleSemanticStreamingResponseBody(nil, true, ctx)
	}
	return r.sendSplicedReply(stream, ctx, response)
}

// sendSplicedReply sends a reply of a re-sent turn. A reply that ends the
// turn while the original call's trailers wait ends the body there, and the
// trailers' own reply completes the exchange, as processResponseTrailers
// would have.
func (r *OpenAIRouter) sendSplicedReply(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
) error {
	plan := ctx.HostRetry
	if ctx.StreamingComplete && plan.trailersPending {
		plan.trailersPending = false
		if err := sendResponse(stream, endedByTrailers(response), "response body"); err != nil {
			return err
		}
		return sendResponseTrailersReply(stream, ctx)
	}
	if err := sendResponse(stream, response, "response body"); err != nil {
		return err
	}
	runPendingSelectionCompletion(ctx)
	return nil
}

// droppedHostChunk answers a body chunk of the call a re-send replaced: none
// of it reaches the client.
func droppedHostChunk(ctx *RequestContext) *ext_proc.ProcessingResponse {
	return buildResponseBodyContinueResponse(responseStreamBodyMutation(ctx, nil, false), nil)
}

// addHostRetryFields writes which call of a re-sent turn a usage line is and
// the host it is attributed to.
func addHostRetryFields(record *llmUsageRecord, ctx *RequestContext) {
	plan := ctx.HostRetry
	if plan == nil || len(plan.attempts) < 2 && plan.attempts[0].host == "" {
		return
	}
	index := len(plan.attempts)
	current := plan.attempts[index-1]
	host, source := current.host, current.source
	if host == "" {
		host, source = plan.currentHost(ctx)
	}
	record.HostAttempt = &index
	record.UpstreamHost = nonEmpty(host)
	record.UpstreamHostSource = nonEmpty(source)
	if !ctx.StartTime.IsZero() {
		startedMS := current.started.Sub(ctx.StartTime).Milliseconds()
		record.HostAttemptStartedMS = &startedMS
	}
	tried := append(plan.tried(), host)
	record.ProviderEndpointsTried = tried
	if index > 1 {
		record.HostRetry = nonEmpty(hostRetryServed)
	}
}
