//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Every failure class, from what the provider answered. The last rows are
// controls: an ordinary bad request and an auth failure are upstream_error,
// not a class they only resemble.
func TestUpstreamFailureClass(t *testing.T) {
	failure := func(code, message string) *llmprotocol.ProtocolError {
		return &llmprotocol.ProtocolError{Code: code, Message: message}
	}
	cases := []struct {
		status int
		err    *llmprotocol.ProtocolError
		want   string
	}{
		{429, failure("rate_limit_error", "Provider returned error"), turnFailureRateLimited},
		{504, failure("", "upstream timed out"), turnFailureTimeout},
		{408, nil, turnFailureTimeout},
		{404, failure("not_found_error", "No endpoints found that satisfy the max price for this request"), turnFailureNoEndpoint},
		{400, failure("invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"), turnFailureContextOverflow},
		{400, failure("", "This model's maximum context length is 131072 tokens"), turnFailureContextOverflow},
		{502, failure("api_error", "bad gateway"), turnFailureUpstream5xx},
		{400, failure("data_inspection_failed", "Input data may contain inappropriate content."), turnFailureRefusal},
		{400, failure("invalid_request_error", "temperature must be at most 2"), turnFailureUpstreamError},
		{401, failure("authentication_error", "invalid key"), turnFailureUpstreamError},
	}
	for _, test := range cases {
		if got := upstreamFailureClass(test.status, test.err); got != test.want {
			t.Errorf("status %d %+v: class %q, want %q", test.status, test.err, got, test.want)
		}
	}
}

// A provider error writes a usage line, which it never did: the failure
// class, unknown usage with null counts, and nothing sent before it.
func TestProviderErrorWritesAClassedUsageLine(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "req-429", RequestModel: "kimi-k3", UpstreamStatusCode: 429,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	router.handleUpstreamTransportError([]byte(`{"error":{"message":"Provider returned error","code":429}}`), ctx)
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] != turnFailureRateLimited || usage["usage_source"] != usageSourceUnknown ||
		usage["prompt_tokens"] != nil || usage["pricing"] != usagePricingNoUsage || usage["content_sent_before_failure"] != false {
		t.Fatalf("provider error usage line = %#v", usage)
	}
	if failed := findLogEvent(t, logs, "turn_failed"); failed["failure_class"] != turnFailureRateLimited || failed["request_id"] != "req-429" {
		t.Fatalf("turn_failed = %#v", failed)
	}
}

// A buffered refusal is classed on its usage line; the same reply answered
// normally carries no class.
func TestRefusalIsClassedOnItsUsageLine(t *testing.T) {
	for _, body := range []string{arcRefusedCompletion, arcCacheTestCompletion} {
		refused := body == arcRefusedCompletion
		logs := captureLogs(t)
		_, router, decision := statusCacheRouter()
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-refusal-class", RequestModel: "test", RequestQuery: "hello",
			SemanticRequest: testNeutralRequest("test", "hello"),
			SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: context.Background(), UpstreamStatusCode: 200,
		}, decision)
		router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
		usage := findLogEvent(t, logs, "llm_usage")
		if refused && (usage["failure_class"] != turnFailureRefusal || usage["content_sent_before_failure"] != false) {
			t.Fatalf("refusal usage line = %#v", usage)
		}
		if !refused && (usage["failure_class"] != nil || usage["content_sent_before_failure"] != nil) {
			t.Fatalf("an answered turn was classed: %#v", usage)
		}
	}
}

// Streamed failures: a mid-stream provider error, a 2xx stream cut before
// its terminal event (with and without output already sent), and two
// controls, a stream the client cancelled and a stream that completed,
// neither of which is a provider failure.
func TestStreamFailureClasses(t *testing.T) {
	item := map[int]*semanticStreamItem{0: {}}
	cases := []struct {
		name      string
		state     *semanticResponseStreamState
		streamErr error
		want      string
		sent      bool
	}{
		{"provider error mid-stream", &semanticResponseStreamState{items: map[int]*semanticStreamItem{},
			failed: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorRateLimited, Message: "overloaded"}}, nil, turnFailureRateLimited, false},
		{"cut before output", &semanticResponseStreamState{items: map[int]*semanticStreamItem{}}, nil, turnFailureStreamCut, false},
		{"cut after output", &semanticResponseStreamState{items: item}, nil, turnFailureStreamCut, true},
		{"router deadline cut", &semanticResponseStreamState{items: item}, (&OpenAIRouter{}).truncatedStreamError(), turnFailureTimeout, true},
		{"malformed provider stream", &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
			llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_provider_response", "bad frame", nil), turnFailureUpstreamError, false},
		{"client cancelled", &semanticResponseStreamState{items: item}, errors.New("canceled"), "", false},
		{"completed", &semanticResponseStreamState{terminal: true, stop: llmprotocol.StopEndTurn, items: map[int]*semanticStreamItem{}}, nil, "", false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{RequestID: "req-stream", SemanticStreamState: test.state}
			response, responseErr := test.state.response()
			(&OpenAIRouter{}).classStreamFailure(ctx, response, responseErr, test.streamErr)
			if ctx.ResponseFailureClass != test.want {
				t.Fatalf("class %q, want %q", ctx.ResponseFailureClass, test.want)
			}
			if test.want != "" && (ctx.ContentSentBeforeFailure == nil || *ctx.ContentSentBeforeFailure != test.sent) {
				t.Fatalf("content sent = %v, want %v", ctx.ContentSentBeforeFailure, test.sent)
			}
		})
	}
}

// A cell-level failure keeps its public header class and refines it on the
// turn line where the package or catalog is the cause. Controls: a contended
// session and a plain backend outage map to their header classes unchanged.
func TestSelectionFailureTurnClasses(t *testing.T) {
	cases := map[string][2]string{
		"policy_service_package_not_loaded":  {turnFailurePackageNotLoaded, selectionFailureUnavailable},
		"policy_no_available_action":         {turnFailureNoAvailableAction, selectionFailureUnavailable},
		"policy_unbound_action":              {turnFailureNoAvailableAction, selectionFailureUnavailable},
		"not_ready":                          {selectionFailureNotReady, selectionFailureNotReady},
		"policy_service_session_busy":        {selectionFailureSessionBusy, selectionFailureSessionBusy},
		"policy_service_backend_unavailable": {selectionFailureUnavailable, selectionFailureUnavailable},
	}
	for internal, want := range cases {
		logs := captureLogs(t)
		ctx := &RequestContext{RequestID: "req-cell"}
		response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(&modelSelectionFailure{algorithm: "rayline_arc", class: internal}, ctx)
		if got := findLogEvent(t, logs, "turn_failed")["failure_class"]; got != want[0] {
			t.Errorf("%s: turn class %v, want %s", internal, got, want[0])
		}
		if header := immediateHeaderValue(response, selectionFailureHeader); header != want[1] {
			t.Errorf("%s: header %q, want %s", internal, header, want[1])
		}
	}
}

// The cell's commit gate (a reply whose turn can no longer be recorded)
// says the cell failed, in the header that survives the cell's scrub.
// Control: the same reply whose turn commits carries no failure class.
func TestCommitGateNamesTheCellFailure(t *testing.T) {
	for _, commitFails := range []bool{true, false} {
		logs := captureLogs(t)
		_, router, decision := statusCacheRouter()
		store, episode := newLedgerTestStore(t)
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-gate", RequestModel: "test", RequestQuery: "hello",
			SemanticRequest: testNeutralRequest("test", "hello"),
			SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: context.Background(), UpstreamStatusCode: 200,
			RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
		}, decision)
		ctx.RaylineARCTransaction.commitOnCompletion = true
		ctx.RaylineARCTransaction.markSelection(0, 10)
		if commitFails {
			ctx.RaylineARCTransaction.leaseLost.Store(true)
		}
		bindRaylineARCSelectionTransaction(ctx)
		response := router.handleNonStreamingResponseBody([]byte(arcCacheTestCompletion), ctx, time.Second)
		header := immediateHeaderValue(response, selectionFailureHeader)
		if (header == selectionFailureUnavailable) != commitFails || (commitFails && ctx.ResponseFailureClass != selectionFailureUnavailable) {
			t.Fatalf("commit fails=%v: header %q, class %q", commitFails, header, ctx.ResponseFailureClass)
		}
		// The usage line, written before the gate fails the reply, carries
		// the class too.
		if class := findLogEvent(t, logs, "llm_usage")["failure_class"]; (class == selectionFailureUnavailable) != commitFails {
			t.Fatalf("commit fails=%v: usage line failure_class %v", commitFails, class)
		}
		finalizeSelectionProcessTerminal(ctx)
	}
}

// A downgraded refusal's body names the turn's failure class beside its
// error, since its 200 status and its headers cannot. A body that is not a
// JSON object, and a turn with no class, travel unchanged.
func TestDowngradedRefusalBodyNamesTheFailureClass(t *testing.T) {
	marked := withFailureClass([]byte(`{"type":"error","error":{"type":"api_error","message":"busy"}}`), selectionFailureUnavailable)
	var body map[string]any
	if err := json.Unmarshal(marked, &body); err != nil || body["failure_class"] != selectionFailureUnavailable || body["error"] == nil {
		t.Fatalf("marked body = %s", marked)
	}
	for _, unchanged := range []struct{ body, class string }{
		{"event: error\ndata: {}\n\n", selectionFailureUnavailable},
		{`{"error":{"message":"x"}}`, ""},
	} {
		if got := string(withFailureClass([]byte(unchanged.body), unchanged.class)); got != unchanged.body {
			t.Fatalf("%q with class %q became %q", unchanged.body, unchanged.class, got)
		}
	}
}

// A lost lease found at the response headers is the same cell failure as at
// the body: the 503 carries the class header and the turn is classed.
// Control: a held lease lets the reply through unmarked.
func TestResponseHeaderGateNamesTheCellFailure(t *testing.T) {
	for _, leaseLost := range []bool{true, false} {
		logs := captureLogs(t)
		store, episode := newLedgerTestStore(t)
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		ctx := &RequestContext{
			RequestID: "req-header-gate", TraceContext: context.Background(), Headers: map[string]string{},
			RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
		}
		ctx.RaylineARCTransaction.commitOnCompletion = true
		ctx.RaylineARCTransaction.markSelection(0, 10)
		if leaseLost {
			ctx.RaylineARCTransaction.leaseLost.Store(true)
		}
		bindRaylineARCSelectionTransaction(ctx)
		response, err := (&OpenAIRouter{}).handleResponseHeaders(arcResponseHeaders("200"), ctx)
		if err != nil {
			t.Fatal(err)
		}
		failed := response.GetImmediateResponse() != nil
		if failed != leaseLost || (leaseLost && immediateHeaderValue(response, selectionFailureHeader) != selectionFailureUnavailable) {
			t.Fatalf("lease lost=%v: failed=%v header %q", leaseLost, failed, immediateHeaderValue(response, selectionFailureHeader))
		}
		if leaseLost && findLogEvent(t, logs, "turn_failed")["failure_class"] != selectionFailureUnavailable {
			t.Fatal("the header-gate failure was not classed")
		}
		finalizeSelectionProcessTerminal(ctx)
	}
}

// A response check that blocks the reply still leaves its usage line: the
// line moved after the commit gate, and every early return writes it first.
func TestBlockedReplyStillWritesItsUsageLine(t *testing.T) {
	logs := captureLogs(t)
	router, ctx := newResponseStageRouter(t, newJailbreakFailingServer(t), config.OnErrorBlock, "block")
	ctx.RequestID, ctx.RequestModel, ctx.UpstreamStatusCode = "req-blocked", "test", 200
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	response := router.handleNonStreamingResponseBody([]byte(arcCacheTestCompletion), ctx, time.Second)
	if response.GetImmediateResponse() == nil {
		t.Fatal("the plugin did not block")
	}
	if usage := findLogEvent(t, logs, "llm_usage"); usage["prompt_tokens"] == nil {
		t.Fatalf("the blocked reply's usage line = %#v", usage)
	}
}
