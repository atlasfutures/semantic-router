//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"io"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"go.opentelemetry.io/otel"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
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
		{"terminal with an unfinished item", unfinishedItemStream(), nil, turnFailureUpstreamError, true},
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
			RequestID: "req-header-gate", RequestModel: "test", TraceContext: context.Background(), Headers: map[string]string{},
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
		// The upstream answered and may bill: its usage line is written here.
		if leaseLost {
			if usage := findLogEvent(t, logs, "llm_usage"); usage["failure_class"] != selectionFailureUnavailable || usage["pricing"] != usagePricingNoUsage {
				t.Fatalf("header-gate usage line = %#v", usage)
			}
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
	if usage := findLogEvent(t, logs, "llm_usage"); usage["prompt_tokens"] == nil ||
		usage["failure_class"] != turnFailureResponseBlocked || usage["failure_detail"] != "response_jailbreak" {
		t.Fatalf("the blocked reply's usage line = %#v", usage)
	}
}

// A reply the Router could not use is a classed failure like the others: its
// code reaches the usage line, the turn_failed line and the counter. One that
// never decoded is classed too, on a line that states no counts.
func TestUnusableResponseIsAClassedFailure(t *testing.T) {
	for _, decoded := range []bool{true, false} {
		logs := captureLogs(t)
		ctx := &RequestContext{RequestID: "req-unusable", RequestModel: "test"}
		if decoded {
			ctx.UpstreamDecodedRemnant = &llmprotocol.Response{Model: "test", StopReason: llmprotocol.StopEndTurn}
		}
		usageRecordRouter().reportUnusableResponseUsage(ctx, time.Second,
			llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "empty_completion", "no output", nil))
		var failed int
		for _, entry := range logs.All() {
			if entry.ContextMap()["event"] == "turn_failed" {
				failed++
			}
		}
		if decoded {
			usage := findLogEvent(t, logs, "llm_usage")
			if failed != 1 || usage["failure_class"] != turnFailureUpstreamError || usage["failure_detail"] != "empty_completion" {
				t.Fatalf("decoded: turn_failed=%d usage=%#v", failed, usage)
			}
		}
		// An undecoded reply is classed too, and its line states no counts.
		if !decoded {
			usage := findLogEvent(t, logs, "llm_usage")
			if failed != 1 || usage["failure_class"] != turnFailureUpstreamError || usage["prompt_tokens"] != nil {
				t.Fatalf("undecoded: turn_failed=%d usage=%#v", failed, usage)
			}
		}
	}
}

// A provider error with no body ends at its headers, so it is classed and its
// usage settled there. Control: the same status with a body still to come is
// left to the body phase.
func TestBodylessProviderErrorIsSettledAtTheHeaders(t *testing.T) {
	for _, endsAtHeaders := range []bool{true, false} {
		logs := captureLogs(t)
		ctx := &RequestContext{RequestID: "req-bodyless", RequestModel: "kimi-k3", StartTime: time.Now(), Headers: map[string]string{}}
		headers := arcResponseHeaders("429")
		headers.ResponseHeaders.EndOfStream = endsAtHeaders
		if _, err := (&OpenAIRouter{}).handleResponseHeaders(headers, ctx); err != nil {
			t.Fatal(err)
		}
		settled := false
		for _, entry := range logs.All() {
			fields := entry.ContextMap()
			if fields["event"] == "llm_usage" && fields["failure_class"] == turnFailureRateLimited {
				settled = true
			}
		}
		if settled != endsAtHeaders {
			t.Fatalf("ends at headers=%v: settled=%v", endsAtHeaders, settled)
		}
	}
}

// A streamed policy turn whose lease was lost while it streamed is the
// cell's failure, and its usage line says so. Control: a turn that commits
// carries no class.
func TestStreamedLeaseLossIsClassedOnItsUsageLine(t *testing.T) {
	for _, commitFails := range []bool{true, false} {
		logs := captureLogs(t)
		_, router, decision := statusCacheRouter()
		store, episode := newLedgerTestStore(t)
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-stream-lease", RequestModel: "test", RequestQuery: "hello",
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
		stream := &semanticResponseStreamState{
			responseID: "chatcmpl-lease", model: "test", stop: llmprotocol.StopEndTurn,
			items: map[int]*semanticStreamItem{}, terminal: true,
			usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		}
		item := stream.item(0)
		item.text, item.completed = "hi", true
		ctx.SemanticStreamState = stream
		router.finalizeSemanticStreamingResponse(ctx, nil)
		runPendingSelectionCompletion(ctx)
		finalizeSelectionProcessTerminal(ctx)
		class := findLogEvent(t, logs, "llm_usage")["failure_class"]
		if (class == selectionFailureUnavailable) != commitFails || (!commitFails && class != nil) {
			t.Fatalf("commit fails=%v: usage line failure_class %v", commitFails, class)
		}
	}
}

// A reply a body guard cut (its deadline or its size) was never sent: it is
// classed, its usage settled as unknown, and its body names the class since
// its 200 headers are spent.
func TestBodyGuardCutIsClassedAndSettled(t *testing.T) {
	cases := map[string]string{"response_body_timeout": turnFailureTimeout, "response_body_too_large": turnFailureUpstreamError}
	for code, want := range cases {
		logs := captureLogs(t)
		ctx := &RequestContext{
			RequestID: "req-guard", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200,
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		}
		response := (&OpenAIRouter{}).responseBodyGuardResponse(ctx,
			llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, code, "cut", nil))
		usage := findLogEvent(t, logs, "llm_usage")
		if usage["failure_class"] != want || usage["failure_detail"] != code || usage["usage_source"] != usageSourceUnknown {
			t.Fatalf("%s: usage line = %#v", code, usage)
		}
		mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
		body := mutation.GetBody()
		if streamed := mutation.GetStreamedResponse(); streamed != nil {
			body = streamed.GetBody()
		}
		var decoded map[string]any
		if err := json.Unmarshal(body, &decoded); err != nil || decoded["failure_class"] != want {
			t.Fatalf("%s: guard body = %s", code, body)
		}
	}
}

// A 2xx that ends at its headers on an inference request is not a served
// reply: it is refused while the headers can be replaced, classed and
// settled. Controls: a bodyless 2xx outside inference passes, and so does an
// inference 2xx whose body is still to come.
func TestEmptySuccessIsRefusedAtTheHeaders(t *testing.T) {
	cases := []struct {
		name      string
		inference bool
		endsHere  bool
		refused   bool
	}{
		{"empty inference reply", true, true, true},
		{"bodyless non-inference reply", false, true, false},
		{"inference reply with a body to come", true, false, false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			logs := captureLogs(t)
			ctx := &RequestContext{RequestID: "req-empty", RequestModel: "test", VSRSelectedModel: "arm-x", StartTime: time.Now(), Headers: map[string]string{}}
			_, span := otel.Tracer("test").Start(context.Background(), "upstream")
			ctx.UpstreamSpan = span
			if test.inference {
				ctx.SemanticRequest = testNeutralRequest("test", "hello")
			}
			headers := arcResponseHeaders("200")
			headers.ResponseHeaders.EndOfStream = test.endsHere
			response, err := (&OpenAIRouter{}).handleResponseHeaders(headers, ctx)
			if err != nil {
				t.Fatal(err)
			}
			if refused := response.GetImmediateResponse() != nil; refused != test.refused {
				t.Fatalf("refused=%v", refused)
			}
			if test.refused {
				if usage := findLogEvent(t, logs, "llm_usage"); usage["failure_class"] != turnFailureUpstreamError || usage["failure_detail"] != "empty_response" {
					t.Fatalf("usage line = %#v", usage)
				}
				// It names the arm that failed, and its upstream span ends.
				if model := immediateHeaderValue(response, "x-vsr-selected-model"); model != "arm-x" {
					t.Fatalf("x-vsr-selected-model = %q", model)
				}
				if ctx.UpstreamSpan != nil {
					t.Fatal("the upstream span was left open")
				}
				// The exchange then ends: a streaming turn settled at its
				// headers is not finalized again, so it keeps its class and
				// one usage line.
				ctx.IsStreamingResponse = true
				_ = (&OpenAIRouter{}).handleProcessReceiveError(ctx, status.Error(codes.Canceled, "context canceled"))
				usageLines := 0
				for _, entry := range logs.All() {
					if entry.ContextMap()["event"] == "llm_usage" {
						usageLines++
					}
				}
				if ctx.ResponseFailureClass != turnFailureUpstreamError || usageLines != 1 {
					t.Fatalf("after the exchange ended: class %q, %d usage lines", ctx.ResponseFailureClass, usageLines)
				}
			}
		})
	}
}

// A lease lost at the dispatch gate is a counted cell failure, like the
// header- and body-phase gates. Control: a held lease dispatches.
func TestDispatchGateLeaseLossIsATurnFailure(t *testing.T) {
	for _, leaseLost := range []bool{true, false} {
		logs := captureLogs(t)
		store, episode := newLedgerTestStore(t)
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		ctx := &RequestContext{
			RequestID: "req-dispatch", TraceContext: context.Background(), Headers: map[string]string{},
			RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
		}
		ctx.RaylineARCTransaction.markSelection(0, 10)
		if leaseLost {
			ctx.RaylineARCTransaction.leaseLost.Store(true)
		}
		bindRaylineARCSelectionTransaction(ctx)
		blocked := (&OpenAIRouter{}).selectionDispatchGateResponse(ctx) != nil
		counted := false
		for _, entry := range logs.All() {
			if fields := entry.ContextMap(); fields["event"] == "turn_failed" && fields["failure_class"] == selectionFailureUnavailable {
				counted = true
			}
		}
		if blocked != leaseLost || counted != leaseLost {
			t.Fatalf("lease lost=%v: blocked=%v counted=%v", leaseLost, blocked, counted)
		}
		finalizeSelectionProcessTerminal(ctx)
	}
}

// A refused reply that a response check then blocks is one failed turn,
// classed by its final outcome.
func TestBlockedRefusalIsOneFailure(t *testing.T) {
	logs := captureLogs(t)
	router, ctx := newResponseStageRouter(t, newJailbreakFailingServer(t), config.OnErrorBlock, "block")
	ctx.RequestID, ctx.RequestModel, ctx.UpstreamStatusCode = "req-blocked-refusal", "test", 200
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	router.handleNonStreamingResponseBody([]byte(arcRefusedCompletion), ctx, time.Second)
	var classes []any
	for _, entry := range logs.All() {
		if fields := entry.ContextMap(); fields["event"] == "turn_failed" {
			classes = append(classes, fields["failure_class"])
		}
	}
	if len(classes) != 1 || classes[0] != turnFailureResponseBlocked {
		t.Fatalf("turn_failed classes = %v, want one response_blocked", classes)
	}
}

// Only a backend's own failure counts against it in the load-balancing
// window. Controls: a refusal, a blocked reply and the cell's failures do not.
func TestBackendErrorClassesForTheWindow(t *testing.T) {
	for class, want := range map[string]bool{
		turnFailureRateLimited: true, turnFailureUpstream5xx: true, turnFailureTimeout: true,
		turnFailureNoEndpoint: true, turnFailureUpstreamError: true, turnFailureStreamCut: true,
		turnFailureRefusal: false, turnFailureResponseBlocked: false, selectionFailureUnavailable: false, "": false,
	} {
		if got := turnFailureIsBackendError(class); got != want {
			t.Errorf("%q: backend error = %v, want %v", class, got, want)
		}
	}
}

// Non-text output beside a refusal reached the client. Control: empty text
// beside a refusal is nothing.
func TestContentBesideARefusal(t *testing.T) {
	refusal := llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: "no"}
	withImage := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Content: []llmprotocol.Content{refusal, {Kind: llmprotocol.ContentImage}}}}}
	withEmptyText := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Content: []llmprotocol.Content{refusal, {Kind: llmprotocol.ContentText}}}}}
	if !contentBeforeRefusal(withImage) || contentBeforeRefusal(withEmptyText) {
		t.Fatalf("image beside refusal=%v, empty text beside refusal=%v", contentBeforeRefusal(withImage), contentBeforeRefusal(withEmptyText))
	}
}

// unfinishedItemStream ended properly, but its one item never completed.
func unfinishedItemStream() *semanticResponseStreamState {
	state := &semanticResponseStreamState{terminal: true, stop: llmprotocol.StopEndTurn, items: map[int]*semanticStreamItem{}}
	state.item(0).text = "hi"
	return state
}

// The provider hop's attempt count rides the usage line: a call Envoy retried
// says so and marks its cost incomplete, the earlier attempts' charge being
// unknown. Controls: a single attempt is complete; a hop that sends no count
// (or a malformed one) leaves both null.
func TestUsageLineCarriesTheUpstreamAttempts(t *testing.T) {
	for _, attempts := range []string{"2", "1", "", "bogus"} {
		logs := captureLogs(t)
		ctx := &RequestContext{RequestID: "req-attempts", RequestModel: "kimi-k3", StartTime: time.Now(), Headers: map[string]string{}}
		headers := arcResponseHeaders("429")
		if attempts != "" {
			headers.ResponseHeaders.Headers.Headers = append(headers.ResponseHeaders.Headers.Headers,
				&core.HeaderValue{Key: "x-vsr-upstream-attempts", RawValue: []byte(attempts)})
		}
		headers.ResponseHeaders.EndOfStream = true
		if _, err := (&OpenAIRouter{}).handleResponseHeaders(headers, ctx); err != nil {
			t.Fatal(err)
		}
		usage := findLogEvent(t, logs, "llm_usage")
		want := map[string][2]interface{}{
			"2": {int64(2), false}, "1": {int64(1), true}, "": {nil, nil}, "bogus": {nil, nil},
		}[attempts]
		if fmt.Sprint(usage["upstream_attempts"]) != fmt.Sprint(want[0]) || fmt.Sprint(usage["cost_complete"]) != fmt.Sprint(want[1]) {
			t.Fatalf("header %q: upstream_attempts = %#v, cost_complete = %#v, want %v",
				attempts, usage["upstream_attempts"], usage["cost_complete"], want)
		}
	}
}

// A stream the codec cut leaves the codec's code and message on the usage
// line, so a cut is diagnosable from the logs rather than reading as a bare
// class; a provider's in-band error leaves only its code, never the
// provider's free text.
func TestStreamFailureDetailNamesTheError(t *testing.T) {
	router := &OpenAIRouter{}
	cut := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_event_after_terminal",
		"Anthropic stream emitted content_block_start after message_stop", nil)
	provider := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "overloaded_error",
		"Overloaded\n"+strings.Repeat("x", 400), nil)
	for name, tc := range map[string]struct {
		streamErr error
		failed    *llmprotocol.ProtocolError
		want      string
	}{
		"codec cut": {streamErr: cut, want: "stream_event_after_terminal"},
		// A provider's own text is never logged: a documented code is kept,
		// anything else is reported by the Router's category.
		"provider in-band": {failed: provider, want: "provider:overloaded_error"},
		"provider unknown code": {failed: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sk-or-v1-secretish", "x", nil),
			want: "provider:" + string(llmprotocol.ErrorUpstreamUnavailable)},
	} {
		logs := captureLogs(t)
		ctx := &RequestContext{
			RequestID: "req-stream-detail", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200,
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1, IsStreamingResponse: true,
			SemanticStreamState: &semanticResponseStreamState{items: map[int]*semanticStreamItem{0: {}}, failed: tc.failed},
		}
		router.finalizeSemanticStreamingResponse(ctx, tc.streamErr)
		if got := findLogEvent(t, logs, "llm_usage")["failure_detail"]; got != tc.want {
			t.Fatalf("%s: failure_detail = %q, want %q", name, got, tc.want)
		}
	}
}

// End to end through the Router's own stream: a provider stream that stops
// at EOF without its terminal event is finalized by the real codec, which
// synthesizes the failure, and the usage line names it.
func TestAnIncompleteStreamNamesItsFailure(t *testing.T) {
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-eof", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	buffers := &semanticStreamBuffers{}
	buffers.push([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"+
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"+
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half an answer\"}}\n\n"), ctx)
	buffers.finalize(ctx)
	(&OpenAIRouter{}).finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	if got := findLogEvent(t, logs, "llm_usage")["failure_detail"]; got != "stream_incomplete" {
		t.Fatalf("failure_detail = %q, want stream_incomplete", got)
	}
}

// A provider that fails while a tool call's arguments are still streaming
// leaves them a truncated object. Through the Router's own buffers, every
// client format is told the provider's failure, never
// invalid_stream_tool_arguments, and the usage and turn_failed lines keep
// the provider's class and code.
func TestAProviderFailureDuringACutToolCallKeepsItsClass(t *testing.T) {
	start := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"bash\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\": \"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls -la /tm\"}}\n\n"
	stop := "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	overloaded := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	for name, frames := range map[string]string{
		"block open":    start + overloaded,
		"block stopped": start + stop + overloaded,
	} {
		for _, client := range []llmprotocol.WireFormat{
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		} {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				logs := captureLogs(t)
				stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, client,
					llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				ctx := &RequestContext{
					RequestID: "req-cut-failure", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
					SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1,
					ProtocolResponseStream: stream,
					SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
				}
				buffers := &semanticStreamBuffers{}
				buffers.push([]byte(frames), ctx)
				buffers.finalize(ctx)
				(&OpenAIRouter{}).finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
				body := string(buffers.translated)
				if buffers.streamErr != nil || strings.Contains(body, "invalid_stream_tool_arguments") {
					t.Fatalf("stream error %v; the client was told the symptom:\n%s", buffers.streamErr, body)
				}
				if !strings.Contains(body, "overloaded_error") {
					t.Fatalf("the client was not told the provider's failure:\n%s", body)
				}
				for _, event := range []string{"llm_usage", "turn_failed"} {
					fields := findLogEvent(t, logs, event)
					if fields["failure_class"] != turnFailureUpstream5xx || fields["failure_detail"] != "provider:overloaded_error" {
						t.Fatalf("%s = class %v detail %v, want %s provider:overloaded_error",
							event, fields["failure_class"], fields["failure_detail"], turnFailureUpstream5xx)
					}
				}
			})
		}
	}
}

// An upstream that stops while a tool call's arguments are streaming, with
// no message_delta or message_stop, is an ordinary cut stream: the client
// is told stream_incomplete and the turn is classed as any stream cut with
// the call still open is, never as a tool-argument error or a provider
// outage. The end arrives with the last frames and, as Envoy also sends it,
// as an empty end-of-stream chunk after them.
func TestAToolCallCutByAStreamWithNoTerminalIsAnIncompleteStream(t *testing.T) {
	open := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"edit\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"/app/x.R\\\",\\\"edits\\\":[{\\\"oldText\\\":\\\"f <- function(x) {\"}}\n\n"
	held := open + "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	run := func(t *testing.T, client llmprotocol.WireFormat, chunks []string) (string, map[string]interface{}) {
		t.Helper()
		logs := captureLogs(t)
		stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, client,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatal(err)
		}
		ctx := &RequestContext{
			RequestID: "req-cut-no-terminal", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
			SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1,
			ProtocolResponseStream: stream,
			SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
		}
		router := &OpenAIRouter{}
		var body strings.Builder
		for index, chunk := range chunks {
			response := router.handleSemanticStreamingResponseBody([]byte(chunk), index == len(chunks)-1, ctx)
			if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
				body.Write(mutation.GetBody())
			} else {
				body.WriteString(chunk)
			}
		}
		return body.String(), findLogEvent(t, logs, "llm_usage")
	}
	for name, split := range map[string]func(string) []string{
		"end with the frames":        func(frames string) []string { return []string{frames} },
		"empty end in a later chunk": func(frames string) []string { return []string{frames, ""} },
	} {
		for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				body, usage := run(t, client, split(held))
				if strings.Contains(body, "invalid_stream_tool_arguments") || !strings.Contains(body, "upstream stream ended before completion") {
					t.Fatalf("the client was not told the stream was incomplete:\n%s", body)
				}
				if client == llmprotocol.OpenAIResponsesV1 && !strings.Contains(body, `"code":"stream_incomplete"`) {
					t.Fatalf("the Responses client was not given stream_incomplete:\n%s", body)
				}
				_, uncut := run(t, client, split(open))
				if usage["failure_detail"] != "stream_incomplete" || usage["failure_class"] != uncut["failure_class"] ||
					usage["content_sent_before_failure"] != true {
					t.Fatalf("llm_usage = class %v detail %v sent %v, want %v stream_incomplete true (a cut with the call open)",
						usage["failure_class"], usage["failure_detail"], usage["content_sent_before_failure"], uncut["failure_class"])
				}
			})
		}
	}
}

// A tool call the provider stopped mid-arguments is classed by why it
// stopped. A refusal is a refusal (declined, with its stop_details), a stop
// at the context window is a served length stop, and only a stop that says
// the reply finished leaves the arguments malformed.
func TestAToolCallCutByAStopIsClassedByTheStop(t *testing.T) {
	held := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"edit\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"/app/x.R\\\",\\\"edits\\\":[{\\\"oldText\\\":\\\"f <- function(x) {\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	stop := func(reason, details string) string {
		return "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"" + reason + "\",\"stop_sequence\":null" + details + "},\"usage\":{\"output_tokens\":9}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	}
	for name, test := range map[string]struct {
		terminal, class, detail, body string
	}{
		"refusal": {
			terminal: stop("refusal", `,"stop_details":{"type":"refusal","category":"cyber","explanation":"Flagged."}`),
			class:    turnFailureRefusal, body: `"reason":"content_filter"`,
		},
		"context window": {terminal: stop("model_context_window_exceeded", ""), body: `"reason":"max_output_tokens"`},
		"end_turn": {
			terminal: stop("end_turn", ""), class: turnFailureUpstreamError, detail: "invalid_stream_tool_arguments",
			body: "invalid_stream_tool_arguments",
		},
	} {
		// The terminal arrives with the end of the stream, or in its own
		// chunk followed by an empty end-of-stream chunk, as Envoy sends it.
		for shape, chunks := range map[string][]string{
			"end with the terminal":      {held, test.terminal},
			"empty end in a later chunk": {held, test.terminal, ""},
		} {
			t.Run(name+"/"+shape, func(t *testing.T) {
				logs := captureLogs(t)
				stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
					llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				ctx := &RequestContext{
					RequestID: "req-cut-stop", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
					SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
					ProtocolResponseStream: stream,
					SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
				}
				router := &OpenAIRouter{}
				var body strings.Builder
				for index, chunk := range chunks {
					response := router.handleSemanticStreamingResponseBody([]byte(chunk), index == len(chunks)-1, ctx)
					if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
						body.Write(mutation.GetBody())
					}
				}
				if !strings.Contains(body.String(), test.body) {
					t.Fatalf("client body lacks %s:\n%s", test.body, body.String())
				}
				if test.class != turnFailureUpstreamError && strings.Contains(body.String(), "invalid_stream_tool_arguments") {
					t.Fatalf("the client was told the arguments were malformed:\n%s", body.String())
				}
				usage := findLogEvent(t, logs, "llm_usage")
				if class, _ := usage["failure_class"].(string); class != test.class {
					t.Fatalf("llm_usage failure_class = %q, want %q", class, test.class)
				}
				if test.detail != "" && usage["failure_detail"] != test.detail {
					t.Fatalf("llm_usage failure_detail = %v, want %s", usage["failure_detail"], test.detail)
				}
				failed := map[string]interface{}(nil)
				for _, entry := range logs.All() {
					if fields := entry.ContextMap(); fields["event"] == "turn_failed" {
						failed = fields
					}
				}
				switch {
				case test.class == "" && failed != nil:
					t.Fatalf("a served length stop was recorded as failed: %v", failed)
				case test.class != "" && (failed == nil || failed["failure_class"] != test.class):
					t.Fatalf("turn_failed = %v, want class %s", failed, test.class)
				case test.class == turnFailureRefusal && (failed["stop_details_category"] != "cyber" || usage["stop_details_category"] != "cyber"):
					t.Fatalf("refusal telemetry lost its category: turn_failed %v, llm_usage %v", failed, usage)
				}
			})
		}
	}
}

// A provider stream the Router's codec rejects mid-stream is the Router's
// rejection, not a provider outage, however the stream then ends. The codec
// replays its stored failure as the stream's failed event when a later
// chunk ends the stream; that event must not be read as the provider's.
func TestACodecRejectionInAnEarlierChunkIsNotAProviderFailure(t *testing.T) {
	rejected := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"bash\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"[1, 2]\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		// The next event releases the held tool block, whose whole non-object
		// arguments the codec refuses here.
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
	rest := "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	for name, chunks := range map[string][]string{
		// Envoy ends a stream with an empty end-of-stream body.
		"empty end in a later chunk":       {rejected, ""},
		"end with frames in a later chunk": {rejected, rest},
		"end in the same chunk":            {rejected + rest},
	} {
		for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				logs := captureLogs(t)
				stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, client,
					llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				ctx := &RequestContext{
					RequestID: "req-codec-rejection", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
					SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1,
					ProtocolResponseStream: stream,
					SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
				}
				router := &OpenAIRouter{}
				for index, chunk := range chunks {
					router.handleSemanticStreamingResponseBody([]byte(chunk), index == len(chunks)-1, ctx)
				}
				for _, event := range []string{"llm_usage", "turn_failed"} {
					fields := findLogEvent(t, logs, event)
					if fields["failure_class"] != turnFailureUpstreamError || fields["failure_detail"] != "invalid_stream_tool_arguments" ||
						fields["content_sent_before_failure"] != true {
						t.Fatalf("%s = class %v detail %v sent %v, want %s invalid_stream_tool_arguments true", event,
							fields["failure_class"], fields["failure_detail"], fields["content_sent_before_failure"], turnFailureUpstreamError)
					}
				}
			})
		}
	}
}

// A stream the client or the proxy ended first is classed client_ended, with
// the usage the provider stated and whether content had reached the client.
// It is neither an arm nor a cell failure: it counts against no backend and
// excludes no route.
func TestAStreamTheClientEndedIsClassedClientEnded(t *testing.T) {
	start := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":23,\"output_tokens\":8}}}\n\n"
	content := "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half\"}}\n\n"
	for name, test := range map[string]struct {
		frames string
		ended  error
		sent   bool
		detail string
	}{
		"after content":        {frames: start + content, ended: status.Error(codes.Canceled, "context canceled"), sent: true, detail: "grpc:canceled"},
		"before content":       {frames: start, ended: io.EOF, sent: false, detail: "grpc:eof"},
		"raw context cancel":   {frames: start, ended: context.Canceled, sent: false, detail: "grpc:canceled"},
		"raw context deadline": {frames: start, ended: fmt.Errorf("recv: %w", context.DeadlineExceeded), sent: false, detail: "grpc:deadlineexceeded"},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatal(err)
			}
			ctx := &RequestContext{
				RequestID: "req-ended", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
				SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
				ProtocolResponseStream: stream,
				SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
			}
			buffers := &semanticStreamBuffers{}
			buffers.push([]byte(test.frames), ctx)
			_ = (&OpenAIRouter{}).handleProcessReceiveError(ctx, test.ended)
			usage := findLogEvent(t, logs, "llm_usage")
			if usage["failure_class"] != turnFailureClientEnded || usage["failure_detail"] != test.detail ||
				usage["content_sent_before_failure"] != test.sent || usage["truncated"] != true {
				t.Fatalf("llm_usage = class %v detail %v sent %v truncated %v, want client_ended %s %v true",
					usage["failure_class"], usage["failure_detail"], usage["content_sent_before_failure"], usage["truncated"], test.detail, test.sent)
			}
			if fmt.Sprint(usage["prompt_tokens"]) != "23" {
				t.Fatalf("prompt_tokens = %v, want the provider's 23", usage["prompt_tokens"])
			}
			if failed := findLogEvent(t, logs, "turn_failed"); failed["failure_class"] != turnFailureClientEnded {
				t.Fatalf("turn_failed class = %v", failed["failure_class"])
			}
		})
	}
	if turnFailureIsBackendError(turnFailureClientEnded) || cellExclusionClasses[turnFailureClientEnded] {
		t.Fatal("client_ended counts against the backend or excludes its route")
	}
}

// A provider error the stream already carried keeps its class when the
// exchange then ends: the disconnect does not hide the provider's failure.
func TestAProviderFailureSeenBeforeTheClientEndedKeepsItsClass(t *testing.T) {
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-failed-then-ended", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	buffers := &semanticStreamBuffers{}
	buffers.push([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n"+
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"), ctx)
	if ctx.SemanticStreamState.failed == nil {
		t.Fatal("the provider's in-band error was not observed")
	}
	_ = (&OpenAIRouter{}).handleProcessReceiveError(ctx, status.Error(codes.Canceled, "context canceled"))
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] == turnFailureClientEnded || usage["failure_detail"] != "provider:overloaded_error" {
		t.Fatalf("llm_usage class %v detail %v, want the provider's failure kept", usage["failure_class"], usage["failure_detail"])
	}
}

// A terminal reply the Router cannot rebuild stays upstream_error when the
// exchange then ends: the disconnect does not hide the unusable reply. (A
// provider's successful terminal is released to the client only at the end
// of the stream, so a stream whose exchange ended first never completed for
// the client and is client_ended.)
func TestAnUnusableTerminalSeenBeforeTheClientEndedKeepsItsClass(t *testing.T) {
	_ = captureLogs(t)
	ctx := &RequestContext{RequestID: "req-unusable-then-ended", StreamEndedByReceiveError: true,
		SemanticStreamState: &semanticResponseStreamState{terminal: true, items: map[int]*semanticStreamItem{0: {}}}}
	(&OpenAIRouter{}).classStreamFailure(ctx, nil, errors.New("item never completed"), status.Error(codes.Canceled, "context canceled"))
	if ctx.ResponseFailureClass != turnFailureUpstreamError || ctx.ResponseFailureDetail != "stream_reconstruction_failed" {
		t.Fatalf("class %q detail %q, want upstream_error stream_reconstruction_failed", ctx.ResponseFailureClass, ctx.ResponseFailureDetail)
	}
}

// A failure the turn already carries is not overwritten when the exchange
// then ends: a codec error an earlier chunk raised, or a class recorded at
// the response headers.
func TestTheClientEndingDoesNotReplaceAnEarlierFailure(t *testing.T) {
	t.Run("codec error in an earlier chunk", func(t *testing.T) {
		logs := captureLogs(t)
		stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatal(err)
		}
		ctx := &RequestContext{
			RequestID: "req-codec-then-ended", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
			ProtocolResponseStream: stream,
			SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
		}
		first := &semanticStreamBuffers{}
		first.push([]byte("event: message_start\ndata: {not json}\n\n"), ctx)
		if first.streamErr == nil {
			t.Fatal("the malformed chunk raised no codec error")
		}
		_ = (&OpenAIRouter{}).handleProcessReceiveError(ctx, status.Error(codes.Canceled, "context canceled"))
		if got := findLogEvent(t, logs, "llm_usage")["failure_class"]; got != turnFailureUpstreamError {
			t.Fatalf("failure_class = %v, want the codec error's upstream_error", got)
		}
	})
	t.Run("class recorded at the headers", func(t *testing.T) {
		_ = captureLogs(t)
		ctx := &RequestContext{RequestID: "req-headers-then-ended", StreamEndedByReceiveError: true, ResponseFailureClass: turnFailureUpstreamError}
		(&OpenAIRouter{}).classStreamFailure(ctx, nil, nil, status.Error(codes.Canceled, "context canceled"))
		if ctx.ResponseFailureClass != turnFailureUpstreamError {
			t.Fatalf("class = %q, want the header's upstream_error kept", ctx.ResponseFailureClass)
		}
	})
}

// The exchange can also end while the Router is sending a chunk: a canceled
// Send finalizes the stream as a receive error does.
func TestAStreamEndedAtASendIsClassedClientEnded(t *testing.T) {
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-send-canceled", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	chunk := []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n")
	mock := &MockStream{Ctx: context.Background(), SendError: status.Error(codes.Canceled, "context canceled"),
		Requests: []*ext_proc.ProcessingRequest{{Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: chunk, EndOfStream: false}}}}}
	_ = (&OpenAIRouter{Config: &config.RouterConfig{}}).processWithContext(mock, ctx)
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] != turnFailureClientEnded || usage["failure_detail"] != "grpc:canceled" {
		t.Fatalf("llm_usage class %v detail %v, want client_ended grpc:canceled", usage["failure_class"], usage["failure_detail"])
	}
}

// Content observed in a chunk whose Send failed never reached the client:
// only what earlier, sent chunks carried counts as sent.
func TestContentInAChunkThatFailedToSendIsNotSent(t *testing.T) {
	start := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n"
	content := "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first words\"}}\n\n"
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-content-unsent", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	mock := &MockStream{Ctx: context.Background(), SendError: status.Error(codes.Canceled, "context canceled"),
		Requests: []*ext_proc.ProcessingRequest{{Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(start + content), EndOfStream: false}}}}}
	_ = (&OpenAIRouter{Config: &config.RouterConfig{}}).processWithContext(mock, ctx)
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] != turnFailureClientEnded || usage["content_sent_before_failure"] != false {
		t.Fatalf("llm_usage class %v content sent %v, want client_ended with nothing sent", usage["failure_class"], usage["content_sent_before_failure"])
	}
}

// Whatever the class, content in a chunk whose Send failed was not sent: a
// provider error in the same chunk as the first content is recorded with
// nothing delivered.
func TestAProviderErrorInAChunkThatFailedToSendSentNothing(t *testing.T) {
	frames := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first words\"}}\n\n" +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-provider-unsent", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	mock := &MockStream{Ctx: context.Background(), SendError: status.Error(codes.Canceled, "context canceled"),
		Requests: []*ext_proc.ProcessingRequest{{Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(frames), EndOfStream: false}}}}}
	_ = (&OpenAIRouter{Config: &config.RouterConfig{}}).processWithContext(mock, ctx)
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] == turnFailureClientEnded || usage["content_sent_before_failure"] != false {
		t.Fatalf("llm_usage class %v content sent %v, want the provider's class with nothing sent", usage["failure_class"], usage["content_sent_before_failure"])
	}
}

// Every path that settles a failed call at the headers (an upstream error,
// a transport error, an empty reply, a refused selection commit) writes its
// usage through reportFailedCallUsage. The exchange ending afterwards, at a
// receive or at a send, writes no second line.
func TestASettledFailedCallIsNotFinalizedAgain(t *testing.T) {
	for name, end := range map[string]func(*OpenAIRouter, *RequestContext){
		"receive": func(r *OpenAIRouter, ctx *RequestContext) {
			_ = r.handleProcessReceiveError(ctx, status.Error(codes.Canceled, "gone"))
		},
		"send": func(r *OpenAIRouter, ctx *RequestContext) {
			r.finalizeEndedStream(ctx, status.Error(codes.Unavailable, "gone"))
		},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			ctx := &RequestContext{RequestID: "req-settled", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
				SemanticStreamState: &semanticResponseStreamState{items: map[int]*semanticStreamItem{}}}
			recordTurnFailure(ctx, selectionFailureUnavailable, false)
			router.reportFailedCallUsage(ctx)
			end(router, ctx)
			usageLines := 0
			for _, entry := range logs.All() {
				if entry.ContextMap()["event"] == "llm_usage" {
					usageLines++
				}
			}
			if usageLines != 1 || ctx.ResponseFailureClass != selectionFailureUnavailable {
				t.Fatalf("%d usage lines, class %q; want one line and the settled class", usageLines, ctx.ResponseFailureClass)
			}
		})
	}
}

// The exchange can end before any response body arrived: the semantic stream
// state does not exist yet. The turn is still classed client_ended, with
// nothing sent, and nothing panics.
func TestAStreamEndedBeforeItsFirstBodyIsClassedClientEnded(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "req-ended-at-headers", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1}
	(&OpenAIRouter{}).finalizeEndedStream(ctx, status.Error(codes.Canceled, "context canceled"))
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] != turnFailureClientEnded || usage["content_sent_before_failure"] != false {
		t.Fatalf("llm_usage class %v content sent %v, want client_ended with nothing sent", usage["failure_class"], usage["content_sent_before_failure"])
	}
}

func TestAStreamEndedSendingItsHeadersIsClassedClientEnded(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "req-headers-send", RequestModel: "test", VSRSelectedModel: "arm-x", StartTime: time.Now(), Headers: map[string]string{},
		SemanticRequest: testNeutralRequest("test", "hello"), IsStreamingResponse: true,
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1}
	headers := arcResponseHeaders("200")
	headers.ResponseHeaders.Headers.Headers = append(headers.ResponseHeaders.Headers.Headers,
		&core.HeaderValue{Key: "content-type", RawValue: []byte("text/event-stream")})
	mock := &MockStream{Ctx: context.Background(), SendError: status.Error(codes.Canceled, "context canceled"),
		Requests: []*ext_proc.ProcessingRequest{{Request: headers}}}
	err := (&OpenAIRouter{Config: &config.RouterConfig{}}).processWithContext(mock, ctx)
	if status.Code(err) == codes.Internal {
		t.Fatalf("processing panicked: %v", err)
	}
	usage := findLogEvent(t, logs, "llm_usage")
	if usage["failure_class"] != turnFailureClientEnded || usage["content_sent_before_failure"] != false {
		t.Fatalf("llm_usage class %v content sent %v, want client_ended with nothing sent", usage["failure_class"], usage["content_sent_before_failure"])
	}
}
