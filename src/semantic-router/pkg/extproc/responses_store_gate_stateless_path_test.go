package extproc

import (
	"bytes"
	"context"
	"errors"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// failingResponseStore is an enabled store whose history lookup fails for a
// reason other than the id being unknown: the router undertook to keep
// history and cannot load it, which is its own fault and not the caller's.
type failingResponseStore struct {
	*MockResponseStore
}

func newFailingResponseStore() *failingResponseStore {
	return &failingResponseStore{MockResponseStore: NewMockResponseStore()}
}

func (s *failingResponseStore) GetConversationChain(context.Context, string) ([]*responseapi.StoredResponse, error) {
	return nil, errors.New("store backend unreachable")
}

// storeGateRequest runs a Responses body through the ingress gate every live
// ExtProc Responses request passes, with the given filter in front of it, and
// hands back the router so the caller can re-encode the refusal.
func storeGateRequest(
	t *testing.T,
	filter *ResponseAPIFilter,
	body string,
) (*OpenAIRouter, *RequestContext, *llmprotocol.Request, *ext_proc.ProcessingResponse) {
	t.Helper()
	router := &OpenAIRouter{ResponseAPIFilter: filter}
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
	request, immediate := router.prepareProtocolRequest([]byte(body), ctx)
	return router, ctx, request, immediate
}

const storeGateContinuationBody = `{"model":"model-a","input":"continue","previous_response_id":"resp_earlier"}`

// A cell with no response store can never resolve a previous_response_id, so
// the store gate refuses the turn the way the ARC policy path and the
// decision API already do: 400, x-vsr-failure-class: stateless_path, and the
// message that names the remedy. The old answer was a 503 with no class,
// which reads as an outage and invites a retry that cannot succeed
// (memex-desktop#7484). The gate runs before #280's branch on the ExtProc
// path, so without this the 400 that PR introduced never reached a client.
func TestResponsesStoreGateRefusesPreviousResponseIDWithoutAStore(t *testing.T) {
	for name, filter := range map[string]*ResponseAPIFilter{
		"no filter":      nil,
		"disabled store": {},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx, request, immediate := storeGateRequest(t, filter, storeGateContinuationBody)
			if request != nil || immediate == nil {
				t.Fatalf("request=%+v immediate=%+v, want a refusal", request, immediate)
			}
			if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != selectionFailureStatelessPath {
				t.Fatalf("protocol error = %+v, want code %q", ctx.ImmediateProtocolError, selectionFailureStatelessPath)
			}
			if got := immediateHeaderValue(immediate, selectionFailureHeader); got != selectionFailureStatelessPath {
				t.Fatalf("%s = %q before encoding, want %q", selectionFailureHeader, got, selectionFailureStatelessPath)
			}
			encoded := router.encodeImmediateResponseForClient(immediate, ctx)
			response := encoded.GetImmediateResponse()
			if response == nil {
				t.Fatalf("response is not an immediate refusal: %+v", encoded)
			}
			if got := int(response.GetStatus().GetCode()); got != 400 {
				t.Fatalf("status = %d, want 400", got)
			}
			message, errorType := immediateErrorBody(t, response.GetBody())
			if message != statelessPathMessage {
				t.Fatalf("message = %q, want %q", message, statelessPathMessage)
			}
			if errorType != "invalid_request_error" {
				t.Fatalf("error type = %q, want invalid_request_error", errorType)
			}
			// The header has to survive the re-encode: that step rewrites the
			// body and content-type, and a class dropped there is never seen.
			if got := immediateHeaderValue(encoded, selectionFailureHeader); got != selectionFailureStatelessPath {
				t.Fatalf("%s = %q after encoding, want %q", selectionFailureHeader, got, selectionFailureStatelessPath)
			}
		})
	}
}

// The gate's refusal and the ARC path's refusal are one contract. A caller
// must not be able to tell which gate turned it away: same status, same
// class, same body bytes once encoded for the client.
func TestResponsesStoreGateRefusalMatchesTheARCPathsRefusal(t *testing.T) {
	router, gateCtx, _, gateResponse := storeGateRequest(t, nil, storeGateContinuationBody)
	gateEncoded := router.encodeImmediateResponseForClient(gateResponse, gateCtx)

	arcCtx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
	arcResponse := router.statelessPathResponse(arcCtx)
	appendImmediateHeader(arcResponse, selectionFailureHeader, publicSelectionFailureClass(arcFailureResponsesPreviousID))
	arcEncoded := router.encodeImmediateResponseForClient(arcResponse, arcCtx)

	gate, arc := gateEncoded.GetImmediateResponse(), arcEncoded.GetImmediateResponse()
	if gate.GetStatus().GetCode() != arc.GetStatus().GetCode() {
		t.Fatalf("status: gate %d, ARC path %d", gate.GetStatus().GetCode(), arc.GetStatus().GetCode())
	}
	if !bytes.Equal(gate.GetBody(), arc.GetBody()) {
		t.Fatalf("body:\n gate %s\n ARC  %s", gate.GetBody(), arc.GetBody())
	}
	gateClass := immediateHeaderValue(gateEncoded, selectionFailureHeader)
	arcClass := immediateHeaderValue(arcEncoded, selectionFailureHeader)
	if gateClass == "" || gateClass != arcClass {
		t.Fatalf("%s: gate %q, ARC path %q", selectionFailureHeader, gateClass, arcClass)
	}
}

// Only the storeless refusal is reclassified. An id the store does not hold
// stays a 404; a store that fails to load history it keeps, or that the cell
// configured and could not open, stays a 503: those are the router's own
// breakage, and reading them as the caller's would hide an outage behind a
// "fix your request".
func TestResponsesStoreGateKeepsTheConfiguredStoreRefusals(t *testing.T) {
	for name, test := range map[string]struct {
		filter     *ResponseAPIFilter
		wantStatus int
		wantCode   string
		wantType   string
	}{
		"id unknown": {
			filter: NewResponseAPIFilter(NewMockResponseStore()), wantStatus: 404,
			wantCode: "previous_response_not_found", wantType: "invalid_request_error",
		},
		"lookup fails": {
			filter: NewResponseAPIFilter(newFailingResponseStore()), wantStatus: 503,
			wantCode: "response_history_unavailable", wantType: "server_error",
		},
		"store failed to open": {
			filter: NewUnavailableResponseAPIFilter(), wantStatus: 503,
			wantCode: "response_history_unavailable", wantType: "server_error",
		},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx, request, immediate := storeGateRequest(t, test.filter, storeGateContinuationBody)
			if request != nil || immediate == nil {
				t.Fatalf("request=%+v immediate=%+v, want a refusal", request, immediate)
			}
			if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != test.wantCode {
				t.Fatalf("protocol error = %+v, want code %q", ctx.ImmediateProtocolError, test.wantCode)
			}
			encoded := router.encodeImmediateResponseForClient(immediate, ctx)
			response := encoded.GetImmediateResponse()
			if got := int(response.GetStatus().GetCode()); got != test.wantStatus {
				t.Fatalf("status = %d, want %d", got, test.wantStatus)
			}
			if got := immediateHeaderValue(encoded, selectionFailureHeader); got != "" {
				t.Fatalf("%s = %q, want none: this refusal is not a selection class", selectionFailureHeader, got)
			}
			message, errorType := immediateErrorBody(t, response.GetBody())
			if message == statelessPathMessage {
				t.Fatalf("an enabled-store refusal must not tell the caller to resend items: %q", message)
			}
			if errorType != test.wantType {
				t.Fatalf("error type = %q, want %q", errorType, test.wantType)
			}
		})
	}
}

// A cell that configured a store and could not open it is not a storeless
// cell. The filter it builds is not nil, keeps retention off, and refuses
// previous_response_id as the router's outage: the id may resolve once the
// store is back, so the caller must not be told to rewrite the request.
func TestResponsesStoreInitFailureStaysUnavailable(t *testing.T) {
	filter := createResponseAPIFilter(&config.RouterConfig{
		ResponseAPI: config.ResponseAPIConfig{Enabled: true, StoreBackend: "no-such-backend"},
	})
	if filter == nil {
		t.Fatal("a configured store that failed to open must not read as no store")
	}
	if filter.IsEnabled() {
		t.Fatal("a store that failed to open must not read as enabled")
	}
	router, ctx, request, immediate := storeGateRequest(t, filter, storeGateContinuationBody)
	if request != nil || immediate == nil {
		t.Fatalf("request=%+v immediate=%+v, want a refusal", request, immediate)
	}
	encoded := router.encodeImmediateResponseForClient(immediate, ctx)
	if got := int(encoded.GetImmediateResponse().GetStatus().GetCode()); got != 503 {
		t.Fatalf("status = %d, want 503: this is the router's outage, not the caller's error", got)
	}
	if got := immediateHeaderValue(encoded, selectionFailureHeader); got != "" {
		t.Fatalf("%s = %q, want none", selectionFailureHeader, got)
	}
	// Stateless turns still proceed: retention is off, generation is not.
	_, ctx, request, immediate = storeGateRequest(t, filter, `{"model":"model-a","input":"hello"}`)
	if immediate != nil || request == nil {
		t.Fatalf("request=%+v immediate=%+v protocol error=%+v, want admission",
			request, immediate, ctx.ImmediateProtocolError)
	}
}

// A store is only needed to resolve an id. A Responses turn that sends its
// items and names no previous_response_id proceeds on a cell with no store,
// exactly as before.
func TestResponsesStoreGateAdmitsAStatelessTurnWithoutAStore(t *testing.T) {
	_, ctx, request, immediate := storeGateRequest(t, &ResponseAPIFilter{}, `{"model":"model-a","input":"hello"}`)
	if immediate != nil || request == nil {
		t.Fatalf("request=%+v immediate=%+v protocol error=%+v, want admission",
			request, immediate, ctx.ImmediateProtocolError)
	}
	if ctx.ResponseObjectState == nil || ctx.ResponseObjectState.PreviousResponseID != "" ||
		len(ctx.ResponseObjectState.ConversationHistory) != 0 {
		t.Fatalf("object state = %+v, want a fresh standalone turn", ctx.ResponseObjectState)
	}
}

// The status follows the error's category, not its text: what the caller
// sent is a 400, what the store does not hold is a 404, and what the router
// failed at is a 503.
func TestResponseObjectStateHTTPStatusFollowsTheCategory(t *testing.T) {
	for name, test := range map[string]struct {
		err  *llmprotocol.ProtocolError
		want int
	}{
		"nil":                {nil, 503},
		"caller's to fix":    {llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, selectionFailureStatelessPath, statelessPathMessage, nil), 400},
		"unknown id":         {llmprotocol.NewError(llmprotocol.ErrorNotFound, "previous_response_not_found", "", nil), 404},
		"router's own fault": {llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "response_history_unavailable", "", nil), 503},
		"anything else":      {llmprotocol.NewError(llmprotocol.ErrorInternal, "other", "", nil), 503},
	} {
		t.Run(name, func(t *testing.T) {
			if got := responseObjectStateHTTPStatus(test.err); got != test.want {
				t.Fatalf("status = %d, want %d", got, test.want)
			}
		})
	}
}

// The whole ExtProc request path, header phase then body phase, on a cell
// that keeps no store: the header phase used to answer 404 for every POST
// /v1/responses here, so the body gate's answer was never reachable live. A
// turn naming previous_response_id now gets the 400 and its class, and a
// turn that sends its items is admitted.
func TestStorelessCellAnswersPreviousResponseIDThroughTheRequestPhases(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	headerPhase := func(t *testing.T) *RequestContext {
		t.Helper()
		ctx := &RequestContext{Headers: map[string]string{}, RequestID: "storeless", TraceContext: t.Context()}
		headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":method", Value: "POST"},
				{Key: ":path", Value: "/v1/responses"},
				{Key: "content-type", Value: "application/json"},
			}},
		}}
		response, err := router.handleRequestHeaders(headers, ctx)
		if err != nil || response.GetImmediateResponse() != nil {
			t.Fatalf("header phase: err=%v immediate=%+v, want the request admitted", err, response.GetImmediateResponse())
		}
		return ctx
	}
	run := func(t *testing.T, body string) (*ext_proc.ProcessingResponse, *RequestContext) {
		t.Helper()
		ctx := headerPhase(t)
		response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
		}, ctx)
		if err != nil {
			t.Fatalf("body phase: %v", err)
		}
		return router.encodeImmediateResponseForClient(response, ctx), ctx
	}

	t.Run("a turn that sends its items is admitted, and store: true retains nothing", func(t *testing.T) {
		// store: true is accepted rather than refused: a client that always
		// sends it (the OpenAI default) must not be turned away by a cell
		// that merely retains nothing. Nothing is persisted, and the turn
		// is answered without a store-addressable id to continue from.
		// The ingress gate is the last phase this bare router can run; the
		// routing stages after it need a built router and are not at issue.
		ctx := headerPhase(t)
		request, immediate := router.prepareProtocolRequest([]byte(`{"model":"model-a","input":"hello","store":true}`), ctx)
		if immediate != nil || request == nil {
			t.Fatalf("stateless turn refused: %+v (protocol error %+v)", immediate, ctx.ImmediateProtocolError)
		}
		if ctx.ResponseObjectState == nil || !ctx.ResponseObjectState.ShouldStore {
			t.Fatalf("object state = %+v, want store: true recorded as asked", ctx.ResponseObjectState)
		}
		ctx.SemanticResponse = &llmprotocol.Response{ID: ctx.ResponseObjectState.GeneratedResponseID}
		ctx.UpstreamStatusCode = 200
		router.persistResponseObject(ctx)
		if ctx.ResponseObjectState.PersistenceAttempted {
			t.Fatal("a cell with no store must not attempt persistence")
		}
	})

	t.Run("previous_response_id is refused as stateless_path", func(t *testing.T) {
		response, _ := run(t, storeGateContinuationBody)
		immediate := response.GetImmediateResponse()
		if immediate == nil {
			t.Fatalf("response = %+v, want an immediate refusal", response)
		}
		if got := int(immediate.GetStatus().GetCode()); got != 400 {
			t.Fatalf("status = %d, want 400", got)
		}
		if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureStatelessPath {
			t.Fatalf("%s = %q, want %q", selectionFailureHeader, got, selectionFailureStatelessPath)
		}
		if message, _ := immediateErrorBody(t, immediate.GetBody()); message != statelessPathMessage {
			t.Fatalf("message = %q, want %q", message, statelessPathMessage)
		}
	})
}
