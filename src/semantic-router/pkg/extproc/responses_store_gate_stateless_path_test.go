package extproc

import (
	"bytes"
	"context"
	"errors"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
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

// noneFilter is the filter a cell with store_backend: none builds.
func noneFilter() *ResponseAPIFilter {
	return NewResponseAPIFilter(responsestore.NewNoneStore())
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

// A cell whose store retains nothing can never resolve a previous_response_id,
// so the store gate refuses the turn the way the ARC policy path and the
// decision API already do: 400, x-vsr-failure-class: stateless_path, and the
// message that names the remedy. The gate runs before #280's branch on the
// ExtProc path, so without this the 400 that PR introduced never reached a
// client (memex-desktop#7484).
func TestResponsesStoreGateRefusesPreviousResponseIDOnANoneStore(t *testing.T) {
	router, ctx, request, immediate := storeGateRequest(t, noneFilter(), storeGateContinuationBody)
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
	// The header has to survive the re-encode: that step rewrites the body
	// and content-type, and a class dropped there is never seen.
	if got := immediateHeaderValue(encoded, selectionFailureHeader); got != selectionFailureStatelessPath {
		t.Fatalf("%s = %q after encoding, want %q", selectionFailureHeader, got, selectionFailureStatelessPath)
	}
}

// The gate's refusal and the ARC path's refusal are one contract. A caller
// must not be able to tell which gate turned it away: same status, same
// class, same body bytes once encoded for the client.
func TestResponsesStoreGateRefusalMatchesTheARCPathsRefusal(t *testing.T) {
	router, gateCtx, _, gateResponse := storeGateRequest(t, noneFilter(), storeGateContinuationBody)
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

// Only the retains-nothing refusal is classed as the caller's. An id a
// retaining store does not hold stays a 404; a store that fails to load
// history it keeps, one the cell configured and could not open, and a nil or
// disabled filter all stay a 503: those are the router's own breakage or
// absence, and reading them as the caller's would hide an outage behind a
// "fix your request". (The nil and disabled cases are unreachable on the
// ExtProc path, where the header phase answers 404 first; the status is
// pinned so the gate stays honest for any caller that reaches it.)
func TestResponsesStoreGateKeepsTheOtherRefusals(t *testing.T) {
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
		"no filter": {
			filter: nil, wantStatus: 503,
			wantCode: "response_history_unavailable", wantType: "server_error",
		},
		"disabled store": {
			filter: &ResponseAPIFilter{}, wantStatus: 503,
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
				t.Fatalf("this refusal must not tell the caller to resend items: %q", message)
			}
			if errorType != test.wantType {
				t.Fatalf("error type = %q, want %q", errorType, test.wantType)
			}
		})
	}
}

// A filter built over no store at all must not panic when the gate asks
// whether it retains. Retains is a method on the store now, so the nil is
// handled once at construction; a filter built from a nil store reads as
// disabled and as retaining nothing, and the gate answers the 503 it
// answered before, not a 400.
func TestNilStoreFilterDoesNotPanicAndStays503(t *testing.T) {
	var store responsestore.ResponseStore
	filter := NewResponseAPIFilter(store)
	if filter.IsEnabled() {
		t.Fatal("a filter over no store must not read as enabled")
	}
	if filter.retains {
		t.Fatal("a filter over no store must not read as retaining")
	}
	router, ctx, request, immediate := storeGateRequest(t, filter, storeGateContinuationBody)
	if request != nil || immediate == nil {
		t.Fatalf("request=%+v immediate=%+v, want a refusal", request, immediate)
	}
	encoded := router.encodeImmediateResponseForClient(immediate, ctx)
	if got := int(encoded.GetImmediateResponse().GetStatus().GetCode()); got != 503 {
		t.Fatalf("status = %d, want 503", got)
	}
	// Persistence reads the same flag and must not dereference the nil store.
	noStoreRouter := &OpenAIRouter{ResponseAPIFilter: filter}
	noStoreRouter.persistResponseObject(&RequestContext{
		SourceFormat:        llmprotocol.OpenAIResponsesV1,
		TraceContext:        t.Context(),
		ResponseObjectState: &ResponseObjectState{GeneratedResponseID: "resp_x", ShouldStore: true},
		SemanticResponse:    &llmprotocol.Response{ID: "resp_x"},
	})
	if err := filter.Close(); err != nil {
		t.Fatalf("Close() = %v", err)
	}
}

// The unavailable filter holds no store, so Retains cannot be asked of it.
// What matters is that its storeUnavailable check runs before the retains
// branch, so the outage keeps answering 503 and is never reported as a cell
// that retains nothing on purpose.
func TestUnavailableFilterRetainsNothingButStill503s(t *testing.T) {
	filter := NewUnavailableResponseAPIFilter()
	if filter.retains {
		t.Fatal("the unavailable filter holds no store, so it retains nothing")
	}
	router, ctx, _, immediate := storeGateRequest(t, filter, storeGateContinuationBody)
	if immediate == nil {
		t.Fatal("want a refusal")
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "response_history_unavailable" {
		t.Fatalf("protocol error = %+v, want response_history_unavailable, not the stateless_path 400",
			ctx.ImmediateProtocolError)
	}
	encoded := router.encodeImmediateResponseForClient(immediate, ctx)
	if got := int(encoded.GetImmediateResponse().GetStatus().GetCode()); got != 503 {
		t.Fatalf("status = %d, want 503", got)
	}
}

// A cell that configured a store and could not open it is not a cell that
// retains nothing. The filter it builds is not nil, is disabled (so the
// surface stays 404 at the header phase during the outage, as before), and
// at the gate refuses previous_response_id as the router's outage: the id
// may resolve once the store is back, so the caller must not be told to
// rewrite the request.
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
	if response := (&OpenAIRouter{ResponseAPIFilter: filter}).validateRequestHeaders("POST", "/v1/responses", &RequestContext{Headers: map[string]string{}}); response == nil ||
		response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound {
		t.Fatalf("POST /v1/responses during the outage = %+v, want the 404 it answered before", response)
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
}

// store_backend: none builds an enabled filter that retains nothing. The
// Responses surface stays on: the header phase admits POST /v1/responses and
// refuses other methods as it does with any enabled store, while GET and
// DELETE /v1/responses/{id} answer 404 as they do for an id never stored.
func TestNoneBackendKeepsTheResponsesSurfaceOn(t *testing.T) {
	filter := createResponseAPIFilter(&config.RouterConfig{
		ResponseAPI: config.ResponseAPIConfig{Enabled: true, StoreBackend: string(responsestore.NoneStoreType)},
	})
	if filter == nil || !filter.IsEnabled() {
		t.Fatalf("filter = %+v, want an enabled filter", filter)
	}
	if filter.retains {
		t.Fatal("the none backend must not read as retaining")
	}
	router := &OpenAIRouter{ResponseAPIFilter: filter}
	ctx := &RequestContext{Headers: map[string]string{}}
	if response := router.validateRequestHeaders("POST", "/v1/responses", ctx); response != nil {
		t.Fatalf("POST /v1/responses was refused at the header phase: %+v", response)
	}
	if response := router.validateRequestHeaders("GET", "/v1/responses", ctx); response == nil ||
		response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_MethodNotAllowed {
		t.Fatalf("GET /v1/responses = %+v, want 405", response)
	}
	for _, method := range []string{"GET", "DELETE"} {
		object, err := router.handleResponseAPIRequestHeaders(method, "/v1/responses/resp_never", &RequestContext{TraceContext: t.Context()})
		if err != nil || object == nil || object.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound {
			t.Fatalf("%s /v1/responses/resp_never = %+v, %v, want 404", method, object, err)
		}
	}
	items, err := router.handleResponseAPIRequestHeaders("GET", "/v1/responses/resp_never/input_items", &RequestContext{TraceContext: t.Context()})
	if err != nil || items == nil || items.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound {
		t.Fatalf("GET input_items = %+v, %v, want 404", items, err)
	}
}

// enabled: false is unchanged: the whole Responses surface is off, answered
// 404 at the header phase, whatever store_backend says.
func TestDisabledResponseAPIStillAnswers404AtTheHeaderPhase(t *testing.T) {
	for name, cfg := range map[string]config.ResponseAPIConfig{
		"disabled":                 {Enabled: false},
		"disabled with none named": {Enabled: false, StoreBackend: string(responsestore.NoneStoreType)},
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{ResponseAPIFilter: createResponseAPIFilter(&config.RouterConfig{ResponseAPI: cfg})}
			response := router.validateRequestHeaders("POST", "/v1/responses", &RequestContext{Headers: map[string]string{}})
			if response == nil || response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound {
				t.Fatalf("POST /v1/responses = %+v, want 404", response)
			}
		})
	}
}

// The whole ExtProc request path, header phase then body phase, on a cell
// with store_backend: none. A turn naming previous_response_id gets the 400
// and its class on the wire; a turn that sends its items is admitted, with
// store: true accepted, a response id minted, and nothing persisted.
func TestNoneBackendAnswersPreviousResponseIDThroughTheRequestPhases(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}, ResponseAPIFilter: noneFilter()}
	headerPhase := func(t *testing.T) *RequestContext {
		t.Helper()
		ctx := &RequestContext{Headers: map[string]string{}, RequestID: "none-backend", TraceContext: t.Context()}
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

	t.Run("a turn that sends its items is admitted, and store: true retains nothing", func(t *testing.T) {
		// store: true is accepted rather than refused: a client that always
		// sends it (the OpenAI default) must not be turned away by a cell
		// that merely retains nothing. The response id is still minted, so
		// turn one looks the same as on any other cell; nothing is written.
		// The ingress gate is the last phase this bare router can run; the
		// routing stages after it need a built router and are not at issue.
		ctx := headerPhase(t)
		request, immediate := router.prepareProtocolRequest([]byte(`{"model":"model-a","input":"hello","store":true}`), ctx)
		if immediate != nil || request == nil {
			t.Fatalf("stateless turn refused: %+v (protocol error %+v)", immediate, ctx.ImmediateProtocolError)
		}
		state := ctx.ResponseObjectState
		if state == nil || !state.ShouldStore || !responseapi.IsValidResponseID(state.GeneratedResponseID) {
			t.Fatalf("object state = %+v, want store: true recorded and a response id minted", state)
		}
		ctx.SemanticResponse = &llmprotocol.Response{ID: state.GeneratedResponseID}
		ctx.UpstreamStatusCode = 200
		router.persistResponseObject(ctx)
		if state.PersistenceAttempted {
			t.Fatal("a store that retains nothing must not attempt persistence")
		}
		if _, err := router.ResponseAPIFilter.store.GetResponse(t.Context(), state.GeneratedResponseID); !errors.Is(err, responsestore.ErrNotFound) {
			t.Fatalf("the turn was read back from a store that retains nothing: %v", err)
		}
	})

	t.Run("previous_response_id is refused as stateless_path", func(t *testing.T) {
		ctx := headerPhase(t)
		response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: []byte(storeGateContinuationBody), EndOfStream: true},
		}, ctx)
		if err != nil {
			t.Fatalf("body phase: %v", err)
		}
		response = router.encodeImmediateResponseForClient(response, ctx)
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

// A memory store's behaviour is unchanged by the capability: it retains, a
// stored turn resolves, and an unknown id is a 404.
func TestRetainingStoreStillResolvesPreviousResponseID(t *testing.T) {
	store := NewMockResponseStore()
	if err := store.StoreResponse(t.Context(), &responseapi.StoredResponse{ID: "resp_earlier"}); err != nil {
		t.Fatal(err)
	}
	filter := NewResponseAPIFilter(store)
	if !filter.retains {
		t.Fatal("a retaining store must be read as retaining")
	}
	_, ctx, request, immediate := storeGateRequest(t, filter, storeGateContinuationBody)
	if immediate != nil || request == nil {
		t.Fatalf("continuation refused on a retaining store: %+v (%+v)", immediate, ctx.ImmediateProtocolError)
	}
	if len(ctx.ResponseObjectState.ConversationHistory) != 1 {
		t.Fatalf("history = %+v, want the stored turn", ctx.ResponseObjectState.ConversationHistory)
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
