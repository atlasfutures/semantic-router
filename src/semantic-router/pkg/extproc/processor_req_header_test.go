package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func newRequestHeaders(method, path string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":method", Value: method},
				{Key: ":path", Value: path},
			}},
		},
	}
}

func TestDetectSourceFormat(t *testing.T) {
	tests := []struct {
		path string
		want llmprotocol.WireFormat
	}{
		{path: "/v1/chat/completions", want: llmprotocol.OpenAIChatV1},
		{path: "/v1/responses", want: llmprotocol.OpenAIResponsesV1},
		{path: "/v1/responses?stream=true", want: llmprotocol.OpenAIResponsesV1},
		{path: "/v1/messages", want: llmprotocol.AnthropicMessagesV1},
		{path: "/v1/messages/count_tokens", want: llmprotocol.AnthropicMessagesV1},
	}
	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			ctx := &RequestContext{}
			detectSourceFormat(test.path, ctx)
			if ctx.SourceFormat != test.want {
				t.Fatalf("source format = %q, want %q", ctx.SourceFormat, test.want)
			}
		})
	}
}

func TestApplyHeaderPassThroughPolicyDropsOnlyTransportHeaders(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"host":              "example.test",
		"content-length":    "42",
		"connection":        "keep-alive",
		"transfer-encoding": "chunked",
		"anthropic-version": "2024-10-22",
		"x-application":     "kept",
	}}
	applyHeaderPassThroughPolicy(ctx)
	for _, key := range []string{"host", "content-length", "connection", "transfer-encoding"} {
		if _, found := ctx.Headers[key]; found {
			t.Fatalf("transport header %q was retained", key)
		}
	}
	for _, key := range []string{"anthropic-version", "x-application"} {
		if _, found := ctx.Headers[key]; !found {
			t.Fatalf("application header %q was removed", key)
		}
	}
	applyHeaderPassThroughPolicy(nil)
}

func TestValidatePublicGenerationEndpoints(t *testing.T) {
	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore())}
	tests := []struct {
		method string
		path   string
		status typev3.StatusCode
	}{
		{method: "POST", path: "/v1/chat/completions"},
		{method: "POST", path: "/v1/responses"},
		{method: "POST", path: "/v1/messages"},
		{method: "GET", path: "/v1/messages", status: typev3.StatusCode_MethodNotAllowed},
		{method: "POST", path: "/v1/messages/count_tokens", status: typev3.StatusCode_NotFound},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			response := router.validateRequestHeaders(test.method, test.path, &RequestContext{Headers: map[string]string{}})
			if test.status == 0 {
				if response != nil {
					t.Fatalf("valid generation endpoint returned immediate response: %+v", response)
				}
				return
			}
			if response == nil || response.GetImmediateResponse() == nil ||
				response.GetImmediateResponse().GetStatus().GetCode() != test.status {
				t.Fatalf("status = %+v, want %s", response, test.status)
			}
		})
	}
}

// A cell that keeps no response store still serves Responses generation.
// The store is needed only to retain turns and to resolve previous_response_id,
// and the body gate answers the latter; a 404 here would take the endpoint
// down on a cell that merely retains nothing, and would hide that answer.
func TestStorelessCellStillServesResponsesGeneration(t *testing.T) {
	for name, filter := range map[string]*ResponseAPIFilter{
		"no filter":            nil,
		"disabled store":       {},
		"store failed to open": NewUnavailableResponseAPIFilter(),
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{ResponseAPIFilter: filter}
			ctx := &RequestContext{Headers: map[string]string{}}
			if response := router.validateRequestHeaders("POST", "/v1/responses", ctx); response != nil {
				t.Fatalf("POST /v1/responses was refused at the header phase: %+v", response)
			}
			response := router.validateRequestHeaders("GET", "/v1/responses", ctx)
			if response == nil || response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_MethodNotAllowed {
				t.Fatalf("GET /v1/responses = %+v, want 405", response)
			}
			// The object endpoints still need the store.
			for _, method := range []string{"GET", "DELETE"} {
				object, err := router.handleResponseAPIRequestHeaders(method, "/v1/responses/resp_missing", &RequestContext{TraceContext: t.Context()})
				if err != nil || object == nil || object.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound {
					t.Fatalf("%s /v1/responses/resp_missing = %+v, %v, want 404", method, object, err)
				}
			}
		})
	}
}

func TestResponseObjectPathParsing(t *testing.T) {
	if got := extractResponseIDFromPath("/v1/responses/resp_123?expand=true"); got != "resp_123" {
		t.Fatalf("response id = %q", got)
	}
	if got := extractResponseIDFromInputItemsPath("/v1/responses/resp_123/input_items"); got != "resp_123" {
		t.Fatalf("input-items response id = %q", got)
	}
	if got := extractResponseIDFromPath("/v1/responses/not-a-response"); got != "" {
		t.Fatalf("invalid response id = %q", got)
	}
}
