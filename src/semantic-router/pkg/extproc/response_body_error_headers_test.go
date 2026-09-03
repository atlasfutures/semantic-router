package extproc

import (
	"context"
	"sort"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func immediateHeaderMap(t *testing.T, response *ext_proc.ProcessingResponse) map[string]string {
	t.Helper()
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("response is not an immediate refusal: %+v", response)
	}
	values := map[string]string{}
	if immediate.Headers == nil {
		return values
	}
	for _, option := range immediate.Headers.SetHeaders {
		values[strings.ToLower(option.GetHeader().GetKey())] = string(option.GetHeader().GetRawValue())
	}
	return values
}

// A response the router replaces during the body phase is applied after
// Envoy's header scrub has already run, so whatever it sets reaches the
// client verbatim. It must therefore present the published contract itself:
// the request id, the model that was selected, the client protocol, and no
// other x-vsr header. The keystone pair is scrubbed on a normal response and
// must not survive here either.
// unexpectedVSRHeaders lists the x-vsr headers a body-phase refusal carries
// beyond the two the contract names.
func unexpectedVSRHeaders(values map[string]string) []string {
	var extra []string
	for key := range values {
		if !strings.HasPrefix(key, "x-vsr-") {
			continue
		}
		if key == headers.VSRSelectedModel || key == headers.VSRClientProtocol {
			continue
		}
		extra = append(extra, key)
	}
	sort.Strings(extra)
	return extra
}

func assertBodyPhaseErrorHeaders(t *testing.T, format llmprotocol.WireFormat) {
	t.Helper()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	requestContext := &RequestContext{
		RequestID:        "req-abc",
		VSRSelectedModel: "deepseek/deepseek-v4-pro@thinking-off",
		SourceFormat:     format,
		TargetFormat:     llmprotocol.OpenAIChatV1,
		TraceContext:     context.Background(),
	}
	response := router.handleNonStreamingResponseBody(undecodableUpstreamBody(), requestContext, 0)
	values := immediateHeaderMap(t, response)

	if values[headers.RequestID] != "req-abc" {
		t.Fatalf("%s = %q, want the request id", headers.RequestID, values[headers.RequestID])
	}
	if values[headers.VSRSelectedModel] != "deepseek/deepseek-v4-pro@thinking-off" {
		t.Fatalf("%s = %q, want the selected model", headers.VSRSelectedModel, values[headers.VSRSelectedModel])
	}
	if values[headers.VSRClientProtocol] == "" {
		t.Fatalf("%s is absent, want the client protocol", headers.VSRClientProtocol)
	}
	if extra := unexpectedVSRHeaders(values); len(extra) > 0 {
		t.Fatalf("body-phase error carries %v, want only the published contract", extra)
	}
}

// A response the router replaces during the body phase is applied after
// Envoy's header scrub has already run, so whatever it sets reaches the
// client verbatim. It must therefore present the published contract itself:
// the request id, the model that was selected, the client protocol, and no
// other x-vsr header. The keystone pair is scrubbed on a normal response and
// must not survive here either.
func TestBodyPhaseErrorPresentsThePublishedHeaderContract(t *testing.T) {
	for name, format := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) { assertBodyPhaseErrorHeaders(t, format) })
	}
}
