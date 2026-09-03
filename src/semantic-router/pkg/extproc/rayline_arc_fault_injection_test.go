package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func faultInjectionContext(t *testing.T, enabled bool, faultHeader string) (*OpenAIRouter, *RequestContext, *config.AlgorithmConfig) {
	t.Helper()
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 4, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	requestHeaders := map[string]string{testEpisodeIDHeader: "session-1"}
	if faultHeader != "" {
		requestHeaders[headers.VSRFault] = faultHeader
	}
	requestContext := &RequestContext{
		RequestID:        "req-fault",
		VSRSelectedModel: "arm-1",
		Headers:          requestHeaders,
		SourceFormat:     llmprotocol.OpenAIChatV1,
		TargetFormat:     llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "public test turn"}},
			}},
		},
		TraceContext: context.Background(),
	}
	algorithm := &config.AlgorithmConfig{
		Type:    config.RaylineARCAlgorithmType,
		OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{
				IDHeader: testEpisodeIDHeader, AcquireTimeoutSeconds: 1,
			},
			FaultInjection: config.RaylineARCFaultInjectionConfig{Enabled: enabled},
		},
	}
	return &OpenAIRouter{RaylineARCEpisodeStore: store, Config: &config.RouterConfig{}}, requestContext, algorithm
}

// A decodable upstream body. The fault has to be what fails it, not the body.
func decodableUpstreamBody() []byte {
	return []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)
}

// With the cell's flag on, the header sends a healthy response down the
// body-phase failure path, which is the path that has no natural trigger left.
func TestFaultInjectionDrivesTheBodyPhaseFailurePath(t *testing.T) {
	router, requestContext, algorithm := faultInjectionContext(t, true, headers.FaultUpstreamDecode)
	router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())

	response := router.handleNonStreamingResponseBody(decodableUpstreamBody(), requestContext, 0)
	encoded := router.encodeImmediateResponseForClient(response, requestContext)
	immediate := encoded.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("fault did not produce an immediate refusal: %+v", encoded)
	}
	if got := int(immediate.GetStatus().GetCode()); got != 502 {
		t.Fatalf("status = %d, want 502", got)
	}
	values := immediateHeaderMap(t, response)
	if values[headers.RequestID] != "req-fault" || values[headers.VSRSelectedModel] != "arm-1" {
		t.Fatalf("headers = %v, want the published contract", values)
	}
	if extra := unexpectedVSRHeaders(values); len(extra) > 0 {
		t.Fatalf("fault response carries %v, want only the published contract", extra)
	}
	var body struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.GetBody(), &body); err != nil {
		t.Fatalf("body is not JSON: %s", immediate.GetBody())
	}
	if body.Error.Message == "" {
		t.Fatal("fault response carries no message")
	}
}

// With the flag off the header means nothing, so the response is served.
func TestFaultInjectionHeaderIsInertWhenDisabled(t *testing.T) {
	router, requestContext, algorithm := faultInjectionContext(t, false, headers.FaultUpstreamDecode)
	router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())
	if requestContext.InjectedFault != "" {
		t.Fatalf("fault = %q, want none while the cell has the flag off", requestContext.InjectedFault)
	}
	response := router.handleNonStreamingResponseBody(decodableUpstreamBody(), requestContext, 0)
	if response.GetImmediateResponse() != nil {
		t.Fatal("a disabled fault still refused the response")
	}
}

// No header, flag on: nothing changes.
func TestFaultInjectionNeedsItsHeader(t *testing.T) {
	router, requestContext, algorithm := faultInjectionContext(t, true, "")
	router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())
	if requestContext.InjectedFault != "" {
		t.Fatalf("fault = %q, want none without the header", requestContext.InjectedFault)
	}
	response := router.handleNonStreamingResponseBody(decodableUpstreamBody(), requestContext, 0)
	if response.GetImmediateResponse() != nil {
		t.Fatal("a request with no fault header was refused")
	}
}

// The header is the caller's instruction to the router and must never reach
// the provider, on or off.
func TestFaultInjectionHeaderIsStrippedBeforeDispatch(t *testing.T) {
	removed := strings.Join(faultInjectionHeadersForRemoval(), ",")
	if !strings.Contains(removed, headers.VSRFault) {
		t.Fatalf("removal list %q does not strip %s", removed, headers.VSRFault)
	}
}

// A cell that has been given the fault affordance says so at boot. The setting
// must never be something an operator has to read the config to discover.
func TestFaultInjectionEnabledIsAnnouncedAtStartup(t *testing.T) {
	enabled := &config.Decision{
		Name: "rayline-arc-dev",
		Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType,
			RaylineARC: &config.RaylineARCAlgorithmConfig{
				FaultInjection: config.RaylineARCFaultInjectionConfig{Enabled: true},
			},
		},
	}
	disabled := &config.Decision{
		Name: "rayline-arc-prod",
		Algorithm: &config.AlgorithmConfig{
			Type:       config.RaylineARCAlgorithmType,
			RaylineARC: &config.RaylineARCAlgorithmConfig{},
		},
	}
	// The warning walks every configured decision and reports only the opted-in
	// ones. It must not panic on a decision that never opted in.
	warnFaultInjectionEnabled([]*config.Decision{disabled, enabled, disabled})
}
