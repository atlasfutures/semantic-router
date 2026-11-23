package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAdaptVertexMessagesRequest(t *testing.T) {
	vertexMessages := &providerDispatch{
		profile: &config.ProviderProfile{Type: "vertex-ai"}, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	cases := []struct {
		name     string
		dispatch *providerDispatch
		body     string
		want     string
	}{
		{
			name:     "model is dropped and anthropic_version is set",
			dispatch: vertexMessages,
			body:     `{"model":"claude-x","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			want:     `{"max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}],"anthropic_version":"vertex-2023-10-16"}`,
		},
		{
			name:     "a caller's anthropic_version is replaced",
			dispatch: vertexMessages,
			body:     `{"anthropic_version":"other","messages":[]}`,
			want:     `{"anthropic_version":"vertex-2023-10-16","messages":[]}`,
		},
		{
			name:     "Vertex AI Chat Completions is left alone",
			dispatch: &providerDispatch{profile: &config.ProviderProfile{Type: "vertex-ai"}, targetFormat: llmprotocol.OpenAIChatV1},
			body:     `{"model":"publisher/m","messages":[]}`,
			want:     `{"model":"publisher/m","messages":[]}`,
		},
		{
			name:     "Anthropic itself is left alone",
			dispatch: &providerDispatch{profile: &config.ProviderProfile{Type: "anthropic"}, targetFormat: llmprotocol.AnthropicMessagesV1},
			body:     `{"model":"claude-x","messages":[]}`,
			want:     `{"model":"claude-x","messages":[]}`,
		},
		{
			name:     "no dispatch",
			dispatch: nil,
			body:     `{"model":"claude-x"}`,
			want:     `{"model":"claude-x"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := adaptVertexMessagesRequest([]byte(tc.body), tc.dispatch)
			if err != nil {
				t.Fatalf("adaptVertexMessagesRequest: %v", err)
			}
			if string(got) != tc.want {
				t.Fatalf("body = %s\nwant   %s", got, tc.want)
			}
		})
	}
	if _, err := adaptVertexMessagesRequest([]byte(`[1]`), vertexMessages); err == nil {
		t.Fatal("a non-object body was adapted; want error")
	}
}

// fakeMetadataServer serves Workload Identity style tokens and counts calls.
func fakeMetadataServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			http.Error(w, "missing Metadata-Flavor", http.StatusForbidden)
			return
		}
		n := calls.Add(1)
		_, _ = fmt.Fprintf(w, `{"access_token":"test-token-%d","expires_in":3599,"token_type":"Bearer"}`, n)
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func vertexTestRouter(t *testing.T, baseURL string, format llmprotocol.WireFormat) (*OpenAIRouter, string) {
	t.Helper()
	metadata, _ := fakeMetadataServer(t)
	logicalModel := "vertex-worker"
	apiFormat := config.APIFormatOpenAI
	if format == llmprotocol.AnthropicMessagesV1 {
		apiFormat = config.APIFormatAnthropic
	}
	host := strings.TrimPrefix(strings.TrimPrefix(baseURL, "https://"), "http://")
	host = strings.SplitN(host, "/", 2)[0]
	address, port := host, 443
	if h, p, ok := strings.Cut(host, ":"); ok {
		address = h
		_, _ = fmt.Sscanf(p, "%d", &port)
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: logicalModel,
			ModelConfig: map[string]config.ModelParams{
				logicalModel: {
					PreferredEndpoints: []string{"vertex"},
					APIFormat:          apiFormat,
					ExternalModelIDs:   map[string]string{"vertex-ai": "claude-x@v1"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "vertex", Address: address, Port: port, Type: "vertex-ai", ProviderProfileName: "vertex",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"vertex": {Type: "vertex-ai", BaseURL: baseURL},
			},
		},
	}
	resolver := authz.NewCredentialResolver(authz.NewCloudTokenProvider(
		&authz.MetadataTokenSource{Host: metadata.URL, Client: metadata.Client()},
	))
	return &OpenAIRouter{Config: cfg, CredentialResolver: resolver}, logicalModel
}

func TestVertexClaudeDispatchAtExtProcBoundary(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			router, model := vertexTestRouter(t,
				"https://vertex.example.com/v1/projects/p/locations/global", llmprotocol.AnthropicMessagesV1)
			request := testNeutralRequest(model, "hello")
			request.Stream = stream
			request.Sampling.MaxOutputTokens = llmprotocol.Int64(64)
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
			ctx.ExpectStreamingResponse = stream

			response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
			if err != nil {
				t.Fatalf("handleSpecifiedModelRouting: %v", err)
			}
			common := response.GetRequestBody().GetResponse()
			if common == nil {
				t.Fatalf("no continue response: %v", response)
			}
			headers := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
			method := "rawPredict"
			if stream {
				method = "streamRawPredict"
			}
			wantPath := "/v1/projects/p/locations/global/publishers/anthropic/models/claude-x@v1:" + method
			if headers[":path"] != wantPath {
				t.Fatalf(":path = %q, want %q", headers[":path"], wantPath)
			}
			if headers["Authorization"] != "Bearer test-token-1" {
				t.Fatalf("Authorization = %q", headers["Authorization"])
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			if _, ok := body["model"]; ok {
				t.Fatalf("body still names the model: %s", common.GetBodyMutation().GetBody())
			}
			if string(body["anthropic_version"]) != `"vertex-2023-10-16"` {
				t.Fatalf("anthropic_version = %s", body["anthropic_version"])
			}
			if _, ok := body["messages"]; !ok {
				t.Fatalf("body lost messages: %s", common.GetBodyMutation().GetBody())
			}
			if stream && string(body["stream"]) != "true" {
				t.Fatalf("stream = %s", body["stream"])
			}
		})
	}
}

func TestVertexOpenAICompatibleDispatchAtExtProcBoundary(t *testing.T) {
	router, model := vertexTestRouter(t,
		"https://vertex.example.com/v1/projects/p/locations/global/endpoints/openapi", llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	request.Stream = true
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ExpectStreamingResponse = true

	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	headers := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if want := "/v1/projects/p/locations/global/endpoints/openapi/chat/completions"; headers[":path"] != want {
		t.Fatalf(":path = %q, want %q", headers[":path"], want)
	}
	if headers["Authorization"] != "Bearer test-token-1" {
		t.Fatalf("Authorization = %q", headers["Authorization"])
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if string(body["model"]) != `"claude-x@v1"` {
		t.Fatalf("chat body model = %s, want the backend model id", body["model"])
	}
	if _, ok := body["anthropic_version"]; ok {
		t.Fatalf("chat body carries anthropic_version: %s", common.GetBodyMutation().GetBody())
	}
}

func TestVertexClaudeDispatchFailsClosedWithoutToken(t *testing.T) {
	router, model := vertexTestRouter(t,
		"https://vertex.example.com/v1/projects/p/locations/global", llmprotocol.AnthropicMessagesV1)
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer down.Close()
	router.CredentialResolver = authz.NewCredentialResolver(authz.NewCloudTokenProvider(
		&authz.MetadataTokenSource{Host: down.URL, Client: down.Client()},
	))
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)

	// The dispatch is refused (credential_resolution_failed) with an
	// immediate 401, never an ExtProc error: Envoy may be configured to let a
	// request through when its ExtProc fails.
	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err != nil {
		t.Fatalf("want an immediate 401, got the error %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate == nil || immediate.GetStatus().GetCode() != 401 {
		t.Fatalf("want the dispatch refused without a token, got %v", response)
	}
}

// A re-send re-reads the body it sends, which on Vertex AI Messages names no
// model: the served model is still the dispatch's.
func TestVertexMessagesOutboundModelSurvivesTheAdapter(t *testing.T) {
	dispatch := &providerDispatch{
		upstreamModel: "claude-x", targetFormat: llmprotocol.AnthropicMessagesV1,
		profile: &config.ProviderProfile{Type: "vertex-ai", BaseURL: "https://vertex.example.com/v1/projects/p/locations/global"},
	}
	adapted, err := adaptVertexMessagesRequest([]byte(`{"model":"claude-x","messages":[]}`), dispatch)
	if err != nil {
		t.Fatal(err)
	}
	if got := readDispatchedOutboundParams(adapted, dispatch).Model; got != "claude-x" {
		t.Fatalf("outbound model = %q", got)
	}
	// Another provider's body without a model stays without one.
	other := &providerDispatch{upstreamModel: "m", targetFormat: llmprotocol.AnthropicMessagesV1,
		profile: &config.ProviderProfile{Type: "anthropic"}}
	if got := readDispatchedOutboundParams([]byte(`{"messages":[]}`), other).Model; got != "" {
		t.Fatalf("non-Vertex outbound model = %q", got)
	}
}

// The direct re-send path (no Envoy) sends the same URL, bearer token and
// body to the provider; an httptest server stands in for Vertex AI.
func TestVertexClaudeResendReachesUpstream(t *testing.T) {
	type seen struct {
		path, auth string
		body       map[string]json.RawMessage
	}
	got := make(chan seen, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		_ = json.Unmarshal(raw, &body)
		got <- seen{path: r.URL.Path, auth: r.Header.Get("Authorization"), body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
	}))
	defer upstream.Close()

	router, model := vertexTestRouter(t, upstream.URL+"/v1/projects/p/locations/global", llmprotocol.AnthropicMessagesV1)
	request := testNeutralRequest(model, "hello")
	request.Stream = true
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(64)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true
	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()

	profile := router.Config.ProviderProfiles["vertex"]
	dispatch := &providerDispatch{
		logicalModel:   model,
		upstreamModel:  "claude-x@v1",
		backendAddress: strings.TrimPrefix(upstream.URL, "http://"),
		backendName:    "vertex",
		profile:        &profile,
		targetFormat:   llmprotocol.AnthropicMessagesV1,
	}
	call, refused := router.buildResendCall(ctx, model, dispatch, body)
	if refused != "" {
		t.Fatalf("buildResendCall refused: %s", refused)
	}
	req, err := http.NewRequest(http.MethodPost, call.url, bytes.NewReader(call.body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header = call.headers
	resp, err := upstream.Client().Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	_ = resp.Body.Close()

	s := <-got
	if s.path != "/v1/projects/p/locations/global/publishers/anthropic/models/claude-x@v1:streamRawPredict" {
		t.Fatalf("upstream path = %q", s.path)
	}
	if !strings.HasPrefix(s.auth, "Bearer test-token-") {
		t.Fatalf("upstream Authorization = %q", s.auth)
	}
	if _, ok := s.body["model"]; ok {
		t.Fatal("upstream body names the model")
	}
	if string(s.body["anthropic_version"]) != `"vertex-2023-10-16"` {
		t.Fatalf("upstream anthropic_version = %s", s.body["anthropic_version"])
	}
}

// Vertex AI names the streaming operation in the URL, so the path follows the
// body actually sent, not the client's request: a parameter policy can drop
// the client's stream flag.
func TestVertexRequestPathFollowsTheSentBody(t *testing.T) {
	profile := &config.ProviderProfile{Type: "vertex-ai", BaseURL: "https://vertex.example.com/v1/projects/p/locations/global"}
	dispatch := &providerDispatch{
		upstreamModel: "claude-x", profile: profile, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	const prefix = "/v1/projects/p/locations/global/publishers/anthropic/models/claude-x:"
	for body, want := range map[string]string{
		`{"messages":[],"stream":true}`:  prefix + "streamRawPredict",
		`{"messages":[]}`:                prefix + "rawPredict",
		`{"messages":[],"stream":false}`: prefix + "rawPredict",
	} {
		mutation := &ext_proc.HeaderMutation{}
		// What the dispatch set from the client's request: streaming.
		setProviderRequestPath(&mutation.SetHeaders, profile, dispatch.targetFormat, "claude-x", true)
		mutation.SetHeaders = append(mutation.SetHeaders, overwriteRequestHeader("x-other", "1"))
		resetVertexRequestPath(mutation, dispatch, []byte(body))
		headers := headerValuesByName(mutation.SetHeaders)
		paths := 0
		for _, option := range mutation.SetHeaders {
			if option.GetHeader().GetKey() == ":path" {
				paths++
			}
		}
		if paths != 1 || headers[":path"] != want || headers["x-other"] != "1" {
			t.Fatalf("%s: :path = %q (%d set), want %q once", body, headers[":path"], paths, want)
		}
	}

	// Another provider's path is left as the dispatch set it.
	other := &providerDispatch{profile: &config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.example.com"},
		targetFormat: llmprotocol.AnthropicMessagesV1}
	mutation := &ext_proc.HeaderMutation{SetHeaders: []*core.HeaderValueOption{overwriteRequestHeader(":path", "/kept")}}
	resetVertexRequestPath(mutation, other, []byte(`{"stream":true}`))
	if got := headerValuesByName(mutation.SetHeaders)[":path"]; got != "/kept" {
		t.Fatalf("non-Vertex :path = %q", got)
	}
}

func TestVertexResendPathFollowsTheSentBody(t *testing.T) {
	router, model := vertexTestRouter(t, "https://vertex.example.com/v1/projects/p/locations/global", llmprotocol.AnthropicMessagesV1)
	profile := router.Config.ProviderProfiles["vertex"]
	dispatch := &providerDispatch{
		logicalModel: model, upstreamModel: "claude-x", backendAddress: "vertex.example.com:443",
		backendName: "vertex", profile: &profile, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, testNeutralRequest(model, "hi"))
	ctx.ExpectStreamingResponse = true
	call, refused := router.buildResendCall(ctx, model, dispatch, []byte(`{"messages":[]}`))
	if refused != "" {
		t.Fatalf("buildResendCall refused: %s", refused)
	}
	if want := "https://vertex.example.com:443/v1/projects/p/locations/global/publishers/anthropic/models/claude-x:rawPredict"; call.url != want {
		t.Fatalf("re-send url = %q, want %q", call.url, want)
	}
}
