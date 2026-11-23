package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// fixedBedrockKey stands in for a short-term key source.
type fixedBedrockKey struct{ err error }

func (s fixedBedrockKey) Token(context.Context) (authz.AccessToken, error) {
	if s.err != nil {
		return authz.AccessToken{}, s.err
	}
	return authz.AccessToken{Value: "bedrock-api-key-test", Expiry: time.Now().Add(time.Hour)}, nil
}

// bedrockTestRouter serves one worker on Amazon Bedrock at baseURL, keyed by
// the bedrock-short-term-key provider.
func bedrockTestRouter(t *testing.T, baseURL string, format llmprotocol.WireFormat, upstreamModel string) (*OpenAIRouter, string) {
	t.Helper()
	logicalModel := "bedrock-worker"
	apiFormat := config.APIFormatOpenAI
	headers := map[string]string(nil)
	if format == llmprotocol.AnthropicMessagesV1 {
		apiFormat = config.APIFormatAnthropic
		headers = map[string]string{"anthropic-version": "2023-06-01"}
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
					PreferredEndpoints: []string{"bedrock"},
					APIFormat:          apiFormat,
					ExternalModelIDs:   map[string]string{"bedrock": upstreamModel},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "bedrock", Address: address, Port: port, Type: "bedrock", ProviderProfileName: "bedrock",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"bedrock": {Type: "bedrock", BaseURL: baseURL, ExtraHeaders: headers},
			},
		},
	}
	resolver := authz.NewCredentialResolver(authz.NewBedrockKeyProvider(fixedBedrockKey{}))
	return &OpenAIRouter{Config: cfg, CredentialResolver: resolver}, logicalModel
}

// Claude on Bedrock's Messages endpoint: one path for streaming and not, the
// key as a Bearer token, anthropic-version as a header, and the inference
// profile id as the body's model. Nothing Vertex-specific is applied.
func TestBedrockClaudeDispatchAtExtProcBoundary(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			router, model := bedrockTestRouter(t, "https://bedrock-runtime.us-east-1.amazonaws.com/anthropic",
				llmprotocol.AnthropicMessagesV1, "us.anthropic.claude-x")
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
			if headers[":path"] != "/anthropic/v1/messages" {
				t.Fatalf(":path = %q", headers[":path"])
			}
			if headers["Authorization"] != "Bearer bedrock-api-key-test" {
				t.Fatalf("Authorization = %q", headers["Authorization"])
			}
			if headers["anthropic-version"] != "2023-06-01" {
				t.Fatalf("anthropic-version = %q", headers["anthropic-version"])
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			if string(body["model"]) != `"us.anthropic.claude-x"` {
				t.Fatalf("model = %s, want the inference profile id", body["model"])
			}
			if _, ok := body["anthropic_version"]; ok {
				t.Fatalf("body carries anthropic_version: %s", common.GetBodyMutation().GetBody())
			}
			if stream && string(body["stream"]) != "true" {
				t.Fatalf("stream = %s", body["stream"])
			}
		})
	}
}

func TestBedrockOpenAICompatibleDispatchAtExtProcBoundary(t *testing.T) {
	router, model := bedrockTestRouter(t, "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
		llmprotocol.OpenAIChatV1, "us.openai.gpt-x")
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
	if headers[":path"] != "/openai/v1/chat/completions" {
		t.Fatalf(":path = %q", headers[":path"])
	}
	if headers["Authorization"] != "Bearer bedrock-api-key-test" {
		t.Fatalf("Authorization = %q", headers["Authorization"])
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if string(body["model"]) != `"us.openai.gpt-x"` {
		t.Fatalf("model = %s", body["model"])
	}
}

// No key (no AWS credentials) refuses the dispatch with an immediate 401.
func TestBedrockDispatchFailsClosedWithoutKey(t *testing.T) {
	router, model := bedrockTestRouter(t, "https://bedrock-runtime.us-east-1.amazonaws.com/anthropic",
		llmprotocol.AnthropicMessagesV1, "us.anthropic.claude-x")
	router.CredentialResolver = authz.NewCredentialResolver(authz.NewBedrockKeyProvider(
		fixedBedrockKey{err: errors.New("no credentials")}))
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)

	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err != nil {
		t.Fatalf("want an immediate 401, got the error %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate == nil || immediate.GetStatus().GetCode() != 401 {
		t.Fatalf("want the dispatch refused without a key, got %v", response)
	}
}

// The direct re-send path (no Envoy) reaches Bedrock's Messages endpoint with
// the same path, key and body; an httptest server stands in for Bedrock.
func TestBedrockClaudeResendReachesUpstream(t *testing.T) {
	type seen struct {
		path, auth, version string
		body                map[string]json.RawMessage
	}
	got := make(chan seen, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		_ = json.Unmarshal(raw, &body)
		got <- seen{path: r.URL.Path, auth: r.Header.Get("Authorization"), version: r.Header.Get("anthropic-version"), body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
	}))
	defer upstream.Close()

	router, model := bedrockTestRouter(t, upstream.URL+"/anthropic", llmprotocol.AnthropicMessagesV1, "us.anthropic.claude-x")
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

	profile := router.Config.ProviderProfiles["bedrock"]
	dispatch := &providerDispatch{
		logicalModel:   model,
		upstreamModel:  "us.anthropic.claude-x",
		backendAddress: strings.TrimPrefix(upstream.URL, "http://"),
		backendName:    "bedrock",
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
	if s.path != "/anthropic/v1/messages" || s.auth != "Bearer bedrock-api-key-test" || s.version != "2023-06-01" {
		t.Fatalf("upstream path %q auth %q version %q", s.path, s.auth, s.version)
	}
	if string(s.body["model"]) != `"us.anthropic.claude-x"` {
		t.Fatalf("upstream model = %s", s.body["model"])
	}
}
