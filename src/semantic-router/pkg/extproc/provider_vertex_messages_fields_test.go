package extproc

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A Claude Code request on a context-management beta carries a top-level
// context_management member. Vertex AI refuses it ("context_management:
// Extra inputs are not permitted") unless the beta arrives in the header, so
// the adapter sends only the members Vertex AI accepts and keeps every one of
// those byte for byte.
func TestVertexAdapterSendsOnlyAcceptedFieldsOfAClaudeCodeRequest(t *testing.T) {
	raw, err := os.ReadFile("testdata/vertex/claude_code_messages_request.json")
	if err != nil {
		t.Fatal(err)
	}
	dispatch := &providerDispatch{
		profile: &config.ProviderProfile{Type: "vertex-ai"}, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	got, dropped, err := adaptVertexMessagesRequest(raw, dispatch)
	if err != nil {
		t.Fatalf("adaptVertexMessagesRequest: %v", err)
	}
	if want := []string{"context_management"}; !reflect.DeepEqual(dropped, want) {
		t.Fatalf("dropped = %v, want %v", dropped, want)
	}
	var in, out map[string]json.RawMessage
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("adapted body is not JSON: %v", err)
	}
	for name := range out {
		if _, accepted := vertexMessagesFields[name]; !accepted {
			t.Errorf("sent member %q, which Vertex AI does not accept", name)
		}
	}
	for name, value := range in {
		switch name {
		case "model", "context_management":
			if _, sent := out[name]; sent {
				t.Errorf("%s was sent", name)
			}
		default:
			if string(out[name]) != string(value) {
				t.Errorf("%s changed:\n got %s\nwant %s", name, out[name], value)
			}
		}
	}
	if string(out["anthropic_version"]) != `"vertex-2023-10-16"` {
		t.Fatalf("anthropic_version = %s", out["anthropic_version"])
	}
}

func TestVertexAdapterDropsMembersOutsideTheAllowlist(t *testing.T) {
	dispatch := &providerDispatch{
		profile: &config.ProviderProfile{Type: "vertex-ai"}, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	body := `{"model":"m","messages":[],"max_tokens":8,"speed":"fast","container":"c","mcp_servers":[],` +
		`"service_tier":"auto","inference_geo":"us","a.b":1,"tool_choice":{"type":"auto"}}`
	got, dropped, err := adaptVertexMessagesRequest([]byte(body), dispatch)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"a.b", "container", "inference_geo", "mcp_servers", "service_tier", "speed"}; !reflect.DeepEqual(dropped, want) {
		t.Fatalf("dropped = %v, want %v", dropped, want)
	}
	if want := `{"messages":[],"max_tokens":8,"tool_choice":{"type":"auto"},"anthropic_version":"vertex-2023-10-16"}`; string(got) != want {
		t.Fatalf("body = %s\nwant   %s", got, want)
	}

	// Another provider's body keeps every member.
	other := &providerDispatch{profile: &config.ProviderProfile{Type: "anthropic"}, targetFormat: llmprotocol.AnthropicMessagesV1}
	kept, dropped, err := adaptVertexMessagesRequest([]byte(body), other)
	if err != nil || string(kept) != body || dropped != nil {
		t.Fatalf("non-Vertex body changed: %s %v %v", kept, dropped, err)
	}
}

func setHeaderValues(mutation *ext_proc.HeaderMutation, name string) []string {
	var values []string
	for _, option := range mutation.GetSetHeaders() {
		if strings.EqualFold(option.GetHeader().GetKey(), name) {
			value := string(option.GetHeader().GetRawValue())
			if value == "" {
				value = option.GetHeader().GetValue()
			}
			values = append(values, value)
		}
	}
	return values
}

func removesHeader(mutation *ext_proc.HeaderMutation, name string) bool {
	for _, removed := range mutation.GetRemoveHeaders() {
		if strings.EqualFold(removed, name) {
			return true
		}
	}
	return false
}

// The client's anthropic-beta is not forwarded to Vertex AI, on the first
// dispatch or a re-send: the body keeps no beta-gated member for it to
// enable, and Vertex AI refuses beta names it does not know. A beta the
// provider profile sets is sent.
func TestVertexDispatchDoesNotForwardTheClientsBetas(t *testing.T) {
	const clientBetas = "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27"
	for _, tc := range []struct {
		name        string
		profileBeta string
	}{
		{name: "no profile beta"},
		{name: "profile beta", profileBeta: "operator-beta-2026-01-01"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, model := vertexTestRouter(t,
				"https://vertex.example.com/v1/projects/p/locations/global", llmprotocol.AnthropicMessagesV1)
			if tc.profileBeta != "" {
				profile := router.Config.ProviderProfiles["vertex"]
				profile.ExtraHeaders = map[string]string{"anthropic-beta": tc.profileBeta}
				router.Config.ProviderProfiles["vertex"] = profile
			}
			request := testNeutralRequest(model, "hello")
			request.Sampling.MaxOutputTokens = llmprotocol.Int64(64)
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
			ctx.Headers["anthropic-beta"] = clientBetas

			response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
			if err != nil {
				t.Fatalf("handleSpecifiedModelRouting: %v", err)
			}
			mutation := response.GetRequestBody().GetResponse().GetHeaderMutation()
			got := setHeaderValues(mutation, "anthropic-beta")
			if tc.profileBeta == "" {
				if len(got) != 0 || !removesHeader(mutation, "anthropic-beta") {
					t.Fatalf("anthropic-beta set %v, removed %v; want removed only", got, mutation.GetRemoveHeaders())
				}
			} else if !reflect.DeepEqual(got, []string{tc.profileBeta}) || removesHeader(mutation, "anthropic-beta") {
				t.Fatalf("anthropic-beta set %v, removed %v; want the profile's only", got, mutation.GetRemoveHeaders())
			}

			profile := router.Config.ProviderProfiles["vertex"]
			dispatch := &providerDispatch{
				logicalModel: model, upstreamModel: "claude-x", backendAddress: "vertex.example.com:443",
				backendName: "vertex", profile: &profile, targetFormat: llmprotocol.AnthropicMessagesV1,
			}
			call, refused := router.buildResendCall(ctx, model, dispatch, []byte(`{"messages":[]}`))
			if refused != "" {
				t.Fatalf("buildResendCall refused: %s", refused)
			}
			if resent := call.headers.Get("anthropic-beta"); resent != tc.profileBeta {
				t.Fatalf("re-send anthropic-beta = %q, want %q", resent, tc.profileBeta)
			}
		})
	}
}

// The Router's own extended-cache-ttl beta, needed for a 1h ttl it
// generated, is still sent, and on an ARC turn a profile's server-side
// fallback beta is not.
func TestVertexAnthropicBetaKeepsTheRoutersCacheBeta(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{"anthropic-beta": "client-beta"}}
	if got := vertexAnthropicBeta("", ctx); got != "" {
		t.Fatalf("no dispatch beta: %q", got)
	}
	ctx.DispatchAutoCache = &dispatchAutoCache{ttl: anthropicLongestCacheTTL}
	if got := vertexAnthropicBeta("own", ctx); got != "own,"+extendedCacheTTLBeta {
		t.Fatalf("with a 1h ttl: %q", got)
	}
	ctx.DispatchAutoCache = nil
	ctx.VSRRaylineARC = &selection.RaylineARCTrace{}
	if got := vertexAnthropicBeta("server-side-fallback-2026-06-01, own", ctx); got != "own" {
		t.Fatalf("ARC turn: %q", got)
	}
}
