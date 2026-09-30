package extproc

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Codex's web_search reaches an OpenRouter Chat arm as the web plugin only
// when the model is admitted with it and the declaration is unrestricted.
func TestOpenRouterWebPluginNeedsAdmissionAndAnUnrestrictedDeclaration(t *testing.T) {
	openRouter := &config.ProviderProfile{Type: "openrouter"}
	declaration := func(raw string) *llmprotocol.Request {
		return &llmprotocol.Request{CarriedTools: []llmprotocol.UnmodeledBlock{{
			Format: llmprotocol.OpenAIResponsesV1, Type: "web_search", Raw: []byte(raw),
		}}}
	}
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"q"}]}`)
	for name, tc := range map[string]struct {
		profile *config.ProviderProfile
		target  llmprotocol.WireFormat
		hosted  []string
		request *llmprotocol.Request
		want    bool
	}{
		"admitted and unrestricted": {openRouter, llmprotocol.OpenAIChatV1, []string{"web_search"}, declaration(`{"type":"web_search"}`), true},
		"not admitted":              {openRouter, llmprotocol.OpenAIChatV1, nil, declaration(`{"type":"web_search"}`), false},
		"restricted (Codex default)": {openRouter, llmprotocol.OpenAIChatV1, []string{"web_search"},
			declaration(`{"type":"web_search","external_web_access":false}`), false},
		"not OpenRouter": {&config.ProviderProfile{Type: "openai"}, llmprotocol.OpenAIChatV1, []string{"web_search"}, declaration(`{"type":"web_search"}`), false},
		"no declaration": {openRouter, llmprotocol.OpenAIChatV1, []string{"web_search"}, &llmprotocol.Request{}, false},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SemanticRequest: tc.request, DispatchHostedTools: tc.hosted}
			out, err := applyOpenRouterWebSearch(body, &providerDispatch{profile: tc.profile, targetFormat: tc.target}, ctx)
			if err != nil {
				t.Fatal(err)
			}
			if got := bytes.Contains(out, []byte(`"plugins":[{"id":"web"}]`)); got != tc.want {
				t.Fatalf("web plugin = %v, want %v: %s", got, tc.want, out)
			}
		})
	}
}

// Through encode and the provider adapter, an admitted Codex search on an
// OpenRouter Chat arm leaves the body with the web plugin and the record with
// no drop of the search.
func TestOpenRouterWebSearchDispatchFlow(t *testing.T) {
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":"what shipped in go 1.30?","tools":[{"type":"web_search"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "provider-model"
	request.Generation++
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &request, ProtocolEnvelope: envelope, DispatchHostedTools: []string{"web_search"},
	}
	router := &OpenAIRouter{}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	body, err = router.adaptProviderRequest(body, &providerDispatch{
		profile: &config.ProviderProfile{Type: "openrouter"}, targetFormat: llmprotocol.OpenAIChatV1,
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"plugins":[{"id":"web"}]`)) {
		t.Fatalf("the web plugin was not sent: %s", body)
	}
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == "tools.web_search" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			t.Fatalf("a search that was sent is recorded as dropped: %+v", ctx.ProtocolDiagnostics)
		}
	}
}
