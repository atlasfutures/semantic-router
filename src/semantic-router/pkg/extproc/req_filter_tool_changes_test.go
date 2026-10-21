package extproc

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestFoldMidConversationToolChanges(t *testing.T) {
	openRouter := &providerDispatch{
		profile: &config.ProviderProfile{Type: "openrouter"}, targetFormat: llmprotocol.AnthropicMessagesV1,
	}
	cases := []struct {
		name     string
		dispatch *providerDispatch
		body     string
		want     string
	}{
		{
			name:     "an addition joins the tools and leaves the transcript",
			dispatch: openRouter,
			body: `{"model":"m","tools":[{"name":"Bash"}],"messages":[` +
				`{"role":"user","content":[{"type":"text","text":"hi"}]},` +
				`{"role":"system","content":[{"type":"text","text":"note"},` +
				`{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Docs"}},"cache_control":{"type":"ephemeral"}}]}]}`,
			want: `{"model":"m","tools":[{"name":"Bash"},{"name":"Docs"}],"messages":[` +
				`{"role":"user","content":[{"type":"text","text":"hi"}]},` +
				`{"role":"system","content":[{"type":"text","text":"note","cache_control":{"type":"ephemeral"}}]}]}`,
		},
		{
			name:     "a removal drops the named tool and an emptied message",
			dispatch: openRouter,
			body: `{"tools":[{"name":"Bash"},{"name":"Docs"}],"messages":[` +
				`{"role":"user","content":"hi"},` +
				`{"role":"system","content":[{"type":"tool_removal","tool":{"type":"tool_reference","name":"Docs"}}]}]}`,
			want: `{"tools":[{"name":"Bash"}],"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:     "changes apply in transcript order",
			dispatch: openRouter,
			body: `{"messages":[` +
				`{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Docs","v":1}}}]},` +
				`{"role":"user","content":"a"},` +
				`{"role":"system","content":[{"type":"tool_removal","tool":{"type":"tool_reference","name":"Docs"}}]},` +
				`{"role":"user","content":"b"},` +
				`{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Docs","v":2}}}]}]}`,
			want: `{"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}],"tools":[{"name":"Docs","v":2}]}`,
		},
		{
			name:     "removing the last tool drops tools",
			dispatch: openRouter,
			body: `{"tools":[{"name":"Docs"}],"messages":[{"role":"user","content":"hi"},` +
				`{"role":"system","content":[{"type":"tool_removal","tool":{"type":"tool_reference","name":"Docs"}}]}]}`,
			want: `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:     "a reference loads a declared deferred tool",
			dispatch: openRouter,
			body: `{"tools":[{"name":"Docs","defer_loading":true}],"messages":[` +
				`{"role":"system","content":[{"type":"tool_removal","tool":{"type":"tool_reference","name":"Docs"}}]},` +
				`{"role":"user","content":"hi"},` +
				`{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}]}]}`,
			want: `{"tools":[{"name":"Docs"}],"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "a body bound for Anthropic itself keeps the blocks",
			dispatch: &providerDispatch{
				profile: &config.ProviderProfile{Type: "anthropic"}, targetFormat: llmprotocol.AnthropicMessagesV1,
			},
			body: `{"messages":[{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}]}]}`,
			want: `{"messages":[{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}]}]}`,
		},
		{
			name:     "a body without tool changes is untouched",
			dispatch: openRouter,
			body:     `{"tools":[{"name":"Bash"}],"messages":[{"role":"user","content":[{"type":"text","text":"tool_addition"}]}]}`,
			want:     `{"tools":[{"name":"Bash"}],"messages":[{"role":"user","content":[{"type":"text","text":"tool_addition"}]}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := foldMidConversationToolChanges([]byte(tc.body), tc.dispatch, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := compactJSON(t, out), compactJSON(t, []byte(tc.want)); !bytes.Equal(got, want) {
				t.Fatalf("got  %s\nwant %s", got, want)
			}
		})
	}
}

// A model named directly reaches the provider with no decision name; its
// tool changes are folded all the same.
func TestAdaptProviderRequestFoldsToolChangesWithoutDecision(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hi"},` +
		`{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Docs"}}}]}]}`)
	out, err := (&OpenAIRouter{}).adaptProviderRequest(body, &providerDispatch{
		profile: &config.ProviderProfile{Type: "openrouter"}, targetFormat: llmprotocol.AnthropicMessagesV1,
	}, &RequestContext{})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"Docs"}]}`
	if got := compactJSON(t, out); !bytes.Equal(got, compactJSON(t, []byte(want))) {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

// A decision that narrows tools keeps them settled: an addition is dropped,
// never added, while a removal still applies. A decision that sends the
// client's tools as declared admits it.
func TestFoldMidConversationToolChangesKeepsAToolPolicy(t *testing.T) {
	noSelection := false
	cases := []struct {
		name    string
		plugins []config.DecisionPlugin
		want    string
	}{
		{
			name:    "tools mode none",
			plugins: []config.DecisionPlugin{toolsPlugin(t, config.DecisionPluginTools, config.ToolsPluginConfig{Enabled: true, Mode: config.ToolsPluginModeNone})},
			want:    `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "tools mode filtered",
			plugins: []config.DecisionPlugin{toolsPlugin(t, config.DecisionPluginTools, config.ToolsPluginConfig{
				Enabled: true, Mode: config.ToolsPluginModeFiltered, AllowTools: []string{"Docs"},
			})},
			want: `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:    "passthrough with semantic selection",
			plugins: []config.DecisionPlugin{toolsPlugin(t, config.DecisionPluginTools, config.ToolsPluginConfig{Enabled: true})},
			want:    `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:    "a tool_selection plugin",
			plugins: []config.DecisionPlugin{toolsPlugin(t, config.DecisionPluginToolSelection, config.ToolSelectionPluginConfig{Enabled: true})},
			want:    `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "passthrough without selection",
			plugins: []config.DecisionPlugin{toolsPlugin(t, config.DecisionPluginTools, config.ToolsPluginConfig{
				Enabled: true, SemanticSelection: &noSelection,
			})},
			want: `{"tools":[{"name":"Docs"}],"messages":[{"role":"user","content":"hi"}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx :=&RequestContext{VSRSelectedDecision: &config.Decision{Name: "route", Plugins: tc.plugins}}
			body := []byte(`{"tools":[{"name":"Bash"}],"messages":[{"role":"user","content":"hi"},` +
				`{"role":"system","content":[{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Docs"}}},` +
				`{"type":"tool_removal","tool":{"type":"tool_reference","name":"Bash"}}]}]}`)
			out, err := foldMidConversationToolChanges(body, &providerDispatch{
				profile: &config.ProviderProfile{Type: "openrouter"}, targetFormat: llmprotocol.AnthropicMessagesV1,
			}, ctx)
			if err != nil {
				t.Fatal(err)
			}
			if got := compactJSON(t, out); !bytes.Equal(got, compactJSON(t, []byte(tc.want))) {
				t.Fatalf("got  %s\nwant %s", got, tc.want)
			}
		})
	}
}

func toolsPlugin(t *testing.T, kind string, cfg any) config.DecisionPlugin {
	t.Helper()
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return config.DecisionPlugin{Type: kind, Configuration: payload}
}
