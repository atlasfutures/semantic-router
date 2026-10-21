package extproc

import (
	"testing"

	"github.com/tidwall/gjson"
)

// A Messages body must not carry a clear_thinking edit without thinking on:
// Anthropic refuses it (prod 2026-10-21, claude-sonnet-5.5 thinking-off arm).
func TestDropClearThinkingWithoutThinking(t *testing.T) {
	const clearThinking = `{"type":"clear_thinking_20251015","keep":"all"}`
	const clearTools = `{"type":"clear_tool_uses_20250919"}`
	tests := []struct {
		name     string
		model    string // "" means anthropic/claude-sonnet-5.5
		thinking string // "" means no thinking member
		edits    string
		want     string // context_management as sent; "" means absent
	}{
		{"non-Claude arm untouched", "z-ai/glm-5.3-flash", "", "[" + clearThinking + "]", `{"edits":[` + clearThinking + `]}`},
		{"bare claude id", "claude-sonnet-5.5", `{"type":"disabled"}`, "[" + clearThinking + "]", ""},
		{"disabled, only edit", "", `{"type":"disabled"}`, "[" + clearThinking + "]", ""},
		{"absent, only edit", "", "", "[" + clearThinking + "]", ""},
		{"disabled, other edits stay", "", `{"type":"disabled"}`, "[" + clearThinking + "," + clearTools + "]", `{"edits":[` + clearTools + `]}`},
		{"adaptive keeps it", "", `{"type":"adaptive"}`, "[" + clearThinking + "]", `{"edits":[` + clearThinking + `]}`},
		{"enabled keeps it", "", `{"type":"enabled","budget_tokens":1024}`, "[" + clearThinking + "]", `{"edits":[` + clearThinking + `]}`},
		{"no clear_thinking untouched", "", `{"type":"disabled"}`, "[" + clearTools + "]", `{"edits":[` + clearTools + `]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := test.model
			if model == "" {
				model = "anthropic/claude-sonnet-5.5"
			}
			body := `{"model":"` + model + `","max_tokens":64,"messages":[{"role":"user","content":"hi"}],`
			if test.thinking != "" {
				body += `"thinking":` + test.thinking + `,`
			}
			body += `"context_management":{"edits":` + test.edits + `}}`
			out, err := dropClearThinkingWithoutThinking([]byte(body))
			if err != nil {
				t.Fatal(err)
			}
			got := gjson.GetBytes(out, "context_management")
			if test.want == "" {
				if got.Exists() {
					t.Fatalf("context_management = %s, want absent", got.Raw)
				}
				return
			}
			if !gjson.Valid(string(out)) || string(compactJSON(t, []byte(got.Raw))) != string(compactJSON(t, []byte(test.want))) {
				t.Fatalf("context_management = %s, want %s", got.Raw, test.want)
			}
			if gjson.GetBytes(out, "messages.0.content").String() != "hi" {
				t.Fatalf("body changed beyond context_management: %s", out)
			}
		})
	}
}
