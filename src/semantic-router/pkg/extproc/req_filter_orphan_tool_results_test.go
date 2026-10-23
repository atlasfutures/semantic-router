package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestFoldOrphanToolResults(t *testing.T) {
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		want   string
		folded int
	}{
		{
			name:   "a summary call's leading result becomes its text",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_gone","content":"file contents"},{"type":"text","text":"Summarize."}]}]}`,
			want: `{"model":"m","messages":[` +
				`{"role":"user","content":[{"type":"text","text":"file contents"},{"type":"text","text":"Summarize."}]}]}`,
			folded: 1,
		},
		{
			name:   "block content is kept and the cache breakpoint moves to its last block",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","is_error":true,` +
				`"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"cache_control":{"type":"ephemeral"}}]}]}`,
			want: `{"messages":[{"role":"user","content":[{"type":"text","text":"a"},` +
				`{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}]}]}`,
			folded: 1,
		},
		{
			name:   "an empty result is stated rather than left empty",
			format: llmprotocol.AnthropicMessagesV1,
			body:   `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x"}]}]}`,
			want:   `{"messages":[{"role":"user","content":[{"type":"text","text":"(empty tool result)"}]}]}`,
			folded: 1,
		},
		{
			name:   "tool references and blank text stay behind; a blank string is empty",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"user","content":[` +
				`{"type":"tool_result","tool_use_id":"a","content":[{"type":"tool_reference","tool_name":"Docs"},{"type":"text","text":" "},{"type":"text","text":"found"}]},` +
				`{"type":"tool_result","tool_use_id":"b","content":"\n"}]}]}`,
			want:   `{"messages":[{"role":"user","content":[{"type":"text","text":"found"},{"type":"text","text":"(empty tool result)"}]}]}`,
			folded: 2,
		},
		{
			name:   "folded content follows the results that stay",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"user","content":"go"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"live","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"gone","content":"old"},{"type":"tool_result","tool_use_id":"live","content":"new"}]}]}`,
			want: `{"messages":[{"role":"user","content":"go"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"live","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"live","content":"new"},{"type":"text","text":"old"}]}]}`,
			folded: 1,
		},
		{
			name:   "a fold that moves past a linked result leaves its cache breakpoint behind",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"live","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"gone","content":"old","cache_control":{"type":"ephemeral","ttl":"1h"}},` +
				`{"type":"text","text":"between"},` +
				`{"type":"tool_result","tool_use_id":"live","content":"new","cache_control":{"type":"ephemeral","ttl":"5m"}},` +
				`{"type":"tool_result","tool_use_id":"late","content":"kept","cache_control":{"type":"ephemeral"}}]}]}`,
			want: `{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"live","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"live","content":"new","cache_control":{"type":"ephemeral","ttl":"5m"}},` +
				`{"type":"text","text":"old"},{"type":"text","text":"between"},` +
				`{"type":"text","text":"kept","cache_control":{"type":"ephemeral"}}]}]}`,
			folded: 2,
		},
		{
			name:   "an escaped block type is still folded",
			format: llmprotocol.AnthropicMessagesV1,
			body:   `{"messages":[{"role":"user","content":[{"type":"tool_\u0072esult","tool_use_id":"x","content":"a"}]}]}`,
			want:   `{"messages":[{"role":"user","content":[{"type":"text","text":"a"}]}]}`,
			folded: 1,
		},
		{
			name:   "a result that answers an earlier call is untouched",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"user","content":"go"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`,
			want: `{"messages":[{"role":"user","content":"go"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`,
		},
		{
			name:   "only the orphan is folded when linked results follow",
			format: llmprotocol.AnthropicMessagesV1,
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"gone","content":"old"}]},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`,
			want: `{"messages":[{"role":"user","content":[{"type":"text","text":"old"}]},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`,
			folded: 1,
		},
		{
			name:   "other wire formats are untouched",
			format: llmprotocol.OpenAIChatV1,
			body:   `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"a"}]}]}`,
			want:   `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"a"}]}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, folded, err := foldOrphanToolResults([]byte(tc.body), tc.format)
			if err != nil {
				t.Fatalf("foldOrphanToolResults() error = %v", err)
			}
			if string(got) != tc.want {
				t.Fatalf("body =\n%s\nwant\n%s", got, tc.want)
			}
			if folded != tc.folded {
				t.Fatalf("folded = %d, want %d", folded, tc.folded)
			}
		})
	}
}

// The body the 2026-10-23 summary calls sent (rt_835c0fd5-3da) is refused by
// the ingress codec as received and decodes once folded.
func TestFoldedOrphanToolResultPassesIngress(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":1024,"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01cut","content":"earlier output"},` +
		`{"type":"text","text":"Summarize this conversation."}]},` +
		`{"role":"assistant","content":[{"type":"text","text":"Working on it."}]},` +
		`{"role":"user","content":[{"type":"text","text":"Go on."}]}]}`)
	engine := protocolcodec.NewBuiltinEngine()
	if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body); err == nil {
		t.Fatal("the unfolded body decoded; the test no longer covers the refusal")
	}
	folded := foldIngressOrphanToolResults(body, llmprotocol.AnthropicMessagesV1, "test")
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, folded)
	if err != nil {
		t.Fatalf("folded body refused: %v", err)
	}
	if got := request.Messages[0].Role; got != llmprotocol.RoleUser {
		t.Fatalf("first message role = %q, want user", got)
	}
	if _, err := raylinearc.ProjectTurns(&request, raylinearc.TurnOptions{}); err != nil {
		t.Fatalf("turn projection refused the folded body: %v", err)
	}
}
