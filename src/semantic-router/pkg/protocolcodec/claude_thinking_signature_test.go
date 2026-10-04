package protocolcodec

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const claudeDetail = `{"type":"reasoning.text","text":"Check the tool.","signature":"sig-claude","format":"anthropic-claude-v1","index":0}`

func chatResponseWithReasoning(reasoning, details string) string {
	return `{"id":"gen-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"Done.","reasoning":` + reasoning + `,"reasoning_details":` + details + `}}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
}

// Claude served over Chat: the anthropic-claude-v1 signature OpenRouter
// returns in reasoning_details becomes the Messages thinking block's
// signature, so a Messages client replays Claude's thinking signed. Only one
// signed Claude item whose text is the block's text maps; any other shape
// leaves the block unsigned, since a signature covers one block's text.
func TestClaudeThinkingSignatureReachesAMessagesClient(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, reasoning, details, want string
	}{
		{"one signed Claude item", `"Check the tool."`, "[" + claudeDetail + "]", "sig-claude"},
		{"another provider's item", `"Check the tool."`, `[{"type":"reasoning.text","text":"Check the tool.","signature":"x","format":"unknown","index":0}]`, ""},
		{"no signature", `"Check the tool."`, `[{"type":"reasoning.text","text":"Check the tool.","format":"anthropic-claude-v1","index":0}]`, ""},
		{"text that is not the item's", `"Check the tool. And more."`, "[" + claudeDetail + "]", ""},
		{"two items", `"Check the tool.Again."`, "[" + claudeDetail + `,{"type":"reasoning.text","text":"Again.","signature":"sig-2","format":"anthropic-claude-v1","index":1}]`, ""},
		{"mixed formats", `"Check the tool."`, "[" + claudeDetail + `,{"type":"reasoning.encrypted","data":"g","format":"google-gemini-v1","index":1}]`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
				[]byte(chatResponseWithReasoning(tc.reasoning, tc.details)), nil)
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Content []struct {
					Type      string  `json:"type"`
					Signature *string `json:"signature"`
				} `json:"content"`
			}
			if err := json.Unmarshal(result.Body, &body); err != nil || len(body.Content) == 0 || body.Content[0].Type != "thinking" {
				t.Fatalf("body = %s", result.Body)
			}
			got := ""
			if body.Content[0].Signature != nil {
				got = *body.Content[0].Signature
			}
			if got != tc.want {
				t.Fatalf("thinking signature = %q, want %q: %s", got, tc.want, result.Body)
			}
		})
	}
}

// The stream carries the same signature as a signature_delta before the
// thinking block stops, where Anthropic sends it, and only when the block
// held one signed Claude item.
func TestClaudeThinkingSignatureStreamsAsASignatureDelta(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	chunk := func(delta string) string {
		return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}` + "\n\n"
	}
	done := `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"
	text := func(s, format string) string {
		return chunk(`{"role":"assistant","content":"","reasoning":"` + s + `","reasoning_details":[{"type":"reasoning.text","text":"` + s + `","format":"` + format + `","index":0}]}`)
	}
	signature := func(format string) string {
		return chunk(`{"content":"","reasoning_details":[{"type":"reasoning.text","signature":"sig-claude","format":"` + format + `","index":0}]}`)
	}
	answer := chunk(`{"content":"Done."}`)
	for _, tc := range []struct {
		name   string
		chunks []string
		signed bool
	}{
		{"one signed Claude item", []string{text("Check ", "anthropic-claude-v1"), text("the tool.", "anthropic-claude-v1"), signature("anthropic-claude-v1"), answer, done}, true},
		{"another provider's item", []string{text("Check the tool.", "unknown"), signature("unknown"), answer, done}, false},
		{"no signature", []string{text("Check the tool.", "anthropic-claude-v1"), answer, done}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public", ProviderModel: "m",
			})
			if err != nil {
				t.Fatal(err)
			}
			var wire strings.Builder
			for _, c := range tc.chunks {
				frames, _, _, err := stream.Push([]byte(c))
				if err != nil {
					t.Fatal(err)
				}
				for _, frame := range frames {
					wire.Write(frame)
				}
			}
			frames, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, frame := range frames {
				wire.Write(frame)
			}
			out := wire.String()
			signedAt := strings.Index(out, `"signature_delta"`)
			if !tc.signed {
				if signedAt >= 0 || strings.Contains(out, "sig-claude") {
					t.Fatalf("a signature reached the client: %s", out)
				}
				return
			}
			stopAt := strings.Index(out, `"content_block_stop"`)
			if signedAt < 0 || !strings.Contains(out, `"signature":"sig-claude"`) || stopAt < signedAt ||
				strings.Index(out, `"thinking":"the tool."`) > signedAt {
				t.Fatalf("no signature_delta between the thinking and its block stop: %s", out)
			}
		})
	}
}

// Interleaved thinking: an assistant turn with several signed thinking blocks
// reaches a Claude worker over Chat as one reasoning_details item per block,
// numbered in order within the message. OpenRouter keys items by index, so a
// repeated index would merge or misattribute signatures.
func TestSignedThinkingAsReasoningDetailsNumbersItemsPerMessage(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"model":"m","max_tokens":64,"messages":[`+
		`{"role":"user","content":"q"},`+
		`{"role":"assistant","content":[{"type":"thinking","thinking":"first","signature":"sig-1"},`+
		`{"type":"tool_use","id":"toolu_1","name":"bash","input":{}},`+
		`{"type":"thinking","thinking":"second","signature":"sig-2"},`+
		`{"type":"tool_use","id":"toolu_2","name":"bash","input":{}}]},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"a"},{"type":"tool_result","tool_use_id":"toolu_2","content":"b"}]},`+
		`{"role":"assistant","content":[{"type":"thinking","thinking":"third","signature":"sig-3"},{"type":"text","text":"done"}]},`+
		`{"role":"user","content":"q2"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if rewritten := SignedThinkingAsReasoningDetails(&request); rewritten != 3 {
		t.Fatalf("rewritten = %d, want 3", rewritten)
	}
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Messages []struct {
			Role             string `json:"role"`
			ReasoningDetails []struct {
				Index     int    `json:"index"`
				Signature string `json:"signature"`
				Text      string `json:"text"`
			} `json:"reasoning_details"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &body); err != nil {
		t.Fatal(err)
	}
	var got [][3]string
	for _, message := range body.Messages {
		for _, item := range message.ReasoningDetails {
			got = append(got, [3]string{fmt.Sprint(item.Index), item.Signature, item.Text})
		}
	}
	want := [][3]string{{"0", "sig-1", "first"}, {"1", "sig-2", "second"}, {"0", "sig-3", "third"}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("reasoning_details items = %v, want %v\n%s", got, want, result.Body)
	}
}
