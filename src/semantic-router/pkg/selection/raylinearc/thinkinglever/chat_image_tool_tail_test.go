/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package thinkinglever

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

func encodeChatMessages(t *testing.T, messages []llmprotocol.Message) []json.RawMessage {
	t.Helper()
	maxTokens := int64(32)
	request := llmprotocol.Request{Generation: 1, Model: "m", Messages: messages}
	request.Sampling.MaxOutputTokens = &maxTokens
	body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode chat messages: %v", err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded messages: %v", err)
	}
	return wire.Messages
}

// semantic-router #238, ADR 0129: a steer after a tool run whose results
// returned images reaches a Chat worker as a text part of the last user
// message carrying those images, never as a user message of its own, which
// kimi-k3 answered as if the user had said it. This is the Hermes-to-kimi
// shape: a Messages client, a Chat worker, the thinking lever.
func TestSteerAfterAnImageToolRunJoinsTheLastHoistedMessageOnChat(t *testing.T) {
	image := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + tinyPNG + `"}}`
	cases := []struct {
		name, history, want string
		messages            int
	}{
		{
			name: "one image result",
			history: `{"role":"user","content":"Read the screenshot."},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"cat shot.png"}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"shot.png"},` + image + `]}]}`,
			want: `{"role":"user","content":[{"type":"text","text":"[images returned by tool call c1]"},` +
				`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + tinyPNG + `"}},` +
				`{"type":"text","text":"` + steerDown + `"}]}`,
			messages: 4,
		},
		{
			name: "mixed run, the image result first",
			history: `{"role":"user","content":"Inspect the render."},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"render"}},` +
				`{"type":"tool_use","id":"c2","name":"Bash","input":{"command":"ls"}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"out.png"},` + image + `]},` +
				`{"type":"tool_result","tool_use_id":"c2","content":"done"}]}`,
			want: `{"role":"user","content":[{"type":"text","text":"[images returned by tool call c1]"},` +
				`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + tinyPNG + `"}},` +
				`{"type":"text","text":"` + steerDown + `"}]}`,
			messages: 5,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none")}
			plan, provider := e.serve(decodeMessages(t, `[`+test.history+`]`), "down")
			if plan.Placement != PlaceUserAfterToolRun {
				t.Fatalf("placement = %q, want the steer after the tool run", plan.Placement)
			}
			sent := encodeChatMessages(t, provider)
			if len(sent) != test.messages {
				t.Fatalf("sent %d messages, want %d (no user message of the steer's own): %s", len(sent), test.messages, sent)
			}
			assertMessageJSON(t, sent[len(sent)-1], test.want)
			e.commit(plan)

			// Replayed on the next turn, it stays inside that message.
			next := `[` + test.history + `,{"role":"assistant","content":[{"type":"text","text":"Seen."}]},{"role":"user","content":"Thanks."}]`
			_, provider = e.serve(decodeMessages(t, next), "down")
			sent = encodeChatMessages(t, provider)
			if len(sent) != test.messages+2 {
				t.Fatalf("sent %d messages on the replay turn, want %d: %s", len(sent), test.messages+2, sent)
			}
			assertMessageJSON(t, sent[test.messages-1], test.want)
		})
	}
}

// Control: with no image in the run, the steer stays a user message of its
// own after the tool messages, as the chat tool_tail golden spells it.
func TestSteerAfterATextToolRunStaysItsOwnMessageOnChat(t *testing.T) {
	history := `{"role":"user","content":"Inspect the repo."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"ls"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"README.md"}]}`
	e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none")}
	_, provider := e.serve(decodeMessages(t, `[`+history+`]`), "down")
	sent := encodeChatMessages(t, provider)
	if len(sent) != 4 {
		t.Fatalf("sent %d messages, want 4: %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[3], `{"role":"user","content":"`+steerDown+`"}`)
}
