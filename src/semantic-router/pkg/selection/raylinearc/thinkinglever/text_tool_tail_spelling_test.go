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
	"bytes"
	"encoding/json"
	"testing"
)

// semantic-router #248, ADR 0131 (pathfinder #4120): on a worker whose cell
// spells it so (kimi-k3 openrouter x chat), a steer after a text tool run is
// a text part of the run's last tool message, not a user message kimi-k3
// answers as the user's turn. The ledger is the same for every worker; only
// this worker's wire spells it so.

const textToolRun = `{"role":"user","content":"Inspect the repo."},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"ls"}},` +
	`{"type":"tool_use","id":"c2","name":"Bash","input":{"command":"pwd"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"README.md"},` +
	`{"type":"tool_result","tool_use_id":"c2","content":"/work"}]}`

func TestSteerAfterATextToolRunJoinsTheLastToolMessageOnAnAppendToToolCell(t *testing.T) {
	e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none"), spelling: Spelling{AppendToTool: true}}
	plan, provider := e.serve(decodeMessages(t, `[`+textToolRun+`]`), "down")
	if plan.Placement != PlaceUserAfterToolRun || !plan.Emitted {
		t.Fatalf("placement = %q emitted %v, want the steer after the tool run", plan.Placement, plan.Emitted)
	}
	want := `{"role":"tool","tool_call_id":"c2","content":[{"type":"text","text":"/work"},{"type":"text","text":"` + steerDown + `"}]}`
	sent := encodeChatMessages(t, provider)
	if len(sent) != 4 {
		t.Fatalf("sent %d messages, want 4 (no user message of the steer's own): %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[2], `{"role":"tool","tool_call_id":"c1","content":"README.md"}`)
	assertMessageJSON(t, sent[3], want)
	e.commit(plan)

	// Replayed on the next turn, it stays on that tool message.
	next := `[` + textToolRun + `,{"role":"assistant","content":[{"type":"text","text":"Seen."}]},{"role":"user","content":"Thanks."}]`
	_, provider = e.serve(decodeMessages(t, next), "down")
	sent = encodeChatMessages(t, provider)
	if len(sent) != 6 {
		t.Fatalf("sent %d messages on the replay turn, want 6: %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[3], want)
}

// An image run keeps ADR 0129's spelling on the same cell: the steer joins
// the last hoisted image message.
func TestAppendToToolLeavesAnImageRunWithItsHoistedMessage(t *testing.T) {
	image := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + tinyPNG + `"}}`
	history := `{"role":"user","content":"Read the screenshot."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"cat shot.png"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"shot.png"},` + image + `]}]}`
	e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none"), spelling: Spelling{AppendToTool: true}}
	_, provider := e.serve(decodeMessages(t, `[`+history+`]`), "down")
	sent := encodeChatMessages(t, provider)
	if len(sent) != 4 {
		t.Fatalf("sent %d messages, want 4: %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[3], `{"role":"user","content":[{"type":"text","text":"[images returned by tool call c1]"},`+
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,`+tinyPNG+`"}},`+
		`{"type":"text","text":"`+steerDown+`"}]}`)
}

// Only a Chat wire spells it: a Messages worker is sent the same bytes with
// or without the spelling.
func TestAppendToToolDoesNotChangeAMessagesWorker(t *testing.T) {
	var sent [2][]json.RawMessage
	for index, spelling := range []Spelling{{}, {AppendToTool: true}} {
		e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none"), spelling: spelling}
		_, provider := e.serve(decodeMessages(t, `[`+textToolRun+`]`), "down")
		sent[index] = encodeMessages(t, provider)
	}
	left, _ := json.Marshal(sent[0])
	right, _ := json.Marshal(sent[1])
	if !bytes.Equal(left, right) {
		t.Fatalf("the spelling changed a Messages worker's body:\n%s\n%s", left, right)
	}
}
