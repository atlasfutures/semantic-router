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

package raylinearc

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestPolicyMessageLayoutDescribesWithoutContent(t *testing.T) {
	layout := NewPolicyMessageLayout(json.RawMessage(`{"role":"user","x-client":1,"content":[
		{"type":"text","text":"<system-reminder>a</system-reminder>"},
		{"type":"text","text":"go","cache_control":{"type":"ephemeral"},"x-secret-name":true},
		{"type":"tool_result","tool_use_id":"t","content":"out"},
		{"type":"client-made","text":"z"}]}`))
	want := PolicyMessageLayout{
		Role: "user", Keys: []string{"content", "other", "role"}, ContentForm: "list", BlockCount: 4,
		Blocks: []PolicyBlockLayout{
			{Type: "text", Keys: []string{"text", "type"}, TextLen: 36, TextSHA256: layoutSHA("<system-reminder>a</system-reminder>"), SystemReminder: true},
			{Type: "text", Keys: []string{"cache_control", "other", "text", "type"}, TextLen: 2, TextSHA256: layoutSHA("go"), CacheControl: true},
			{Type: "tool_result", Keys: []string{"content", "tool_use_id", "type"}, TextLen: -1},
			{Type: "other", Keys: []string{"text", "type"}, TextLen: 1, TextSHA256: layoutSHA("z")},
		},
	}
	if !reflect.DeepEqual(layout, want) {
		t.Fatalf("layout = %+v\nwant     %+v", layout, want)
	}
	if text := NewPolicyMessageLayout(json.RawMessage(`{"role":"user","content":"go"}`)); text.ContentForm != "string" ||
		text.BlockCount != 1 || text.Blocks[0].TextSHA256 != layoutSHA("go") || text.Blocks[0].TextLen != 2 {
		t.Fatalf("string content = %+v", text)
	}
	if many := NewPolicyMessageLayout(json.RawMessage(`{"role":"user","a":1,"b":2,"content":[{"type":"text","text":null,"c":1,"d":2}]}`)); !reflect.DeepEqual(many.Keys, []string{"content", "other", "role"}) ||
		!reflect.DeepEqual(many.Blocks[0].Keys, []string{"other", "text", "type"}) || many.Blocks[0].TextLen != -1 || many.Blocks[0].TextSHA256 != "" {
		t.Fatalf("unknown names and null text = %+v", many)
	}
	if null := NewPolicyMessageLayout(json.RawMessage(`{"role":"assistant","content":null}`)); null.ContentForm != "null" || null.BlockCount != 0 {
		t.Fatalf("null content = %+v", null)
	}
	blocks := make([]any, MaxPolicyLayoutBlocks+5)
	for index := range blocks {
		blocks[index] = map[string]any{"type": "tool_result", "content": []any{map[string]any{"type": "text", "text": "x"}}}
	}
	raw, _ := json.Marshal(map[string]any{"role": "user", "content": blocks})
	if long := NewPolicyMessageLayout(raw); long.ContentForm != "list" || long.BlockCount != len(blocks) ||
		len(long.Blocks) != MaxPolicyLayoutBlocks || long.Blocks[0].Type != "tool_result" {
		t.Fatalf("long list = %d blocks of %d", len(long.Blocks), long.BlockCount)
	}
	if bare := NewPolicyMessageLayout(json.RawMessage(`[1]`)); bare.ContentForm != "other" || bare.Role != "" {
		t.Fatalf("not an object = %+v", bare)
	}
}

// A shape keeps the layout of its last messages only.
func TestPolicyPrefixShapeKeepsTheTailLayouts(t *testing.T) {
	messages := make([]json.RawMessage, PolicyPrefixShapeTail+3)
	for index := range messages {
		messages[index] = json.RawMessage(`{"role":"user","content":"m"}`)
	}
	shape := NewPolicyPrefixShape(messages, nil, nil)
	if _, kept := shape.Layout(2); kept {
		t.Fatal("kept a layout before the tail")
	}
	if layout, kept := shape.Layout(len(messages) - 1); !kept || layout.ContentForm != "string" {
		t.Fatalf("last layout = %+v, %v", layout, kept)
	}
	if _, kept := shape.Layout(len(messages)); kept {
		t.Fatal("kept a layout past the end")
	}
}

func layoutSHA(text string) string {
	return textLayout("text", nil, text, true, false).TextSHA256
}
