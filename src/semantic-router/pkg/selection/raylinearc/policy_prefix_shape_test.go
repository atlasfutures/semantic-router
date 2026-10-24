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
	"testing"
)

func shapeMessages(texts ...string) []json.RawMessage {
	messages := make([]json.RawMessage, len(texts))
	for index, text := range texts {
		messages[index], _ = json.Marshal(map[string]any{"role": "user", "content": text})
	}
	return messages
}

func TestPolicyPrefixShapeFirstDivergence(t *testing.T) {
	recorded := NewPolicyPrefixShape(shapeMessages("a", "b", "c"), nil, nil)
	for _, c := range []struct {
		name    string
		current []string
		index   int
	}{
		{"extends", []string{"a", "b", "c", "d"}, 3},
		{"changed", []string{"a", "x", "c"}, 1},
		{"shrank", []string{"a", "b"}, 2},
		{"rewritten", []string{"z"}, 0},
	} {
		index, ok := recorded.FirstDivergence(NewPolicyPrefixShape(shapeMessages(c.current...), nil, nil))
		if !ok || index != c.index {
			t.Errorf("%s = %d %v, want %d", c.name, index, ok, c.index)
		}
	}
	// A cache_control move is not a divergence, as PolicyTurn reads it.
	moved, _ := json.Marshal(map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": "a", "cache_control": map[string]any{"type": "ephemeral"}},
	}})
	plain, _ := json.Marshal(map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "a"}}})
	if index, _ := NewPolicyPrefixShape([]json.RawMessage{plain}, nil, nil).FirstDivergence(
		NewPolicyPrefixShape([]json.RawMessage{moved}, nil, nil)); index != 1 {
		t.Fatalf("cache_control move diverged at %d", index)
	}
	long := make([]string, MaxPolicyPrefixShapeMessages+2)
	changed := append([]string(nil), long...)
	changed[MaxPolicyPrefixShapeMessages+1] = "x"
	if _, ok := NewPolicyPrefixShape(shapeMessages(long...), nil, nil).FirstDivergence(
		NewPolicyPrefixShape(shapeMessages(changed...), nil, nil)); ok {
		t.Fatal("a divergence past the kept digests was reported known")
	}
}

func TestPolicyPrefixShapeRecordsTheCommittedPrefix(t *testing.T) {
	messages := shapeMessages("a", "b")
	state := (&PolicyEpisodeState{}).Next(messages, "action", "arm")
	if !NewPolicyPrefixShape(messages, nil, nil).Records(state) {
		t.Fatal("the committed request's shape does not record its prefix")
	}
	if NewPolicyPrefixShape(shapeMessages("a"), nil, nil).Records(state) {
		t.Fatal("a shorter request's shape records the prefix")
	}
}

func TestPolicyMessageKind(t *testing.T) {
	for _, c := range []struct {
		message, role, block string
	}{
		{`{"role":"user","content":"hi"}`, "user", "text"},
		{`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t"}]}`, "user", "tool_result"},
		{`{"role":"assistant","content":[]}`, "assistant", ""},
		{`{"type":"function_call_output","call_id":"c","output":"x"}`, "function_call_output", ""},
		{`{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}`, "user", "input_text"},
		{`not json`, "", ""},
		{`{"role":"PRIVATE_ROLE","content":[{"type":"PRIVATE_BLOCK"}]}`, "other", "other"},
		{`{"type":"PRIVATE_ITEM","content":"x"}`, "other", "text"},
	} {
		role, block := PolicyMessageKind(json.RawMessage(c.message))
		if role != c.role || block != c.block {
			t.Errorf("%s = %q/%q, want %q/%q", c.message, role, block, c.role, c.block)
		}
	}
}
