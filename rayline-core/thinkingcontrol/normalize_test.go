package thinkingcontrol

import (
	"bytes"
	"strings"
	"testing"
)

func TestStandaloneFoldUsesExistingRuleWithoutSteering(t *testing.T) {
	cases := []struct{ name, body, want string }{
		{"user", `{"model":"m","messages":[{"role":"system","content":"leading"},{"role":"user","content":"question"},{"role":"system","content":[{"type":"text","text":"later ü","cache_control":{"type":"ephemeral"}}]}]}`,
			`{"model":"m","messages":[{"role":"system","content":"leading"},{"role":"user","content":[{"type":"text","text":"question"},{"type":"text","text":"later ü","cache_control":{"type":"ephemeral"}}]}]}`},
		{"tool", `{"messages":[{"role":"user","content":"question"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"result"},{"role":"system","content":"after result"}]}`,
			`{"messages":[{"role":"user","content":"question"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"result"},{"role":"user","content":[{"type":"text","text":"after result"}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := FoldInbandSystem([]byte(tc.body), FormatChat)
			if err != nil || string(got) != tc.want {
				t.Fatalf("fold = %s, %v", got, err)
			}
			again, err := FoldInbandSystem(got, FormatChat)
			if err != nil || !bytes.Equal(got, again) {
				t.Fatal("fold was not idempotent")
			}
		})
	}
}

func TestStandaloneFoldPreservesExistingRefusals(t *testing.T) {
	for name, body := range map[string]string{
		"answered":    `{"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"},{"role":"system","content":"later"}]}`,
		"split tools": `{"messages":[{"role":"tool","tool_call_id":"a","content":"a"},{"role":"system","content":"later"},{"role":"tool","tool_call_id":"b","content":"b"}]}`,
		"extra key":   `{"messages":[{"role":"user","content":"question"},{"role":"system","content":"later","name":"must not disappear"}]}`,
		"nontext":     `{"messages":[{"role":"user","content":"question"},{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := FoldInbandSystem([]byte(body), FormatChat); err == nil || !strings.Contains(err.Error(), "in-band system") {
				t.Fatalf("unsupported fold accepted: %v", err)
			}
		})
	}
}
