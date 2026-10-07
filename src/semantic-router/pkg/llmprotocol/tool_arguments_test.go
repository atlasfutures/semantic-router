package llmprotocol

import (
	"strings"
	"testing"
)

func TestNormalizeToolArgumentsSettlesWhatAnthropicAccepts(t *testing.T) {
	depth := DefaultPolicy().Limits.JSONDepth
	for arguments, want := range map[string]string{
		// Strict arguments come back byte for byte.
		` {"a": 1, "b": [true, null]} `: ` {"a": 1, "b": [true, null]} `,
		`{"a":"\ud83d` + `\ude00"}`:     `{"a":"\ud83d` + `\ude00"}`,
		// A repeated member keeps its last value in its first place.
		`{"path":"a.py","edits":[],"path":"b.py"}`: `{"path":"b.py","edits":[]}`,
		`{"a":{"x":1,"x":{"y":2}}, "b": 1.50e3}`:   `{"a":{"x":{"y":2}},"b":1.50e3}`,
		// An unpaired surrogate escape becomes U+FFFD; nothing else changes.
		`{"t":"x \ud83d <&> \n y"}`: "{\"t\":\"x � <&> \\n y\"}",
		`{"t":["\ude00"]}`:          "{\"t\":[\"�\"]}",
	} {
		got, err := NormalizeToolArguments([]byte(arguments), depth)
		if err != nil || string(got) != want {
			t.Errorf("NormalizeToolArguments(%s) = %s, %v; want %s", arguments, got, err, want)
			continue
		}
		if ValidateJSONObject(got, depth) != nil {
			t.Errorf("NormalizeToolArguments(%s) = %s is not strict", arguments, got)
		}
	}
}

func TestNormalizeToolArgumentsRelaxesNothingElse(t *testing.T) {
	for _, arguments := range []string{
		"{\"a\":\"x\ny\",\"a\":1}", "{\"a\":\"\xe2\x89\",\"a\":1}", `[{"a":1,"a":2}]`, `"x"`, `null`,
		`{"a":1,"a":2} {}`, `{"a":1,"a":2`, `{"a":"\ud83d`, ``,
		// Nested past the limit, as ValidateJSONObject counts it.
		`{"a":1,"a":` + strings.Repeat("[", 5) + strings.Repeat("]", 5) + `}`,
	} {
		if got, err := NormalizeToolArguments([]byte(arguments), 4); err == nil {
			t.Errorf("NormalizeToolArguments(%q) = %s, want a refusal", arguments, got)
		}
	}
}
