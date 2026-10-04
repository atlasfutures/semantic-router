package llmprotocol

import (
	"strings"
	"testing"
)

func TestTruncatedJSONObjectAcceptsOnlyAStrictPrefixOfAnObject(t *testing.T) {
	depth := DefaultPolicy().Limits.JSONDepth
	for _, prefix := range []string{
		`{`, `{"a`, `{"a":`, `{"a":"x`, `{"a":[1,`, `{"a":{"b":`,
		`{"a":1`, `{"a":tr`, `{"a":"x\u00`, `{"a":{},"b":[{"c":null}`, ` {"a": "b", `,
		// A surrogate pair cut between its halves.
		`{"a":"\ud83d`, `{"a":"\ud83d\u`, `{"a":"\ud83d\udc`,
	} {
		if !TruncatedJSONObject([]byte(prefix), depth) {
			t.Errorf("prefix %q was refused", prefix)
		}
	}
	for _, refused := range []string{
		// No appended bytes make these valid.
		`{]`, `{"x":]`, `{"a" "b"`, `{,`, `{"a":1,,`, `{"a":tx`, `{1:`,
		// A member repeated, as the strict validator refuses.
		`{"a":1,"a":`,
		// A lone surrogate, which the strict validator refuses.
		`{"a":"\ud83dx`, `{"a":"\ud83d\u0041`, `{"a":"\udc00`,
		// Whole values: an object is not truncated; the rest are not objects.
		`{}`, `{"a":1}`, `{"a":1}}`, `[]`, `true`, `"x"`, `[1, 2`, `run ls`, ``,
	} {
		if TruncatedJSONObject([]byte(refused), depth) {
			t.Errorf("%q was accepted as a truncated object", refused)
		}
	}
}

func TestTruncatedJSONObjectKeepsTheDepthBound(t *testing.T) {
	const limit = 3
	// The top object's members sit at depth 1, as ValidateJSONObject counts.
	within := `{"a":` + strings.Repeat(`[`, limit)
	if !TruncatedJSONObject([]byte(within), limit) {
		t.Fatalf("a prefix nested to the limit was refused: %s", within)
	}
	deeper := `{"a":` + strings.Repeat(`[`, limit+1)
	if TruncatedJSONObject([]byte(deeper), limit) {
		t.Fatalf("a prefix nested past the limit was accepted: %s", deeper)
	}
	if ValidateJSONObject([]byte(within+strings.Repeat(`]`, limit)+`}`), limit) != nil {
		t.Fatal("the depth counted here differs from ValidateJSONObject's")
	}
	if ValidateJSONObject([]byte(deeper+strings.Repeat(`]`, limit+1)+`}`), limit) == nil {
		t.Fatal("the depth counted here differs from ValidateJSONObject's")
	}
}

// The depth bound applies as soon as a member of a too-deep container
// begins. Each prefix is paired with a completion that shows which way
// ValidateJSONObject decides at depth 1: a container at the deepest legal
// depth can still close empty, but no key or element can enter it.
func TestTruncatedJSONObjectRefusesAMemberBeyondTheDepthLimit(t *testing.T) {
	for _, test := range []struct {
		prefix, completion string
		atDepthOne         bool
	}{
		{`{"x":{"y":`, `1}}`, false},
		{`{"x":{"y"`, `:1}}`, false},
		{`{"x":{"y`, `":1}}`, false},
		{`{"x":["a`, `"]}`, false},
		{`{"x":[1,`, `2]}`, false},
		{`{"x":{`, `}}`, true},
		{`{"x":[`, `]}`, true},
		{`{"x": { `, `}}`, true},
	} {
		whole := ValidateJSONObject([]byte(test.prefix+test.completion), 1) == nil
		if whole != test.atDepthOne {
			t.Fatalf("%s%s at depth 1: ValidateJSONObject says %v, the case says %v",
				test.prefix, test.completion, whole, test.atDepthOne)
		}
		if got := TruncatedJSONObject([]byte(test.prefix), 1); got != test.atDepthOne {
			t.Errorf("%q at depth 1: accepted = %v, want %v", test.prefix, got, test.atDepthOne)
		}
		if !TruncatedJSONObject([]byte(test.prefix), 2) {
			t.Errorf("%q at depth 2 was refused", test.prefix)
		}
	}
}
