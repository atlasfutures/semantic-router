package thinkingcontrol

import (
	"strings"
	"testing"
)

func TestPythonDumpsMatchesPythonJSON(t *testing.T) {
	raw := []byte(`{"a":1e15,"b":[1,2.5,{"c":"é\n"}],"d":{},"e":[],"f":12345678901234567890,"g":1E-7}`)
	for _, tc := range []struct {
		indent int
		ascii  bool
		want   string
	}{
		// python3 -c 'import json; json.dumps(json.loads(s), ensure_ascii=False)'
		{0, false, `{"a": 1000000000000000.0, "b": [1, 2.5, {"c": "é\n"}], "d": {}, "e": [], "f": 12345678901234567890, "g": 1e-07}`},
		// json.dumps(json.loads(s), indent=4)
		{4, true, "{\n    \"a\": 1000000000000000.0,\n    \"b\": [\n        1,\n        2.5,\n        {\n            \"c\": \"\\u00e9\\n\"\n        }\n    ],\n    \"d\": {},\n    \"e\": [],\n    \"f\": 12345678901234567890,\n    \"g\": 1e-07\n}"},
	} {
		got, err := PythonDumps(raw, tc.indent, tc.ascii)
		if err != nil || got != tc.want {
			t.Fatalf("indent %d ascii %v:\n%s\nwant\n%s (%v)", tc.indent, tc.ascii, got, tc.want, err)
		}
	}
	members, err := PythonMembers([]byte(`{"path":"/x","n":2.0,"o":{"k":[1]}}`))
	if err != nil || len(members) != 3 || !members[0].IsString || members[0].Str != "/x" || members[1].JSON != "2.0" || members[2].JSON != `{"k": [1]}` {
		t.Fatalf("members %+v %v", members, err)
	}
}

// TestWorstCaseBytesChargesNormalization: NFKC spells U+FDFA (one 6-byte
// escape) in 33 UTF-8 bytes and NFD a Hangul syllable in 9, so each charges
// the difference, in keys, in strings and inside reloaded arguments, from the
// same parse as the spelling (an integer too large for a float64 included).
func TestWorstCaseBytesChargesNormalization(t *testing.T) {
	if got := normalizationSlack("\ufdfa"); got != 27 {
		t.Errorf("U+FDFA slack %d, want 27", got)
	}
	if got := normalizationSlack("\ud55c"); got != 3 {
		t.Errorf("Hangul slack %d, want 3", got)
	}
	if got := normalizationSlack("plain é"); got != 0 {
		t.Errorf("composed Latin slack %d, want 0", got)
	}
	big := strings.Repeat("9", 400)
	for name, test := range map[string]struct {
		plain, expanding string
		slack            int64
	}{
		"string":    {`{"n":` + big + `,"s":"a"}`, `{"n":` + big + `,"s":"\ufdfa"}`, 27},
		"key":       {`{"a":1}`, `{"\ufdfa":1}`, 27},
		"arguments": {`{"arguments":"{\"s\":\"a\"}"}`, `{"arguments":"{\"s\":\"\\ufdfa\"}"}`, 27},
	} {
		plain, err := PythonWorstCaseBytes([]byte(test.plain), "arguments")
		if err != nil {
			t.Fatal(name, err)
		}
		expanding, err := PythonWorstCaseBytes([]byte(test.expanding), "arguments")
		if err != nil {
			t.Fatal(name, err)
		}
		// Each escape replaces one ASCII byte with six: \u spelling growth.
		spelling := int64(5)
		if name == "arguments" {
			// The outer string spells the escape's backslash twice (\\ufdfa,
			// 7 bytes for 1); the reloaded value spells \ufdfa (6 for 1).
			spelling = 6 + 5
		}
		if got := expanding - plain - spelling; got != test.slack {
			t.Errorf("%s: slack %d, want %d", name, got, test.slack)
		}
	}
}
