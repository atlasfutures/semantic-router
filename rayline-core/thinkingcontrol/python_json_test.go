package thinkingcontrol

import "testing"

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
