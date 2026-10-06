package llmprotocol

import (
	"fmt"
	"strings"
	"testing"
)

func TestDescribeToolArgumentsStatesShapeWithoutContent(t *testing.T) {
	for _, test := range []struct {
		arguments string
		want      ToolArgumentsFacts
	}{
		{``, ToolArgumentsFacts{FirstByte: "empty", ValidUTF8: true, JSONErrorOffset: 0}},
		{`run ls`, ToolArgumentsFacts{FirstByte: "other", ValidUTF8: true, JSONErrorOffset: 1}},
		{` {"a":{"b":[1]}}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, MaxDepth: 3}},
		{`{"a":1,"a":2}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, DuplicateKey: true, MaxDepth: 1}},
		{`{"a":{"b":1},"c":{"b":2}}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, MaxDepth: 2}},
		{`{"a":"\ud83d x"}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, LoneSurrogate: true, MaxDepth: 1}},
		{`{"a":"\ud83d` + `\ude00"}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, MaxDepth: 1}},
		{"{\"a\":\"x\ty\"}", ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, JSONErrorOffset: 8, RawControl: true, MaxDepth: 1}},
		{`{"a":"\t\"\\"}`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, MaxDepth: 1}},
		{"{\"a\":\"�\"}", ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, StdlibObject: true, JSONErrorOffset: -1, ReplacementChar: true, MaxDepth: 1}},
		{"{\"a\":\"\xe2\x89\"}", ToolArgumentsFacts{FirstByte: "{", StdlibObject: true, JSONErrorOffset: -1, MaxDepth: 1}},
		{`[1,[2]]`, ToolArgumentsFacts{FirstByte: "[", ValidUTF8: true, JSONErrorOffset: -1, MaxDepth: 2}},
		{`{"a":`, ToolArgumentsFacts{FirstByte: "{", ValidUTF8: true, JSONErrorOffset: 5, MaxDepth: 1}},
	} {
		test.want.Bytes = len(test.arguments)
		if got := DescribeToolArguments([]byte(test.arguments)); got != test.want {
			t.Errorf("DescribeToolArguments(%q) = %+v, want %+v", test.arguments, got, test.want)
		}
	}
}

func TestToolArgumentsFactsErrorCarriesNoContent(t *testing.T) {
	arguments := `{"path":"SECRET-ARGUMENT-TEXT"}`
	facts := DescribeToolArguments([]byte(arguments))
	facts.Stage, facts.Chunks = "item_completed", 2
	for _, text := range []string{facts.Error(), fmt.Sprintf("%+v", facts)} {
		if strings.Contains(text, "SECRET") || strings.Contains(text, "path") {
			t.Fatalf("facts carry argument text: %s", text)
		}
	}
	if facts.Error() != "tool arguments refused at item_completed: 31 bytes in 2 chunks" {
		t.Fatalf("Error() = %q", facts.Error())
	}
}
