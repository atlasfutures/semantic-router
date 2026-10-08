package thinkingcontrol

import "testing"

// The refusal text writes an extra key as Python's repr does, control
// characters escaped, so it matches pathfinder's message byte for byte.
func TestFoldRefusalWritesKeysAsPythonRepr(t *testing.T) {
	cases := map[string]string{
		"name":      "'name'",
		"bad\nkey":  `'bad\nkey'`,
		"it's":      `"it's"`,
		"tab\there": `'tab\there'`,
		"nul\x00":   `'nul\x00'`,
		"ü":         "'ü'",
	}
	for in, want := range cases {
		if got := pythonStrRepr(in); got != want {
			t.Fatalf("pythonStrRepr(%q) = %s, want %s", in, got, want)
		}
	}
	row := objectValue(member{"role", stringValue("system")}, member{"content", stringValue("S")},
		member{"b", intValue(0)}, member{"a", intValue(0)})
	if _, err := foldedUnits(row, FormatChat); err == nil ||
		err.Error() != "an in-band system message carries ['a', 'b']; only its content can be folded" {
		t.Fatalf("refusal = %v", err)
	}
}

// A Responses custom_tool_call_output tail (codex's freeform tools) is a tool
// result: an instruction is inserted after it, as pathfinder places it.
func TestCustomToolCallOutputIsAToolResultTail(t *testing.T) {
	unit := objectValue(member{"type", stringValue("custom_tool_call_output")}, member{"call_id", stringValue("c")},
		member{"output", stringValue("O")})
	placement, err := placementFor(unit, FormatResponses)
	if err != nil || placement != PlacementInsertAfter {
		t.Fatalf("placementFor = %q, %v; want insert_after", placement, err)
	}
}
