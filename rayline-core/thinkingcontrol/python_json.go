package thinkingcontrol

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf16"
)

// Exported Python spellings of a JSON document, for callers that must size
// what a Python chat template renders (rayline-core/execution capacity).
// Values follow this package's json.loads model: a literal with a fraction or
// exponent is a float (spelt by repr), any other an exact integer.

// PythonStrip is Python's str.strip().
func PythonStrip(s string) string { return pythonStrip(s) }

// PythonString is Python's json.dumps(s, ensure_ascii=False).
func PythonString(s string) string {
	var buf bytes.Buffer
	writeString(&buf, s)
	return buf.String()
}

// PythonDumps is json.dumps(json.loads(raw), ensure_ascii=ensureASCII,
// indent=indent or None): ", " and ": " separators without an indent, "," and
// newline-indented items with one.
func PythonDumps(raw []byte, indent int, ensureASCII bool) (string, error) {
	parsed, err := parseJSON(raw)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := writePython(&buf, parsed, indent, 0, ensureASCII); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// PythonMember is one member of a parsed JSON object: its key, its value as
// Python's json.dumps(ensure_ascii=False) spells it, and for a string value
// the string itself.
type PythonMember struct {
	Key      string
	JSON     string
	IsString bool
	Str      string
}

// PythonMembers parses a JSON object (members in order; a repeated key keeps
// its first position and last value).
func PythonMembers(raw []byte) ([]PythonMember, error) {
	parsed, err := parseJSON(raw)
	if err != nil {
		return nil, err
	}
	if parsed.kind != kindObject {
		return nil, fmt.Errorf("not a JSON object")
	}
	members := make([]PythonMember, 0, len(parsed.members))
	for _, m := range parsed.members {
		var buf bytes.Buffer
		if err := writePython(&buf, m.val, 0, 0, false); err != nil {
			return nil, err
		}
		members = append(members, PythonMember{Key: m.key, JSON: buf.String(), IsString: m.val.kind == kindString, Str: m.val.str})
	}
	return members, nil
}

func writePython(buf *bytes.Buffer, v *value, indent, depth int, ensureASCII bool) error {
	text := func(s string) {
		if !ensureASCII {
			writeString(buf, s)
			return
		}
		var plain bytes.Buffer
		writeString(&plain, s)
		for _, r := range plain.String() {
			switch {
			case r < 0x80:
				buf.WriteRune(r)
			case r > 0xFFFF:
				high, low := utf16.EncodeRune(r)
				fmt.Fprintf(buf, `\u%04x\u%04x`, high, low)
			default:
				fmt.Fprintf(buf, `\u%04x`, r)
			}
		}
	}
	newline := func(level int) {
		if indent > 0 {
			buf.WriteByte('\n')
			buf.WriteString(strings.Repeat(" ", indent*level))
		}
	}
	itemSeparator := ", "
	if indent > 0 {
		itemSeparator = ","
	}
	switch v.kind {
	case kindNull:
		buf.WriteString("null")
	case kindBool:
		buf.WriteString(strconv.FormatBool(v.boolean))
	case kindInt:
		buf.WriteString(v.integer.String())
	case kindFloat:
		s, err := pythonFloatRepr(v.float)
		if err != nil {
			return err
		}
		buf.WriteString(s)
	case kindString:
		text(v.str)
	case kindArray:
		buf.WriteByte('[')
		for i, item := range v.items {
			if i > 0 {
				buf.WriteString(itemSeparator)
			}
			newline(depth + 1)
			if err := writePython(buf, item, indent, depth+1, ensureASCII); err != nil {
				return err
			}
		}
		if len(v.items) > 0 {
			newline(depth)
		}
		buf.WriteByte(']')
	case kindObject:
		buf.WriteByte('{')
		for i, m := range v.members {
			if i > 0 {
				buf.WriteString(itemSeparator)
			}
			newline(depth + 1)
			text(m.key)
			buf.WriteString(": ")
			if err := writePython(buf, m.val, indent, depth+1, ensureASCII); err != nil {
				return err
			}
		}
		if len(v.members) > 0 {
			newline(depth)
		}
		buf.WriteByte('}')
	}
	return nil
}
