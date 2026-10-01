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

package thinkingcontrol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// kind is a JSON value's type.
type kind int

const (
	kindNull kind = iota
	kindBool
	kindInt
	kindFloat
	kindString
	kindArray
	kindObject
)

// value is one JSON value with its object members in the order the client
// sent them, so a rendered body keeps the client's key order and puts
// inserted keys last, as pathfinder's renderer (a Python dict) does.
type value struct {
	kind    kind
	boolean bool
	integer *big.Int
	float   float64
	str     string
	items   []*value
	members []member
}

type member struct {
	key string
	val *value
}

func nullValue() *value          { return &value{kind: kindNull} }
func stringValue(s string) *value { return &value{kind: kindString, str: s} }
func intValue(n int64) *value     { return &value{kind: kindInt, integer: big.NewInt(n)} }
func arrayValue(items ...*value) *value {
	return &value{kind: kindArray, items: append([]*value{}, items...)}
}

func objectValue(members ...member) *value {
	obj := &value{kind: kindObject}
	for _, m := range members {
		obj.set(m.key, m.val)
	}
	return obj
}

// get returns an object member, or nil when the value is not an object or
// has no such member.
func (v *value) get(key string) *value {
	if v == nil || v.kind != kindObject {
		return nil
	}
	for _, m := range v.members {
		if m.key == key {
			return m.val
		}
	}
	return nil
}

func (v *value) has(key string) bool {
	if v == nil || v.kind != kindObject {
		return false
	}
	for _, m := range v.members {
		if m.key == key {
			return true
		}
	}
	return false
}

// set replaces a member in place or appends a new one, as a Python dict does.
func (v *value) set(key string, val *value) {
	for i, m := range v.members {
		if m.key == key {
			v.members[i].val = val
			return
		}
	}
	v.members = append(v.members, member{key: key, val: val})
}

// remove deletes a member and returns it, or nil.
func (v *value) remove(key string) *value {
	if v == nil || v.kind != kindObject {
		return nil
	}
	for i, m := range v.members {
		if m.key == key {
			v.members = append(v.members[:i], v.members[i+1:]...)
			return m.val
		}
	}
	return nil
}

func (v *value) isString() bool { return v != nil && v.kind == kindString }

// clone is a deep copy.
func (v *value) clone() *value {
	if v == nil {
		return nil
	}
	out := *v
	if v.integer != nil {
		out.integer = new(big.Int).Set(v.integer)
	}
	if v.items != nil {
		out.items = make([]*value, len(v.items))
		for i, item := range v.items {
			out.items[i] = item.clone()
		}
	}
	if v.members != nil {
		out.members = make([]member, len(v.members))
		for i, m := range v.members {
			out.members[i] = member{key: m.key, val: m.val.clone()}
		}
	}
	return &out
}

// equal compares two values the way Python compares the parsed objects:
// members by key regardless of order, numbers by value.
func (v *value) equal(other *value) bool {
	if v == nil || other == nil {
		return v == other
	}
	if v.kind == kindInt && other.kind == kindFloat || v.kind == kindFloat && other.kind == kindInt {
		return v.asFloat() == other.asFloat()
	}
	if v.kind != other.kind {
		return false
	}
	switch v.kind {
	case kindNull:
		return true
	case kindBool:
		return v.boolean == other.boolean
	case kindInt:
		return v.integer.Cmp(other.integer) == 0
	case kindFloat:
		return v.float == other.float
	case kindString:
		return v.str == other.str
	case kindArray:
		if len(v.items) != len(other.items) {
			return false
		}
		for i := range v.items {
			if !v.items[i].equal(other.items[i]) {
				return false
			}
		}
		return true
	default:
		if len(v.members) != len(other.members) {
			return false
		}
		for _, m := range v.members {
			if !other.has(m.key) || !m.val.equal(other.get(m.key)) {
				return false
			}
		}
		return true
	}
}

func (v *value) asFloat() float64 {
	if v.kind == kindFloat {
		return v.float
	}
	f, _ := new(big.Float).SetInt(v.integer).Float64()
	return f
}

var errTrailingData = errors.New("trailing data after the JSON value")

// parseJSON reads one JSON document. Numbers follow Python's json.loads: a
// literal with a fraction or exponent is a float, any other an exact integer.
// A repeated key keeps its first position and its last value.
func parseJSON(raw []byte) (*value, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := parseValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errTrailingData
	}
	return v, nil
}

func parseValue(dec *json.Decoder) (*value, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case nil:
		return nullValue(), nil
	case bool:
		return &value{kind: kindBool, boolean: t}, nil
	case string:
		return stringValue(t), nil
	case json.Number:
		return parseNumber(string(t))
	case json.Delim:
		switch t {
		case '[':
			arr := &value{kind: kindArray, items: []*value{}}
			for dec.More() {
				item, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				arr.items = append(arr.items, item)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return arr, nil
		case '{':
			obj := &value{kind: kindObject, members: []member{}}
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := keyTok.(string)
				if !ok {
					return nil, fmt.Errorf("object key is not a string")
				}
				val, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				obj.set(key, val)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return obj, nil
		}
	}
	return nil, fmt.Errorf("unexpected JSON token %v", tok)
}

func parseNumber(literal string) (*value, error) {
	if strings.ContainsAny(literal, ".eE") {
		f, err := strconv.ParseFloat(literal, 64)
		if err != nil {
			return nil, fmt.Errorf("number %q: %w", literal, err)
		}
		return &value{kind: kindFloat, float: f}, nil
	}
	n, ok := new(big.Int).SetString(literal, 10)
	if !ok {
		return nil, fmt.Errorf("number %q is not an integer", literal)
	}
	return &value{kind: kindInt, integer: n}, nil
}

// dump is the provider body: compact JSON, UTF-8 unescaped, object members
// in their order. It matches Python's
// json.dumps(ensure_ascii=False, separators=(",", ":"), allow_nan=False).
func dump(v *value) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeValue(&buf, v, false); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// dumpSorted is dump with object keys sorted by code point, Python's
// sort_keys=True; the anchor digest hashes it.
func dumpSorted(v *value) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeValue(&buf, v, true); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeValue(buf *bytes.Buffer, v *value, sortKeys bool) error {
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
		writeString(buf, v.str)
	case kindArray:
		buf.WriteByte('[')
		for i, item := range v.items {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeValue(buf, item, sortKeys); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case kindObject:
		members := v.members
		if sortKeys {
			members = append([]member{}, members...)
			sort.SliceStable(members, func(i, j int) bool { return members[i].key < members[j].key })
		}
		buf.WriteByte('{')
		for i, m := range members {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeString(buf, m.key)
			buf.WriteByte(':')
			if err := writeValue(buf, m.val, sortKeys); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	}
	return nil
}

// writeString escapes as Python's json.dumps(ensure_ascii=False) and RFC 8785
// both do: the quote, the backslash, the five short control escapes, and any
// other control character as a lowercase \u00XX.
func writeString(buf *bytes.Buffer, s string) {
	const hex = "0123456789abcdef"
	buf.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			buf.WriteString(`\"`)
		case '\\':
			buf.WriteString(`\\`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		case '\b':
			buf.WriteString(`\b`)
		case '\f':
			buf.WriteString(`\f`)
		default:
			if r < 0x20 {
				buf.WriteString(`\u00`)
				buf.WriteByte(hex[r>>4])
				buf.WriteByte(hex[r&0xf])
			} else {
				buf.WriteRune(r)
			}
		}
	}
	buf.WriteByte('"')
}

// pythonFloatRepr is Python's repr(float): the shortest round-trip digits,
// positional between 1e-4 and 1e16 with at least one fractional digit,
// otherwise scientific with a two-digit signed exponent.
func pythonFloatRepr(f float64) (string, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "", fmt.Errorf("non-finite number %v", f)
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0", nil
		}
		return "0.0", nil
	}
	sci := strconv.FormatFloat(f, 'e', -1, 64) // -d.ddde±XX
	sign := ""
	if sci[0] == '-' {
		sign, sci = "-", sci[1:]
	}
	mantissa, expPart, _ := strings.Cut(sci, "e")
	exp, err := strconv.Atoi(expPart)
	if err != nil {
		return "", err
	}
	digits := strings.Replace(mantissa, ".", "", 1)
	if exp < -4 || exp >= 16 {
		out := digits[:1]
		if len(digits) > 1 {
			out += "." + digits[1:]
		}
		expSign := "+"
		if exp < 0 {
			expSign, exp = "-", -exp
		}
		return fmt.Sprintf("%s%se%s%02d", sign, out, expSign, exp), nil
	}
	point := exp + 1 // digits before the decimal point
	switch {
	case point <= 0:
		return sign + "0." + strings.Repeat("0", -point) + digits, nil
	case point >= len(digits):
		return sign + digits + strings.Repeat("0", point-len(digits)) + ".0", nil
	default:
		return sign + digits[:point] + "." + digits[point:], nil
	}
}

// jcs is RFC 8785 canonical JSON for the value domain controls use: keys
// sorted by UTF-16 code units, no whitespace, integers exactly representable
// as IEEE doubles, and no floats (JCS number formatting is ECMAScript's, and
// no control carries one).
func jcs(v *value) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeJCS(&buf, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var maxExactInt = big.NewInt(1<<53 - 1)

func writeJCS(buf *bytes.Buffer, v *value) error {
	switch v.kind {
	case kindInt:
		if new(big.Int).Abs(v.integer).Cmp(maxExactInt) > 0 {
			return errors.New("JCS integers must be exactly representable as IEEE doubles")
		}
		buf.WriteString(v.integer.String())
	case kindFloat:
		return errors.New("controls carry no floats; JCS number formatting is not implemented")
	case kindArray:
		buf.WriteByte('[')
		for i, item := range v.items {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeJCS(buf, item); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case kindObject:
		members := append([]member{}, v.members...)
		sort.SliceStable(members, func(i, j int) bool { return utf16Less(members[i].key, members[j].key) })
		buf.WriteByte('{')
		for i, m := range members {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeString(buf, m.key)
			buf.WriteByte(':')
			if err := writeJCS(buf, m.val); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		return writeValue(buf, v, false)
	}
	return nil
}

func utf16Less(a, b string) bool {
	ua, ub := utf16.Encode([]rune(a)), utf16.Encode([]rune(b))
	for i := 0; i < len(ua) && i < len(ub); i++ {
		if ua[i] != ub[i] {
			return ua[i] < ub[i]
		}
	}
	return len(ua) < len(ub)
}
