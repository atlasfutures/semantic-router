package thinkingcontrol

import (
	"bytes"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestJSONStringEscaping(t *testing.T) {
	for _, test := range []struct{ input, want string }{
		{"", `""`},
		{"plain <>& é 世界 🦀 \u2028\u2029", "\"plain <>& é 世界 🦀 \u2028\u2029\""},
		{"\"\\\b\f\n\r\t\x00\x01\x1f", `"\"\\\b\f\n\r\t\u0000\u0001\u001f"`},
		{"a\xff\xc0\x80z", `"a���z"`},
	} {
		var got bytes.Buffer
		writeString(&got, test.input)
		if got.String() != test.want {
			t.Errorf("writeString(%q) = %q, want %q", test.input, got.String(), test.want)
		}
	}
}

// Keep the previous rune-at-a-time serializer as a byte-parity oracle. The
// renderer's compact Python-compatible output is also hashed for placement,
// so all Unicode scalars and malformed UTF-8 must keep their spelling.
func TestJSONStringMatchesRuneSerializer(t *testing.T) {
	var all strings.Builder
	for r := rune(0); r <= utf8.MaxRune; r++ {
		if r < 0xd800 || r > 0xdfff {
			all.WriteRune(r)
		}
	}
	for _, input := range []string{
		all.String(), strings.Repeat("ordinary text", 1<<16),
		strings.Repeat("é世界\"\\\n\x00\xff", 1<<12),
		"\xed\xa0\x80\xe2\x82\xf4\x90\x80\x80\xef\xbf\xbd",
	} {
		var got, want bytes.Buffer
		writeString(&got, input)
		writeStringByRune(&want, input)
		if !bytes.Equal(got.Bytes(), want.Bytes()) {
			t.Fatalf("serializer changed bytes for %d-byte input", len(input))
		}
	}
}

func FuzzJSONStringMatchesRuneSerializer(f *testing.F) {
	for _, s := range []string{"plain", "é世界🦀", "\"\\\n\x00", "\xff\xe2\x82"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		var got, want bytes.Buffer
		writeString(&got, s)
		writeStringByRune(&want, s)
		if !bytes.Equal(got.Bytes(), want.Bytes()) {
			t.Fatalf("serializer changed bytes for %q", s)
		}
	})
}

func BenchmarkJSONString(b *testing.B) {
	for _, test := range []struct{ name, text string }{
		{"plain", "synthetic plain text "},
		{"unicode", "synthetic é 世界 🦀 text "},
		{"escaped", "synthetic \n\t\\\" text "},
	} {
		input := strings.Repeat(test.text, (1<<20)/len(test.text))
		for _, writer := range []struct {
			name  string
			write func(*bytes.Buffer, string)
		}{
			{"rune", writeStringByRune}, {"current", writeString},
		} {
			b.Run(test.name+"/"+writer.name, func(b *testing.B) {
				b.SetBytes(int64(len(input)))
				b.ReportAllocs()
				var buf bytes.Buffer
				for i := 0; i < b.N; i++ {
					buf.Reset()
					writer.write(&buf, input)
				}
			})
		}
	}
}

func writeStringByRune(buf *bytes.Buffer, s string) {
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
