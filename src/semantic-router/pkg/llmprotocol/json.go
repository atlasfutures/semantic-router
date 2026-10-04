package llmprotocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"
)

// ValidateJSONObject validates protocol-neutral JSON values that are carried
// as strings or raw messages, such as tool arguments and JSON Schemas. The
// standard library accepts duplicate object members and replaces malformed
// Unicode surrogates; both behaviors would make translation non-deterministic.
func ValidateJSONObject(body []byte, maximumDepth int) error {
	if !utf8.Valid(body) {
		return fmt.Errorf("JSON is not valid UTF-8")
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok || delimiter != '{' {
		return fmt.Errorf("JSON value must be an object")
	}
	if err := consumeJSONObject(decoder, 0, maximumDepth); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("JSON contains an additional document")
		}
		return err
	}
	return nil
}

func consumeJSONValue(decoder *json.Decoder, depth, maximumDepth int) error {
	if maximumDepth <= 0 || depth > maximumDepth {
		return fmt.Errorf("JSON nesting exceeds the configured limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		return consumeJSONObject(decoder, depth, maximumDepth)
	case '[':
		return consumeJSONArray(decoder, depth, maximumDepth)
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}

func consumeJSONObject(decoder *json.Decoder, depth, maximumDepth int) error {
	seen := make(map[string]struct{})
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := keyToken.(string)
		if !ok {
			return fmt.Errorf("JSON object key is not a string")
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("JSON object contains duplicate field %q", key)
		}
		seen[key] = struct{}{}
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return fmt.Errorf("unterminated JSON object")
	}
	return nil
}

func consumeJSONArray(decoder *json.Decoder, depth, maximumDepth int) error {
	for decoder.More() {
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim(']') {
		return fmt.Errorf("unterminated JSON array")
	}
	return nil
}

// encoding/json replaces unpaired UTF-16 surrogates with U+FFFD. Reject them
// before decoding so identifiers, tool arguments, and schemas remain exact.
func validateJSONUnicodeEscapes(body []byte) error {
	insideString := false
	for index := 0; index < len(body); index++ {
		if body[index] == '"' {
			insideString = !insideString
			continue
		}
		if body[index] != '\\' || !insideString {
			continue
		}
		next, err := advanceJSONUnicodeEscape(body, index)
		if err != nil {
			return err
		}
		index = next
	}
	return nil
}

func advanceJSONUnicodeEscape(body []byte, index int) (int, error) {
	if index+1 >= len(body) {
		return index, nil
	}
	if body[index+1] != 'u' {
		return index + 1, nil
	}
	value, ok := decodeHexQuad(body, index+2)
	if !ok {
		return index, nil
	}
	if value >= 0xdc00 && value <= 0xdfff {
		return index, fmt.Errorf("JSON contains a lone low surrogate")
	}
	if value < 0xd800 || value > 0xdbff {
		return index + 5, nil
	}
	if !hasLowSurrogateEscape(body, index) {
		return index, fmt.Errorf("JSON high surrogate is not followed by a low surrogate")
	}
	return index + 11, nil
}

func hasLowSurrogateEscape(body []byte, index int) bool {
	if index+11 >= len(body) || body[index+6] != '\\' || body[index+7] != 'u' {
		return false
	}
	low, validLow := decodeHexQuad(body, index+8)
	return validLow && low >= 0xdc00 && low <= 0xdfff
}

func decodeHexQuad(body []byte, start int) (uint16, bool) {
	if start < 0 || start+4 > len(body) {
		return 0, false
	}
	var value uint16
	for _, character := range body[start : start+4] {
		value <<= 4
		switch {
		case character >= '0' && character <= '9':
			value += uint16(character - '0')
		case character >= 'a' && character <= 'f':
			value += uint16(character-'a') + 10
		case character >= 'A' && character <= 'F':
			value += uint16(character-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}

// TruncatedJSONObject reports arguments that are a strict prefix of a JSON
// object: what a model leaves when its output limit cuts it off while it
// writes a tool call's arguments. The bytes open an object, tokenize without
// a syntax error until they run out, stay within maximumDepth, and repeat no
// member name, as ValidateJSONObject requires of a whole object. A whole
// object is not truncated, and neither is a value that is not an object or
// text that no appended bytes could make valid, such as {] or {"x":].
func TruncatedJSONObject(arguments []byte, maximumDepth int) bool {
	trimmed := bytes.TrimSpace(arguments)
	if len(trimmed) == 0 || trimmed[0] != '{' || maximumDepth <= 0 || !utf8.Valid(trimmed) {
		return false
	}
	if !validJSONEscapesPrefix(trimmed) {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.UseNumber()
	err := walkJSONObjectPrefix(decoder, maximumDepth)
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

// jsonPrefixFrame is one container still open while a prefix is walked.
type jsonPrefixFrame struct {
	object    bool
	expectKey bool
	seen      map[string]struct{}
}

// walkJSONObjectPrefix walks tokens until the input runs out, which the
// decoder reports as io.EOF or io.ErrUnexpectedEOF. Any other error, or the
// top object closing, means the bytes are no strict prefix of an object. The
// decoder enforces the grammar (colons, commas, closers); the walk adds what
// ValidateJSONObject adds: string keys, no duplicate member, and depth,
// counted as consumeJSONValue counts it, with the top object's members at 1.
func walkJSONObjectPrefix(decoder *json.Decoder, maximumDepth int) error {
	var stack []jsonPrefixFrame
	for {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, isDelimiter := token.(json.Delim)
		switch {
		case isDelimiter && (delimiter == '}' || delimiter == ']'):
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				return fmt.Errorf("JSON object is complete")
			}
			markJSONPrefixValue(&stack[len(stack)-1])
		case len(stack) > 0 && stack[len(stack)-1].object && stack[len(stack)-1].expectKey:
			if err := takeJSONPrefixKey(&stack[len(stack)-1], token); err != nil {
				return err
			}
		case len(stack) > maximumDepth:
			return fmt.Errorf("JSON nesting exceeds the configured limit")
		case isDelimiter:
			stack = append(stack, jsonPrefixFrame{
				object: delimiter == '{', expectKey: delimiter == '{', seen: map[string]struct{}{},
			})
		case len(stack) == 0:
			return fmt.Errorf("JSON value must be an object")
		default:
			markJSONPrefixValue(&stack[len(stack)-1])
		}
	}
}

func takeJSONPrefixKey(frame *jsonPrefixFrame, token json.Token) error {
	key, ok := token.(string)
	if !ok {
		return fmt.Errorf("JSON object key is not a string")
	}
	if _, duplicate := frame.seen[key]; duplicate {
		return fmt.Errorf("JSON object contains duplicate field %q", key)
	}
	frame.seen[key] = struct{}{}
	frame.expectKey = false
	return nil
}

func markJSONPrefixValue(frame *jsonPrefixFrame) {
	if frame.object {
		frame.expectKey = true
	}
}

// validJSONEscapesPrefix applies validateJSONUnicodeEscapes to a prefix. The
// one escape a prefix may leave unfinished is a surrogate pair cut between
// its halves: a high-surrogate escape followed by the start of a low one.
func validJSONEscapesPrefix(body []byte) bool {
	if validateJSONUnicodeEscapes(body) == nil {
		return true
	}
	start := cutSurrogatePair(body)
	return start >= 0 && validateJSONUnicodeEscapes(body[:start]) == nil
}

// cutSurrogatePair returns where a trailing high-surrogate escape starts
// when what follows it can still become its low surrogate, or -1.
func cutSurrogatePair(body []byte) int {
	for start := len(body) - 6; start >= 0 && start > len(body)-12; start-- {
		if body[start] != '\\' || body[start+1] != 'u' {
			continue
		}
		high, ok := decodeHexQuad(body, start+2)
		if ok && high >= 0xd800 && high <= 0xdbff && lowSurrogateEscapePrefix(body[start+6:]) {
			return start
		}
	}
	return -1
}

// lowSurrogateEscapePrefix reports a strict prefix of an escape \uDC00
// through \uDFFF.
func lowSurrogateEscapePrefix(tail []byte) bool {
	if len(tail) >= 6 {
		return false
	}
	for index, character := range tail {
		var ok bool
		switch index {
		case 0:
			ok = character == '\\'
		case 1:
			ok = character == 'u'
		case 2:
			ok = character == 'd' || character == 'D'
		case 3:
			ok = bytes.IndexByte([]byte("cdefCDEF"), character) >= 0
		default:
			ok = bytes.IndexByte([]byte("0123456789abcdefABCDEF"), character) >= 0
		}
		if !ok {
			return false
		}
	}
	return true
}
