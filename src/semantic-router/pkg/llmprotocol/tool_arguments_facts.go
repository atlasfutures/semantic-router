package llmprotocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"unicode/utf8"
)

// ToolArgumentsFacts describes tool-call arguments the Router refused, for an
// operator log line, without any of their content: sizes, shapes and flags,
// never a byte of the text. It rides as the cause of the refusal, so the
// error that reaches the log carries it and the client's error body, which
// is built from the message, does not.
type ToolArgumentsFacts struct {
	// Stage is where the refusal was made: "item_completed" when the call
	// completed, "after_cut" when output followed a held cut call, and
	// "terminal" when the turn ended under a stop that does not cut.
	Stage string
	// ToolName is the called tool's name, a client-declared identifier.
	ToolName string
	// Bytes is the arguments' length.
	Bytes int
	// FirstByte is the first non-whitespace byte as a category: "{", "[",
	// "other", or "empty".
	FirstByte string
	// ValidUTF8 reports whether the arguments are valid UTF-8.
	ValidUTF8 bool
	// StdlibObject reports whether encoding/json accepts them as an object.
	StdlibObject bool
	// JSONErrorOffset is where encoding/json's syntax error sits, or -1.
	JSONErrorOffset int64
	// DuplicateKey reports a member name repeated within one object.
	DuplicateKey bool
	// LoneSurrogate reports an unpaired surrogate escape.
	LoneSurrogate bool
	// RawControl reports a raw control character inside a string.
	RawControl bool
	// ReplacementChar reports U+FFFD, which an upstream that split a
	// multi-byte character and replaced the halves would leave.
	ReplacementChar bool
	// MaxDepth is the deepest container nesting reached, the top object
	// being 1, up to the first syntax error.
	MaxDepth int
	// Chunks is how many streamed argument deltas made the arguments.
	Chunks int
	// ChunkSplitRune reports a delta that ended inside a UTF-8 sequence.
	ChunkSplitRune bool
}

func (facts *ToolArgumentsFacts) Error() string {
	if facts == nil {
		return ""
	}
	return fmt.Sprintf("tool arguments refused at %s: %d bytes in %d chunks", facts.Stage, facts.Bytes, facts.Chunks)
}

// DescribeToolArguments computes the content-free facts of arguments. The
// caller fills Stage, ToolName and the chunk fields, which the arguments
// alone do not say.
func DescribeToolArguments(arguments []byte) ToolArgumentsFacts {
	facts := ToolArgumentsFacts{
		Bytes:           len(arguments),
		FirstByte:       "empty",
		ValidUTF8:       utf8.Valid(arguments),
		JSONErrorOffset: -1,
		LoneSurrogate:   validateJSONUnicodeEscapes(arguments) != nil,
		ReplacementChar: bytes.Contains(arguments, []byte(string(utf8.RuneError))),
		RawControl:      rawControlInString(arguments),
	}
	if trimmed := bytes.TrimSpace(arguments); len(trimmed) > 0 {
		switch trimmed[0] {
		case '{', '[':
			facts.FirstByte = string(trimmed[0])
		default:
			facts.FirstByte = "other"
		}
	}
	var object map[string]json.RawMessage
	err := json.Unmarshal(arguments, &object)
	facts.StdlibObject = err == nil && facts.FirstByte == "{"
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		facts.JSONErrorOffset = syntaxErr.Offset
	}
	facts.MaxDepth, facts.DuplicateKey = walkJSONShape(arguments)
	return facts
}

// walkJSONShape returns the deepest nesting and whether any object repeats a
// member name, reading tokens until the first error.
func walkJSONShape(arguments []byte) (int, bool) {
	decoder := json.NewDecoder(bytes.NewReader(arguments))
	type frame struct {
		object, expectKey bool
		seen              map[string]struct{}
	}
	var stack []frame
	maxDepth, duplicate := 0, false
	for {
		token, err := decoder.Token()
		if err != nil {
			return maxDepth, duplicate
		}
		delimiter, isDelimiter := token.(json.Delim)
		switch {
		case isDelimiter && (delimiter == '}' || delimiter == ']'):
			stack = stack[:len(stack)-1]
			if len(stack) > 0 && stack[len(stack)-1].object {
				stack[len(stack)-1].expectKey = true
			}
		case len(stack) > 0 && stack[len(stack)-1].object && stack[len(stack)-1].expectKey:
			top := &stack[len(stack)-1]
			if key, ok := token.(string); ok {
				if _, repeated := top.seen[key]; repeated {
					duplicate = true
				}
				top.seen[key] = struct{}{}
			}
			top.expectKey = false
		case isDelimiter:
			stack = append(stack, frame{object: delimiter == '{', expectKey: delimiter == '{', seen: map[string]struct{}{}})
			maxDepth = max(maxDepth, len(stack))
		case len(stack) > 0 && stack[len(stack)-1].object:
			stack[len(stack)-1].expectKey = true
		}
	}
}

// rawControlInString reports a byte below 0x20 inside a JSON string, which
// JSON requires to be escaped.
func rawControlInString(arguments []byte) bool {
	inside := false
	for index := 0; index < len(arguments); index++ {
		switch character := arguments[index]; {
		case character == '"':
			inside = !inside
		case inside && character == '\\':
			index++
		case inside && character < 0x20:
			return true
		}
	}
	return false
}
