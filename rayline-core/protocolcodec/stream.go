package protocolcodec

import (
	"bytes"
	"encoding/json"
	"io"
	"unicode/utf8"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

type sseFrame struct {
	Event   string
	Data    []byte
	HasData bool
	// HasComment and HasField record a comment line (": ...") and any field
	// line. A frame of comment lines only is a keepalive.
	HasComment bool
	HasField   bool
}

// commentOnly reports a frame that holds SSE comment lines and nothing else,
// the form a provider keeps a quiet stream alive with (OpenRouter's
// ": OPENROUTER PROCESSING").
func (frame sseFrame) commentOnly() bool {
	return frame.HasComment && !frame.HasField
}

// streamWireIndexes maps protocol-neutral item identities to the compact,
// contiguous indexes required by provider stream contracts. A source format
// may reserve neutral indexes; those implementation details must never leak
// onto a target wire.
type streamWireIndexes struct {
	items map[int]int
	next  int
}

type streamContentKey struct {
	item    int
	content int
}

func contentKey(event llmprotocol.Event) streamContentKey {
	return streamContentKey{item: event.ItemIndex, content: event.ContentIndex}
}

func (indexes *streamWireIndexes) translate(event llmprotocol.Event) int {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		if indexes.items == nil {
			indexes.items = make(map[int]int)
		}
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
		wireIndex := indexes.next
		indexes.next++
		indexes.items[event.ItemIndex] = wireIndex
		return wireIndex
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventImageGenerationProgress,
		llmprotocol.EventOutputItemCompleted:
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
	}
	return event.ItemIndex
}

// sseFramer turns arbitrary transport chunks into complete SSE events. It is
// request-scoped, retains at most one bounded unfinished event, and accepts LF,
// CRLF, or CR line endings without assuming network read boundaries.
type sseFramer struct {
	buffer []byte
	limit  int
	// skipLF is set when a frame ended on a CR that was the last byte read:
	// that CR may be the first half of a CRLF split across reads, so a LF
	// that opens the next chunk completes it and is not a line of its own.
	skipLF bool
}

func newSSEFramer(limit int) sseFramer { return sseFramer{limit: limit} }

func (framer *sseFramer) Push(chunk []byte) ([][]byte, error) {
	if len(chunk) == 0 {
		return nil, nil
	}
	if framer.skipLF && chunk[0] == '\n' {
		chunk = chunk[1:]
	}
	framer.skipLF = false
	framer.buffer = append(framer.buffer, chunk...)
	frames := make([][]byte, 0, 1)
	for {
		end, complete := completeSSEFrame(framer.buffer)
		if !complete {
			break
		}
		if end > framer.limit {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is too large", nil)
		}
		frames = append(frames, append([]byte(nil), framer.buffer[:end]...))
		framer.skipLF = end == len(framer.buffer) && framer.buffer[end-1] == '\r'
		framer.buffer = append(framer.buffer[:0], framer.buffer[end:]...)
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	return frames, nil
}

func (framer *sseFramer) Finalize() ([][]byte, error) {
	if len(bytes.TrimSpace(framer.buffer)) == 0 {
		framer.buffer = nil
		return nil, nil
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	frame := append([]byte(nil), framer.buffer...)
	framer.buffer = nil
	if sseFrameDataCutBeforeItsValue(frame) {
		// The frame's "data:" arrived and its value did not. Pushing it on
		// would be refused for an empty document, which is what a provider's
		// own empty frame is -- this is a read that stopped at that boundary,
		// and the bytes that never came would have completed it. Saying so
		// keeps a cut here from reaching the client as malformed upstream
		// bytes with its count dropped (truncatedTrailingFragment).
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, upstreamFrameCutCode,
			"upstream SSE frame ends before its data value", io.ErrUnexpectedEOF,
		)
	}
	return [][]byte{frame}, nil
}

// upstreamFrameCutCode names an unfinished frame that stops where its data
// value would begin.
const upstreamFrameCutCode = "upstream_frame_cut"

// afterTerminalDiagnostic records a frame dropped because it came after the
// stream's terminal event, when the message was already complete.
func afterTerminalDiagnostic(source llmprotocol.WireFormat, terminal, frame string) llmprotocol.Diagnostics {
	return llmprotocol.Diagnostics{{
		Source: source, Field: "stream.after_" + terminal, Action: llmprotocol.DiagnosticDropped,
		Reason: "the provider sent " + frame + " after " + terminal + "; the message was already complete",
	}}
}

// sentinelCutShort reports data that is a proper prefix of the OpenAI SSE
// convention's terminal sentinel: a read that stopped partway through
// "[DONE]". Anything else, including a complete sentinel, is left alone.
//
// Both the Chat decoder and the Anthropic decoder need this: OpenRouter's
// Messages adapter sends the sentinel as a trailer after message_stop, so a
// stream carrying Anthropic events can end inside it too.
func sentinelCutShort(data []byte) bool {
	trimmed := bytes.TrimSpace(data)
	sentinel := []byte("[DONE]")
	if len(trimmed) == 0 || len(trimmed) >= len(sentinel) {
		return false
	}
	return bytes.Equal(sentinel[:len(trimmed)], trimmed)
}

// sseFrameDataCutBeforeItsValue reports whether an unfinished frame names a
// data field and carries no value for it.
//
// The frame is read with the stream's own line parser, so every line
// terminator the format allows is handled and several data lines accumulate
// into one value, as they do on a complete frame. A frame whose accumulated
// value is empty stopped where that value would begin; one carrying anything
// -- even half a JSON object, even an earlier line's whole value -- is passed
// on and classified by the decoder that reads it, so a provider's malformed
// bytes are never hidden behind a trailing bare "data:".
func sseFrameDataCutBeforeItsValue(frame []byte) bool {
	// first=true so a leading BOM is stripped rather than refused: this frame
	// may be the stream's first, and whether it is one is not what decides
	// whether the read stopped before a value.
	normalized, err := normalizeSSEFrame(frame, true)
	if err != nil {
		// A frame this parser refuses is not one to reclassify.
		normalized = frame
	}
	var parsed sseFrame
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		parseSSELine(&parsed, line)
	}
	return parsed.HasData && len(bytes.TrimSpace(parsed.Data)) == 0
}

func completeSSEFrame(payload []byte) (int, bool) {
	lineStart := 0
	for index := 0; index < len(payload); {
		lineEnd := index
		terminator := 0
		switch payload[index] {
		case '\n':
			terminator = 1
		case '\r':
			terminator = 1
			if index+1 < len(payload) && payload[index+1] == '\n' {
				terminator = 2
			}
		default:
			index++
			continue
		}
		if lineEnd == lineStart {
			return index + terminator, true
		}
		index += terminator
		lineStart = index
	}
	return 0, false
}

type (
	decoderFrameFinalizer func() ([][]byte, error)
	decoderFrameProcessor func([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error)
)

func finalizeDecoderFrames(
	finalize decoderFrameFinalizer,
	process decoderFrameProcessor,
	diagnosticLimit int,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := finalize()
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := process(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, diagnosticLimit)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func parseSSEFrame(frame []byte, limit int) (sseFrame, error) {
	return parseSSEFrameAtPosition(frame, limit, true)
}

func parseSSEFrameAtPosition(frame []byte, limit int, first bool) (sseFrame, error) {
	if err := validateSSEFrameBytes(frame, limit); err != nil {
		return sseFrame{}, err
	}
	normalized, err := normalizeSSEFrame(frame, first)
	if err != nil {
		return sseFrame{}, err
	}
	var result sseFrame
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		parseSSELine(&result, line)
	}
	return result, nil
}

func validateSSEFrameBytes(frame []byte, limit int) error {
	if len(frame) == 0 || len(frame) > limit {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is empty or too large", nil)
	}
	if !utf8.Valid(frame) {
		if utf8RuneCutAtTail(frame) {
			// Valid up to a character the bytes stopped inside: what a cut
			// leaves behind, not what a provider sends.
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable, upstreamUTF8CutCode, "upstream SSE frame ends inside a UTF-8 character", nil,
			)
		}
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8", "upstream SSE frame is not valid UTF-8", nil,
		)
	}
	return nil
}

// upstreamUTF8CutCode names a frame whose only fault is a multibyte character
// cut in half at its end. Appending the missing bytes would have made it
// valid, which is never true of an invalid byte; the two must not share a
// code, because a cut turn bills and a malformed one does not.
const upstreamUTF8CutCode = "upstream_utf8_cut"

// utf8RuneCutAtTail reports whether a frame is valid UTF-8 except for an
// incomplete multibyte sequence at its very end.
func utf8RuneCutAtTail(frame []byte) bool {
	for i := 0; i < len(frame); {
		r, size := utf8.DecodeRune(frame[i:])
		if r != utf8.RuneError || size != 1 {
			i += size
			continue
		}
		tail := frame[i:]
		if len(tail) >= utf8.UTFMax || utf8.FullRune(tail) {
			return false
		}
		// The tail is a lead byte and continuation bytes of a longer sequence
		// that never finished; any other arrangement is simply invalid.
		if tail[0] < 0xc2 || tail[0] > 0xf4 {
			return false
		}
		for _, b := range tail[1:] {
			if b&0xc0 != 0x80 {
				return false
			}
		}
		return true
	}
	return false
}

func normalizeSSEFrame(frame []byte, first bool) ([]byte, error) {
	bom := []byte{0xef, 0xbb, 0xbf}
	if !first && bytes.HasPrefix(frame, bom) {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unexpected_stream_bom",
			"SSE stream contains a UTF-8 BOM after its first event",
			nil,
		)
	}
	normalized := frame
	if first {
		normalized = bytes.TrimPrefix(normalized, bom)
	}
	normalized = bytes.ReplaceAll(normalized, []byte("\r\n"), []byte("\n"))
	normalized = bytes.ReplaceAll(normalized, []byte("\r"), []byte("\n"))
	return normalized, nil
}

func parseSSELine(result *sseFrame, line []byte) {
	if len(line) == 0 {
		return
	}
	if line[0] == ':' {
		result.HasComment = true
		return
	}
	result.HasField = true
	name, value, found := bytes.Cut(line, []byte{':'})
	if !found {
		name, value = line, nil
	}
	value = bytes.TrimPrefix(value, []byte{' '})
	switch string(name) {
	case "event":
		result.Event = string(value)
	case "data":
		result.HasData = true
		if len(result.Data) > 0 {
			result.Data = append(result.Data, '\n')
		}
		result.Data = append(result.Data, value...)
	}
}

// sseKeepaliveComment is a keepalive as an SSE comment, which every SSE
// parser skips. Chat and Responses have no keepalive event of their own.
func sseKeepaliveComment() []byte {
	return []byte(": keepalive\n\n")
}

func encodeSSE(event string, data any) ([]byte, error) {
	body, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	var buffer bytes.Buffer
	if event != "" {
		buffer.WriteString("event: ")
		buffer.WriteString(event)
		buffer.WriteByte('\n')
	}
	buffer.WriteString("data: ")
	buffer.Write(body)
	buffer.WriteString("\n\n")
	return buffer.Bytes(), nil
}
