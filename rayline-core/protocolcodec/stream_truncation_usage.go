package protocolcodec

import (
	"encoding/json"
	"errors"
	"io"

	logging "github.com/atlasfutures/semantic-router/rayline-core/diagnostics"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Saying so when a cut turn's count cannot travel.
//
// A turn the host ended has already cost what it cost, and the count it hands
// the codec is the only record the client and whatever bills behind it will
// ever get: the stream never reached its usage frame. That count was once
// dropped for every non-Messages client by an optional capability assertion
// that returned silently -- no log, no error, no diagnostic -- so the drop was
// invisible on both sides while the upstream billed for the whole generation.
//
// So no path may drop one quietly again. Where a format genuinely cannot carry
// the count, or the client declined to be told, that is said here by name, with
// the encoder that could not write it.

// reportUnwritableTruncationUsage names the encoder that could not state a cut
// turn's count, and why. No counts and no response content pass through here:
// the encoder name and the reason are both closed vocabulary.
//
// Every stream that drops a count says so. A process-wide memo was tried and
// was wrong: the second turn to lose its count went unreported, which is the
// silence this exists to break. Within one stream the same drop is reported
// once, because a terminal can be prepared more than once and one turn is one
// line (reportedDrops).
func (engine *StreamEngine) reportUnwritableTruncationUsage(encoder, reason string) {
	if engine == nil {
		reportUnwritableTruncationUsage(encoder, reason)
		return
	}
	key := encoder + ":" + reason
	if engine.reportedDrops == nil {
		engine.reportedDrops = map[string]struct{}{}
	}
	if _, seen := engine.reportedDrops[key]; seen {
		return
	}
	engine.reportedDrops[key] = struct{}{}
	reportUnwritableTruncationUsage(encoder, reason)
}

func reportUnwritableTruncationUsage(encoder, reason string) {
	logging.ComponentWarnEvent("protocolcodec", "truncation_usage_unwritten", map[string]interface{}{
		"encoder": encoder,
		"reason":  reason,
	})
}

// truncationUsageAvailable reports whether a cut turn's terminal has a count to
// state. A turn cut before the upstream produced anything has none to estimate
// from, and a terminal of zeros would bill a turn that generated nothing.
func truncationUsageAvailable(usage *llmprotocol.Usage) bool {
	return usage != nil && usage.State == llmprotocol.UsageAvailable
}

// truncatedTrailingFragment reports whether a decode failure is what a cut
// leaves behind: bytes that simply stop.
//
// A frame the Router cut in half is valid JSON as far as it goes and fails for
// running out, so appending the bytes that never arrived would have completed
// it. Any other failure is the upstream's own doing -- a syntax error cannot be
// repaired by more bytes -- and a provider that broke the protocol must not be
// reported to the client as an ordinary truncation just because a deadline
// happened to fall afterwards.
func truncatedTrailingFragment(decodeErr error) bool {
	if decodeErr == nil {
		return false
	}
	// The bytes stopped: inside a value, inside an object or array
	// (unterminatedJSON says so by wrapping the EOF), or inside a multibyte
	// character (upstreamUTF8CutCode, which the validator tells apart from a
	// byte that is simply invalid). Each is complete as far as it goes and
	// would have been whole had the bytes kept coming. A closing token that
	// was wrong, or a byte that was never valid, is the upstream's doing.
	if errors.Is(decodeErr, io.ErrUnexpectedEOF) || errors.Is(decodeErr, io.EOF) {
		return true
	}
	// encoding/json reports input that simply ran out as a syntax error of
	// its own, "unexpected end of JSON input", with no EOF to unwrap -- and
	// it does so only for that case, naming the offending character whenever
	// the bytes were genuinely wrong. Unmarshal takes this path where the
	// streaming Decoder wraps io.ErrUnexpectedEOF instead, so a cut reached
	// one classifier and not the other depending on which read it. Reading
	// only the sentinel, never the offset, keeps a wrong closing token at the
	// end of a document ("{\"id\":\"x\"]") out: that one names its character.
	var syntax *json.SyntaxError
	if errors.As(decodeErr, &syntax) {
		return syntax.Error() == "unexpected end of JSON input"
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(decodeErr, &protocolError) {
		return false
	}
	return protocolError.Code == upstreamUTF8CutCode || protocolError.Code == upstreamFrameCutCode
}

// providerStatedFailure reports whether a failed terminal carries a failure the
// upstream itself stated, rather than one the Router's cut produced.
//
// The two reach markTruncatedTerminal by the same door, so the decoders mark
// the ones they read off the wire (UpstreamStated). Asking the event is the
// only reliable way to tell: a cut's terminal is synthesized from the reason
// the host finalized on, and inferring the difference from error codes instead
// would mistake every provider code nobody had listed yet for the Router's own.
//
// This matters only because a read can end between a provider error's JSON and
// its blank line, which leaves the decoder flushing that error at finalization
// -- after the host has handed over a count and a cause. Without this the turn
// would be re-scoped and the provider's own word for what went wrong replaced.
func providerStatedFailure(event llmprotocol.Event) bool {
	return event.Error != nil && event.UpstreamStated
}

// concedeOutputToTruncation lets a cut turn's terminal restate the output legs
// of a count the stream had already accumulated.
//
// The merge refuses a count of lower evidence replacing a higher one, and for
// every other terminal that is right. A truncated terminal is the one case
// where the lower-ranked count is the better one: the stream's statement was
// made before the output it then ran past (Anthropic's message_start says
// output_tokens 1 before the first delta), while the Router's estimate counts
// everything it forwarded. The snapshot is known to be short and the estimate
// is low on average, and the larger is nearer the truth. So the accumulated output legs yield to
// the terminal's when the terminal's total is no smaller; the splits go with
// the total, so the usage still adds up. The input legs are never conceded: a
// prompt does not grow after it is counted.
func concedeOutputToTruncation(current, terminal llmprotocol.Usage) llmprotocol.Usage {
	if terminal.OutputTotal.Value == nil {
		return current
	}
	if current.OutputTotal.Value != nil && *current.OutputTotal.Value > *terminal.OutputTotal.Value {
		return current
	}
	// Nor is a decrease of any kind conceded: a terminal whose overall total
	// is below what the stream accumulated is left to the merge, which refuses
	// it. No decoder states a total without its legs, but the contract allows
	// one, and the concession must not be the place that lets it shrink.
	if current.Total.Value != nil && terminal.Total.Value != nil && *terminal.Total.Value < *current.Total.Value {
		return current
	}
	current.OutputTotal = terminal.OutputTotal
	current.OutputReasoning = terminal.OutputReasoning
	current.OutputOther = terminal.OutputOther
	current.Total = terminal.Total
	return current
}
