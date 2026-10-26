package protocolcodec

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

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

// reportedUnwritableTruncationUsage keeps one line per encoder and reason per
// process. A cell under load cuts many turns for the same reason, and the set
// of encoders and reasons is closed and small, so every distinct drop is still
// named while a steady stream of them cannot flood the log.
var reportedUnwritableTruncationUsage sync.Map

// reportUnwritableTruncationUsage names the encoder that could not state a cut
// turn's count, and why. No counts and no response content pass through here:
// the encoder name and the reason are both closed vocabulary.
func reportUnwritableTruncationUsage(encoder, reason string) {
	key := encoder + ":" + reason
	if _, seen := reportedUnwritableTruncationUsage.LoadOrStore(key, struct{}{}); seen {
		return
	}
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
	var syntax *json.SyntaxError
	if errors.As(decodeErr, &syntax) {
		return false
	}
	return errors.Is(decodeErr, io.ErrUnexpectedEOF) || errors.Is(decodeErr, io.EOF) ||
		strings.Contains(decodeErr.Error(), "unexpected EOF") ||
		strings.Contains(decodeErr.Error(), "unexpected end of JSON input")
}

// truncationTerminalCarriesUsage reports whether this target's own terminal has
// somewhere to state a count. Chat's usage chunk is opt-in, so a client that
// did not ask for it is told no counts on any terminal; Messages and Responses
// state usage on theirs unconditionally.
func (engine *StreamEngine) truncationTerminalCarriesUsage() bool {
	if engine.targetFormat != llmprotocol.OpenAIChatV1 {
		return true
	}
	return streamUsageRequested(engine.streamContext)
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
