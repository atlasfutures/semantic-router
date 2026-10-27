package protocolcodec

import (
	"fmt"
	"math"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// How a stream ends.
//
// Every decision about a terminal event lives here: which terminal a turn is
// allowed to publish, when a successful one has to be held back or suppressed,
// and what the client is told when the turn failed instead. The rest of the
// engine moves frames; this decides whether the turn is over and how.
//
// The rule the whole file exists to hold: a turn that did not succeed must
// never publish a success terminal, whatever went wrong and whoever noticed.

func (engine *StreamEngine) recordTerminalEvent(event llmprotocol.Event) {
	if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
		engine.terminal = true
	}
	// An output the upstream itself finished states the turn's real stop. Only
	// the transport is left to go wrong after that, and a cut there must not
	// restate a finished answer as one stopped at a limit
	// (markTruncatedTerminal).
	if event.Type != llmprotocol.EventResponseFailed && naturalStop(event.StopReason) {
		engine.servedStop = event.StopReason
	}
	// Every count the stream states passes through here. The terminal the
	// engine synthesizes for a cut is not built by the decoder, so it does not
	// inherit what the decoder accumulated; keeping the last statement is how
	// that terminal still knows what the upstream had said (reconciledTruncationUsage).
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		stated := *event.Usage
		engine.statedUsage = &stated
		engine.outputSinceUsage = false
	}
	// Output that arrives after a count was stated makes that count's output
	// leg a snapshot rather than a settlement (reconciledTruncationUsage).
	if outputDelta(event) {
		engine.outputSinceUsage = true
	}
}

// outputDelta reports an event that adds to what the turn has generated.
func outputDelta(event llmprotocol.Event) bool {
	switch event.Type {
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta:
		return event.Delta != ""
	case llmprotocol.EventToolCallDelta:
		return event.ToolCall != nil && event.ToolCall.Arguments != ""
	}
	return false
}

// sameUsageTotals reports whether two counts state the same totals, so a
// repeat of a statement can be told from a new one.
func sameUsageTotals(left, right llmprotocol.Usage) bool {
	same := func(a, b llmprotocol.TokenCount) bool {
		if a.Value == nil || b.Value == nil {
			return a.Value == nil && b.Value == nil
		}
		return *a.Value == *b.Value
	}
	return same(left.InputTotal, right.InputTotal) && same(left.OutputTotal, right.OutputTotal) &&
		same(left.Total, right.Total)
}

// naturalStop reports a stop the model reached on its own, as opposed to one
// naming a failure or a stop nothing has stated yet.
func naturalStop(stop llmprotocol.StopReason) bool {
	switch stop {
	case "", llmprotocol.StopError, llmprotocol.StopUnknown:
		return false
	}
	return true
}

func suppressSuccessfulTerminal(events []llmprotocol.Event) []llmprotocol.Event {
	result := events[:0]
	for _, event := range events {
		if event.Type != llmprotocol.EventResponseCompleted {
			result = append(result, event)
		}
	}
	return result
}

// A provider's semantic success event is held until the HTTP body reaches a
// clean end-of-stream. Otherwise a terminal frame followed by a malformed
// unterminated fragment in a later transport read could publish success before
// the codec has had an opportunity to validate the trailing bytes.
func (engine *StreamEngine) deferSuccessfulTerminal(events []llmprotocol.Event) []llmprotocol.Event {
	result := events[:0]
	for index := range events {
		if events[index].Type != llmprotocol.EventResponseCompleted {
			result = append(result, events[index])
			continue
		}
		completion := events[index]
		engine.pendingCompletion = &completion
	}
	return result
}

func (engine *StreamEngine) poison(err error) {
	if err == nil {
		return
	}
	if engine.failure == nil {
		engine.failure = err
	}
	engine.pendingCompletion = nil
	engine.terminal = true
}

func (engine *StreamEngine) Finalize(reason error) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if engine == nil || engine.decoder == nil || engine.encoder == nil {
		return nil, nil, nil, fmt.Errorf("stream codec engine is unavailable")
	}
	if engine.finalized {
		return nil, nil, nil, nil
	}
	engine.finalized = true
	finalization := engine.prepareFinalization(reason)
	frames, acceptedEvents, diagnostics, eventErr := engine.encodeEvents(
		finalization.events,
		finalization.diagnostics,
	)
	if eventErr != nil {
		return engine.finalizeFailure(frames, acceptedEvents, diagnostics, eventErr)
	}
	encoded, encodeDiagnostics, encodeErr := engine.encoder.Finalize(finalization.terminalReason)
	diagnostics = appendDiagnostics(diagnostics, encodeDiagnostics, engine.maxDiagnostics)
	engine.terminal = true
	if encodeErr == nil {
		frames = append(frames, encoded...)
	}
	if finalization.decodeErr != nil {
		return frames, acceptedEvents, diagnostics, finalization.resultError()
	}
	return frames, acceptedEvents, diagnostics, encodeErr
}

type streamFinalization struct {
	events         []llmprotocol.Event
	diagnostics    llmprotocol.Diagnostics
	terminalReason error
	decodeErr      error
	firstFailure   error
}

func (engine *StreamEngine) prepareFinalization(reason error) streamFinalization {
	firstFailure := engine.failure
	if firstFailure != nil {
		reason = firstFailure
	}
	events, diagnostics, decodeErr := engine.decoder.Finalize(reason)
	if engine.cutByCallerOrCounted() {
		events, diagnostics = engine.withholdOutputFlushedAtCut(events, diagnostics)
	}
	result := streamFinalization{
		events: events, diagnostics: diagnostics, terminalReason: reason,
		decodeErr: decodeErr, firstFailure: firstFailure,
	}
	switch {
	case decodeErr != nil && engine.cutByCallerOrCounted() && firstFailure == nil &&
		truncatedTrailingFragment(decodeErr):
		// The Router cut this turn, and the bytes it cut it in the middle of
		// were half a frame. A transport read does not end on a frame boundary,
		// so that is the ordinary shape of a cut rather than a provider sending
		// something malformed, and the trailing fragment is a consequence of
		// the cut and not its cause. The turn ends on the Router's reason,
		// carrying the count, as a cut on a frame boundary does; a partial
		// frame cannot be the difference between a turn that bills and one
		// that does not. The fragment itself is still never published.
		result.events = engine.transportFailureEvents(events, reason)
		result.decodeErr = nil
	case decodeErr != nil:
		result.events = suppressSuccessfulTerminal(events)
		engine.pendingCompletion = nil
		result.terminalReason = decodeErr
		if firstFailure != nil {
			result.terminalReason = firstFailure
		}
	case reason != nil:
		result.events = engine.transportFailureEvents(events, reason)
	case engine.pendingCompletion != nil:
		result.events = append(events, *engine.pendingCompletion)
		engine.pendingCompletion = nil
	}
	return result
}

func (engine *StreamEngine) transportFailureEvents(
	events []llmprotocol.Event,
	reason error,
) []llmprotocol.Event {
	events = suppressSuccessfulTerminal(events)
	// The held completion is dropped, because a turn that did not reach a clean
	// end of stream must not publish a success terminal. What it had already
	// established about the turn is not dropped with it: the tier it was served
	// at, the stop the model reached, and the count the upstream stated. A
	// successful terminal never passes through recordTerminalEvent, so this is
	// the only place that evidence is seen.
	var servedTier string
	if engine.pendingCompletion != nil {
		servedTier = engine.pendingCompletion.UpstreamServiceTier
		if naturalStop(engine.pendingCompletion.StopReason) {
			engine.servedStop = engine.pendingCompletion.StopReason
		}
		if engine.pendingCompletion.Usage != nil &&
			engine.pendingCompletion.Usage.State == llmprotocol.UsageAvailable {
			// Whether the completion's count is a fresh statement depends on
			// the format. A Chat [DONE] completion carries the usage the stream
			// had accumulated, whatever its age; a Responses response.completed
			// carries the provider's final count on the resource itself. The
			// difference is visible in the numbers: a count that repeats the
			// last statement is a carry-forward, and whether output ran past
			// that statement is left exactly as it was; a count that says
			// something new is a settlement, after everything the turn
			// generated.
			stated := *engine.pendingCompletion.Usage
			if engine.statedUsage == nil || !sameUsageTotals(*engine.statedUsage, stated) {
				engine.outputSinceUsage = false
			}
			engine.statedUsage = &stated
		}
	}
	engine.pendingCompletion = nil
	if containsFailedTerminal(events) {
		// The decoder reached the failure first -- for a Router cut it always
		// does, because the cut is handed to it as the reason it finalizes on.
		// That event is the one about to be published, so it is the one the
		// count has to ride.
		return engine.markTruncatedTerminal(events)
	}
	return engine.markTruncatedTerminal(append(events, llmprotocol.Event{
		Type:                llmprotocol.EventResponseFailed,
		StopReason:          llmprotocol.StopError,
		Error:               streamFinalizationError(reason, "stream ended before completion"),
		Failure:             llmprotocol.FailureTransport,
		UpstreamServiceTier: servedTier,
	}))
}

// withholdOutputFlushedAtCut keeps output the decoder only produced at
// finalization from reaching a client whose turn the Router has already cut.
//
// A transport read can end after a complete event and before the blank line
// that terminates it. The framer holds that event, and Finalize flushes it --
// after the Router has counted the turn and handed the count over. Publishing
// it would deliver output the terminal's count does not cover, and the count
// cannot be retaken here: the Router estimates from what it forwarded, and
// this is forwarded by nobody yet. So the output is withheld, and the
// diagnostics say so. What else the flush carried -- a count the upstream
// stated, a stop an item reached -- is kept and recorded ahead of the terminal
// it informs, since the terminal is reconciled before these events are
// otherwise seen.
func (engine *StreamEngine) withholdOutputFlushedAtCut(
	events []llmprotocol.Event,
	diagnostics llmprotocol.Diagnostics,
) ([]llmprotocol.Event, llmprotocol.Diagnostics) {
	kept := events[:0]
	withheld := 0
	for _, event := range events {
		if outputDelta(event) || event.Type == llmprotocol.EventImageGenerationProgress {
			withheld++
			continue
		}
		// An item completed here may carry what nobody has seen -- a call's
		// whole arguments, an image, a search's result -- so it is not
		// published either; the cut's own closer states the item as what had
		// reached the client. The stop it states is still recorded, because
		// a stop is a fact about the turn rather than output.
		if event.Type == llmprotocol.EventOutputItemCompleted {
			engine.recordTerminalEvent(event)
			withheld++
			continue
		}
		// A completion flushed here is one the transport cut off from its
		// blank line: the provider finished and said so. It is held exactly
		// as deferSuccessfulTerminal holds one, so transportFailureEvents
		// harvests what it established -- its count, its stop, its tier --
		// before the cut's terminal is built. A failed terminal the decoder
		// built carries the usage it accumulated, not a fresh statement;
		// recording it here would make a snapshot look settled, so it is
		// reconciled on its own, after this.
		if event.Type == llmprotocol.EventResponseCompleted {
			completion := event
			engine.pendingCompletion = &completion
			continue
		}
		if event.Type != llmprotocol.EventResponseFailed {
			engine.recordTerminalEvent(event)
		}
		kept = append(kept, event)
	}
	if withheld > 0 {
		diagnostics = appendDiagnostics(diagnostics, llmprotocol.Diagnostics{{
			Source: engine.sourceFormat, Field: "stream.output", Action: llmprotocol.DiagnosticDropped,
			Reason: "output decoded at the cut is not published: the turn was counted before it arrived",
		}}, engine.maxDiagnostics)
	}
	return kept, diagnostics
}

// markTruncatedTerminal re-scopes the terminal of a turn the Router ended.
//
// Such a turn is not a transport failure: the generation happened, the upstream
// billed us for it, and the count the Router handed over says how much. So the
// terminal carries that count, a length stop, and the truncated scope, and each
// encoder ends the turn in the shape its own format gives a response that
// stopped early rather than as an error with no numbers on it. Without a count
// there is nothing to say and nothing to re-scope: the turn produced no output,
// so it ends as the failure it is.
func (engine *StreamEngine) markTruncatedTerminal(events []llmprotocol.Event) []llmprotocol.Event {
	if engine.truncationUsage == nil {
		return events
	}
	// Something was already wrong with this stream before the Router ended it:
	// the upstream broke the protocol and the engine is poisoned. The turn ends
	// on that, because a deadline reached afterwards does not make a malformed
	// stream well-formed, and a clean length stop here would present a
	// provider's violation as an ordinary truncation.
	if engine.failure != nil {
		reportUnwritableTruncationUsage(
			string(engine.targetFormat), "the stream had already failed before the router ended it",
		)
		return events
	}
	// Which stop the terminal states is the stop the turn actually reached. A
	// deadline stopped the model where it stood, so a length stop is true of
	// it. An upstream that had already stated its own stop finished its answer
	// and was cut only while the connection closed behind it; telling that
	// client the answer ran out of room would be false, so its own stop is what
	// travels.
	stop := llmprotocol.StopMaxTokens
	if naturalStop(engine.servedStop) {
		// A turn whose generation finished is only being rewritten at all so
		// its count can reach the client, because the error frame has nowhere
		// to put one. Where that count cannot travel anyway -- a Chat client
		// that did not ask for the usage chunk -- rewriting buys nothing and
		// costs the client the failure that named what broke, so the turn keeps
		// it. A cut mid-generation is rewritten either way: its terminal is
		// what lets the client close the stream at all.
		if !engine.truncationTerminalCarriesUsage() {
			reportUnwritableTruncationUsage(
				string(engine.targetFormat),
				"the turn finished before the transport failed and its count cannot travel on this terminal",
			)
			return events
		}
		stop = engine.servedStop
	}
	for index := range events {
		if events[index].Type != llmprotocol.EventResponseFailed {
			continue
		}
		events[index].Failure = llmprotocol.FailureTruncated
		events[index].StopReason = stop
		events[index].Truncation = engine.truncationCause
		events[index].Usage = reconciledTruncationUsage(
			engine.truncationUsage, engine.statedTerminalUsage(events[index].Usage), engine.outputSinceUsage,
		)
	}
	return events
}

// statedTerminalUsage is what the stream had said about the turn by the time it
// was cut. A terminal the decoder built carries that already; one the engine
// synthesized does not, because it never passed through the decoder's own
// terminal handling, so the running statement stands in.
func (engine *StreamEngine) statedTerminalUsage(terminal *llmprotocol.Usage) *llmprotocol.Usage {
	if terminal != nil && terminal.State == llmprotocol.UsageAvailable {
		return terminal
	}
	return engine.statedUsage
}

// reconciledTruncationUsage states a cut turn's count without contradicting
// what the stream already said.
//
// The two can disagree per count rather than wholesale. A Messages stream opens
// with the provider's authoritative input tokens and no output count at all,
// and a stream cut after its own usage frame has authoritative counts for
// everything; the Router's row is an estimate of the same turn throughout. So
// each count keeps the better evidence -- the stream's where it has any, the
// estimate only where the stream said nothing -- which is both the honest
// answer and what the stream's own merge requires: restating an authoritative
// count as an estimate is a quality decrease mergeTokenCount refuses, and
// refusing it replaces the deadline the client should be told about with a
// usage-accounting error.
func reconciledTruncationUsage(truncation, stated *llmprotocol.Usage, outputSinceStated bool) *llmprotocol.Usage {
	if truncation == nil {
		return stated
	}
	if stated == nil || stated.State != llmprotocol.UsageAvailable {
		return truncation
	}
	// A stated output count the turn then ran past is a snapshot. Anthropic's
	// message_start states output_tokens 1 before the first delta, and a turn
	// cut after hundreds of deltas was reaching the client as one token,
	// authoritative (measured live 2026-10-26). The output leg then takes the
	// larger of the snapshot and the estimate, and its splits go with it so
	// the total still adds up. Whichever wins, the leg is an estimate now:
	// the snapshot is known to be short, and the estimate is bytes over four,
	// which is low on average rather than a bound. The input leg is left as
	// stated: a prompt does not grow.
	if outputSinceStated && truncation.OutputTotal.Value != nil {
		restated := *stated
		if stated.OutputTotal.Value == nil || *truncation.OutputTotal.Value > *stated.OutputTotal.Value {
			restated.OutputTotal = truncation.OutputTotal
			restated.OutputReasoning = truncation.OutputReasoning
			restated.OutputOther = truncation.OutputOther
		} else {
			restated.OutputTotal.Provenance = llmprotocol.UsageEstimated
			restated.OutputReasoning = llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
			restated.OutputOther = llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
		}
		stated = &restated
	}
	// Only the input and output totals are reconciled, and only where the
	// stream stated none. A total has to equal the sum of its own splits
	// (validateUsageTotals), so mixing an estimated split into a stated total
	// -- or the reverse -- states a usage that contradicts itself. The splits
	// are therefore left exactly as the stream gave them, and an estimate only
	// fills a total the stream never reached.
	merged := *stated
	merged.State = llmprotocol.UsageAvailable
	if merged.InputTotal.Value == nil {
		merged.InputTotal = truncation.InputTotal
		merged.InputUncached = truncation.InputUncached
		merged.InputCacheRead = truncation.InputCacheRead
		merged.InputCacheWrite = truncation.InputCacheWrite
	}
	if merged.OutputTotal.Value == nil {
		merged.OutputTotal = truncation.OutputTotal
		merged.OutputReasoning = truncation.OutputReasoning
		merged.OutputOther = truncation.OutputOther
	}
	merged.Total = summedTruncationTotal(merged)
	return &merged
}

// summedTruncationTotal restates the overall total as the sum of the two totals
// beside it, so a usage assembled from a statement and an estimate still adds
// up. A turn missing either leg has no overall total to state.
func summedTruncationTotal(usage llmprotocol.Usage) llmprotocol.TokenCount {
	if usage.InputTotal.Value == nil || usage.OutputTotal.Value == nil {
		return llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
	}
	input, output := *usage.InputTotal.Value, *usage.OutputTotal.Value
	if input > math.MaxInt64-output {
		// Two counts that cannot be added are two counts this cannot state a
		// total for. A wrapped total is a negative one, which the usage
		// contract refuses and would fail the whole terminal.
		return llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
	}
	total := input + output
	provenance := usage.InputTotal.Provenance
	// The weaker of the two legs is what the sum is worth.
	if usageEvidenceRank(usage.OutputTotal.Provenance) < usageEvidenceRank(provenance) {
		provenance = usage.OutputTotal.Provenance
	}
	return llmprotocol.TokenCount{Value: llmprotocol.Int64(total), Provenance: provenance}
}

func (finalization streamFinalization) resultError() error {
	if finalization.firstFailure != nil {
		return finalization.firstFailure
	}
	return finalization.decodeErr
}

func containsFailedTerminal(events []llmprotocol.Event) bool {
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseFailed {
			return true
		}
	}
	return false
}

func (engine *StreamEngine) finalizeFailure(
	frames [][]byte,
	events []llmprotocol.Event,
	diagnostics llmprotocol.Diagnostics,
	cause error,
) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	engine.poison(cause)
	encoded, finalDiagnostics, finalizeErr := engine.encoder.Finalize(cause)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.maxDiagnostics)
	if finalizeErr != nil {
		return frames, events, diagnostics, finalizeErr
	}
	frames = append(frames, encoded...)
	return frames, events, diagnostics, cause
}
