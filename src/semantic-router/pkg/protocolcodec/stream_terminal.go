package protocolcodec

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	result := streamFinalization{
		events: events, diagnostics: diagnostics, terminalReason: reason,
		decodeErr: decodeErr, firstFailure: firstFailure,
	}
	switch {
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
	engine.pendingCompletion = nil
	if containsFailedTerminal(events) {
		return events
	}
	return append(events, llmprotocol.Event{
		Type:       llmprotocol.EventResponseFailed,
		StopReason: llmprotocol.StopError,
		Error:      streamFinalizationError(reason, "stream ended before completion"),
		Failure:    llmprotocol.FailureTransport,
	})
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
