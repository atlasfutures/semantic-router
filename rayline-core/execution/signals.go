package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Typed signals (pathfinder docs/plans/2026-10-25-decisions-topic-signal-contract.md)
// are caller-facing metadata: they change no route. A caller opts in with
// routing.signals, an array of signal names, which the core passes to Decisions
// unchanged; the response's signals block returns to the caller in the
// x-vsr-signals header, as cloud VSR carries it on generation turns.
const signalsHeader = "X-Vsr-Signals"

const maxSignalNames = 32

// validateSignalNames accepts an array of nonempty signal-name strings. The
// service owns the vocabulary (signal_not_supported / signal_not_admitted).
func validateSignalNames(raw json.RawMessage) error {
	var names []string
	if err := json.Unmarshal(raw, &names); err != nil || names == nil || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return fmt.Errorf("routing.signals must be an array of signal names")
	}
	if len(names) > maxSignalNames {
		return fmt.Errorf("routing.signals names at most %d signals", maxSignalNames)
	}
	for _, name := range names {
		if name == "" || len(name) > 128 {
			return fmt.Errorf("routing.signals names must be nonempty strings of at most 128 bytes")
		}
	}
	return nil
}

// extractRoutingSignals removes a request's routing member when it carries
// only signals, and returns the signal names' bytes unchanged. A body without
// routing is returned byte-identical. A routing member with any other key is
// left in place, where Decide refuses it as before: per-request routing
// preferences still belong in the execution config.
func extractRoutingSignals(body []byte) ([]byte, json.RawMessage, error) {
	if !bytes.Contains(body, []byte(`"routing"`)) {
		return body, nil, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return body, nil, nil
	}
	raw, exists := fields["routing"]
	if !exists {
		return body, nil, nil
	}
	var routing map[string]json.RawMessage
	if err := json.Unmarshal(raw, &routing); err != nil || len(routing) != 1 {
		return body, nil, nil
	}
	signals, exists := routing["signals"]
	if !exists {
		return body, nil, nil
	}
	if err := validateSignalNames(signals); err != nil {
		return nil, nil, err
	}
	delete(fields, "routing")
	stripped, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, err
	}
	return stripped, signals, nil
}

// signalsHeaderValue is the compact JSON of a signals block, or "" when the
// block is absent.
func signalsHeaderValue(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, trimmed); err != nil {
		return ""
	}
	return compact.String()
}
