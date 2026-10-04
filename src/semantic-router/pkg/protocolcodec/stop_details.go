package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// decodeStopDetails reads a provider's stop_details: Anthropic's structured
// stop reason, {"type":"refusal","category":"cyber","explanation":...}, which
// OpenRouter may pass through. Only the type and category are kept; the
// explanation is prose the Router does not need. An absent, null or
// unreadable value has none: telemetry never fails a turn.
func decodeStopDetails(raw json.RawMessage) *llmprotocol.StopDetails {
	if !hasJSONValue(raw) {
		return nil
	}
	var wire struct {
		Type     string `json:"type"`
		Category string `json:"category"`
	}
	if json.Unmarshal(raw, &wire) != nil || (wire.Type == "" && wire.Category == "") {
		return nil
	}
	return &llmprotocol.StopDetails{Type: boundStopTelemetry(wire.Type), Category: boundStopTelemetry(wire.Category)}
}

// stopTelemetryBytes bounds each stop telemetry string. They are short
// identifiers; a longer value is cut rather than failing the turn.
const stopTelemetryBytes = 64

func boundStopTelemetry(value string) string {
	if len(value) <= stopTelemetryBytes {
		return value
	}
	return strings.ToValidUTF8(value[:stopTelemetryBytes], "")
}
