package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Event kinds and bases of the execution-context v2 contract. The harness
// adapter states what just happened; the core never upgrades an unknown turn
// into a switching boundary.
const (
	EventTaskStart     = "task_start"
	EventUserMessage   = "user_message"
	EventToolResult    = "tool_result"
	EventCompaction    = "compaction"
	EventSubagentStart = "subagent_start"
	EventSidecall      = "sidecall"

	BasisExplicitClient = "explicit_client"
	BasisMarkerInferred = "marker_inferred"
	BasisUnknown        = "unknown"
)

// eventHeader carries an explicit event from a harness adapter that can set
// per-request headers. eventBodyField carries it from one that can only edit
// the provider payload (Pi's before_provider_request hook). The core removes
// the body field before decoding; it never reaches a provider or Decisions.
const (
	eventHeader    = "X-Rayline-Event"
	eventBodyField = "rayline_event"
)

// CompactionEvent indexes the request body as sent. RetainedFrom names the
// first retained message after the summary (Pi's retained tail), or null.
type CompactionEvent struct {
	SummaryMessage int  `json:"summary_message"`
	RetainedFrom   *int `json:"retained_from"`
	New            bool `json:"new"`
}

// ExecutionEvent is the v2 envelope's event. Kind is nil exactly when Basis is
// unknown.
type ExecutionEvent struct {
	Kind       *string          `json:"kind"`
	Basis      string           `json:"basis"`
	Compaction *CompactionEvent `json:"compaction"`
}

func (event ExecutionEvent) is(kind string) bool { return event.Kind != nil && *event.Kind == kind }

// kindName is for logs; an unknown kind logs as "unknown".
func (event ExecutionEvent) kindName() string {
	if event.Kind == nil {
		return "unknown"
	}
	return *event.Kind
}

// extractExplicitEvent removes the body field (if any) and decodes the
// explicit event from either carrier. Both carriers at once are refused.
func extractExplicitEvent(header http.Header, body []byte) ([]byte, *ExecutionEvent, error) {
	var raw json.RawMessage
	if value := header.Get(eventHeader); value != "" {
		raw = json.RawMessage(value)
	}
	if bytes.Contains(body, []byte(`"`+eventBodyField+`"`)) {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return nil, nil, fmt.Errorf("invalid request body")
		}
		if field, exists := fields[eventBodyField]; exists {
			if raw != nil {
				return nil, nil, fmt.Errorf("execution event supplied in both header and body")
			}
			raw = field
			delete(fields, eventBodyField)
			stripped, err := json.Marshal(fields)
			if err != nil {
				return nil, nil, err
			}
			body = stripped
		}
	}
	if raw == nil {
		return body, nil, nil
	}
	var wire struct {
		Kind       *string          `json:"kind"`
		Compaction *CompactionEvent `json:"compaction"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil || wire.Kind == nil {
		return nil, nil, fmt.Errorf("execution event must be an object with a known kind")
	}
	switch *wire.Kind {
	case EventTaskStart, EventUserMessage, EventToolResult, EventSubagentStart, EventSidecall:
		if wire.Compaction != nil {
			return nil, nil, fmt.Errorf("execution event compaction is only valid for kind compaction")
		}
	case EventCompaction:
		if wire.Compaction == nil {
			return nil, nil, fmt.Errorf("compaction event requires summary_message, retained_from and new")
		}
	default:
		return nil, nil, fmt.Errorf("unknown execution event kind %q", *wire.Kind)
	}
	return body, &ExecutionEvent{Kind: wire.Kind, Basis: BasisExplicitClient, Compaction: wire.Compaction}, nil
}

// validateCompactionIndexes checks indexes against the wire message array.
func validateCompactionIndexes(event *ExecutionEvent, roles []string) error {
	if event == nil || event.Compaction == nil {
		return nil
	}
	compaction := event.Compaction
	if roles == nil || compaction.SummaryMessage < 0 || compaction.SummaryMessage >= len(roles) {
		return fmt.Errorf("compaction summary_message is outside the request's messages")
	}
	if compaction.RetainedFrom != nil && (*compaction.RetainedFrom <= compaction.SummaryMessage || *compaction.RetainedFrom >= len(roles)) {
		return fmt.Errorf("compaction retained_from must follow the summary inside the request's messages")
	}
	return nil
}

// inferEvent reads only documented wire markers. A retained-tail compaction is
// never inferred: it requires the harness's explicit event.
func inferEvent(request llmprotocol.Request, freshBranch bool, branch string) ExecutionEvent {
	kind := func(value string) ExecutionEvent {
		return ExecutionEvent{Kind: &value, Basis: BasisMarkerInferred}
	}
	if freshBranch && !hasExecutionHistory(request) {
		if branch != "main" {
			return kind(EventSubagentStart)
		}
		return kind(EventTaskStart)
	}
	if len(request.Messages) == 0 {
		return ExecutionEvent{Basis: BasisUnknown}
	}
	last := request.Messages[len(request.Messages)-1]
	switch last.Role {
	case llmprotocol.RoleTool:
		return kind(EventToolResult)
	case llmprotocol.RoleUser:
		// Text beside Anthropic tool results stays in the tool-result wire group.
		if last.WireGroup != 0 {
			return kind(EventToolResult)
		}
		for _, content := range last.Content {
			if content.Kind == llmprotocol.ContentToolResult {
				return kind(EventToolResult)
			}
		}
		return kind(EventUserMessage)
	}
	return ExecutionEvent{Basis: BasisUnknown}
}

// wireMessageFingerprints identifies each wire message independently of
// moving cache directives, so a compaction's retained tail can be matched to
// the messages it carried over.
func wireMessageFingerprints(format llmprotocol.WireFormat, body []byte) []string {
	var wire map[string]json.RawMessage
	if json.Unmarshal(body, &wire) != nil {
		return nil
	}
	field := "messages"
	if format == llmprotocol.OpenAIResponsesV1 {
		field = "input"
	}
	var messages []any
	if json.Unmarshal(wire[field], &messages) != nil {
		return nil
	}
	result := make([]string, len(messages))
	for index, message := range messages {
		encoded, err := json.Marshal(withoutCacheDirectives(message))
		if err != nil {
			return nil
		}
		result[index] = fingerprint(encoded)
	}
	return result
}

func withoutCacheDirectives(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, item := range typed {
			if key != "cache_control" {
				result[key] = withoutCacheDirectives(item)
			}
		}
		return result
	case []any:
		result := make([]any, len(typed))
		for index, item := range typed {
			result[index] = withoutCacheDirectives(item)
		}
		return result
	}
	return value
}

// remapRetainedAttribution moves attribution for retained messages to their
// new positions. Only an explicit compaction with a retained tail can do this;
// a match is by content, in order, never by position alone. previous holds the
// last committed request's wire messages; the reply that request produced is
// attributed at index len(previous), so it maps to the message following the
// retained copy of previous's last message when that message is an assistant's.
func remapRetainedAttribution(ledger []attributedTurn, previous, current, roles []string, format llmprotocol.WireFormat, compaction *CompactionEvent) []attributedTurn {
	if compaction == nil || compaction.RetainedFrom == nil || len(previous) == 0 {
		return nil
	}
	moved := make(map[int]int)
	cursor := 0
	for index := *compaction.RetainedFrom; index < len(current); index++ {
		for search := cursor; search < len(previous); search++ {
			if previous[search] == current[index] {
				moved[search] = index
				cursor = search + 1
				break
			}
		}
	}
	if last, ok := moved[len(previous)-1]; ok && last+1 < len(roles) && roles[last+1] == "assistant" {
		moved[len(previous)] = last + 1
	}
	var result []attributedTurn
	for _, turn := range ledger {
		if position, ok := moved[turn.entry.Message]; ok && turn.format == format && roles[position] == "assistant" {
			turn.entry.Message = position
			result = append(result, turn)
		}
	}
	return result
}
