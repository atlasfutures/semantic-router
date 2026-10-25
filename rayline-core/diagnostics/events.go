// Package diagnostics connects execution-core events to its host's logger.
package diagnostics

import (
	"context"
	"log/slog"
	"sync/atomic"
)

// EventSink receives metadata-only events. Implementations must be safe for
// concurrent calls and must not retain or mutate fields.
type EventSink func(level slog.Level, component, event string, fields map[string]interface{})

var eventSink atomic.Pointer[EventSink]

// SetEventSink installs the host's logger. Before installation the core emits
// events through the standard library logger.
func SetEventSink(sink EventSink) {
	if sink == nil {
		panic("diagnostics: event sink must not be nil")
	}
	eventSink.Store(&sink)
}

func emit(level slog.Level, component, event string, fields map[string]interface{}) {
	if sink := eventSink.Load(); sink != nil {
		(*sink)(level, component, event, fields)
		return
	}
	args := make([]any, 0, 2+2*len(fields))
	args = append(args, "component", component)
	for key, value := range fields {
		args = append(args, key, value)
	}
	slog.Log(context.Background(), level, event, args...)
}

func ComponentEvent(component, event string, fields map[string]interface{}) {
	emit(slog.LevelInfo, component, event, fields)
}

func ComponentWarnEvent(component, event string, fields map[string]interface{}) {
	emit(slog.LevelWarn, component, event, fields)
}
