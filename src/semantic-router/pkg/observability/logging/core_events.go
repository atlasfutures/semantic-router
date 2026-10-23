package logging

import (
	"log/slog"

	"github.com/atlasfutures/semantic-router/rayline-core/diagnostics"
)

func init() {
	diagnostics.SetEventSink(func(level slog.Level, component, event string, fields map[string]interface{}) {
		if level >= slog.LevelWarn {
			ComponentWarnEvent(component, event, fields)
			return
		}
		ComponentEvent(component, event, fields)
	})
}
