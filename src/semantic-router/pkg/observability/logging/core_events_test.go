package logging

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/diagnostics"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

func TestSharedCoreEventsUseTheCloudLogger(t *testing.T) {
	core, observed := observer.New(zapcore.InfoLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(restore)
	diagnostics.ComponentEvent("protocolcodec", "thinking_marker_stripped", map[string]interface{}{"family": "anthropic", "count": 1})
	diagnostics.ComponentWarnEvent("protocolcodec", "upstream_response_field_dropped", map[string]interface{}{"field": "usage.future"})
	entries := observed.All()
	if len(entries) != 2 || entries[0].Level != zapcore.InfoLevel || entries[1].Level != zapcore.WarnLevel {
		t.Fatalf("core events lost their cloud logging levels: %+v", entries)
	}
	if entries[0].Message != "thinking_marker_stripped" || entries[0].ContextMap()["component"] != "protocolcodec" || entries[1].ContextMap()["field"] != "usage.future" {
		t.Fatalf("core diagnostic metadata changed: %+v", entries)
	}
}
