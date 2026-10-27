package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The surfaces a rayline_arc_turn_timing line names.
const (
	raylineARCTurnSurfaceHosted = "hosted"
	raylineARCTurnSurfaceRoutes = "routes"
)

// startRaylineARCTurnTiming opens the turn's timing when its body arrives:
// headers_to_body is the request headers to the whole body, as ext_proc saw
// them. It only measures; nothing reads it but the timing line.
func startRaylineARCTurnTiming(ctx *RequestContext, bodyBytes int) {
	if ctx == nil {
		return
	}
	timing := raylinearc.NewTurnTiming()
	if !ctx.StartTime.IsZero() && !ctx.ProcessingStartTime.IsZero() {
		timing.Set(raylinearc.SpanHeadersToBody, ctx.ProcessingStartTime.Sub(ctx.StartTime))
	}
	timing.SetBodyBytes(bodyBytes)
	ctx.RaylineARCTurnTiming = timing
}

func raylineARCTurnTimingOf(ctx *RequestContext) *raylinearc.TurnTiming {
	if ctx == nil {
		return nil
	}
	return ctx.RaylineARCTurnTiming
}

// emitRaylineARCTurnTiming logs the turn's rayline_arc_turn_timing line once,
// when its routing ends, whether it selected or failed: names and numbers
// only (hashed ids, bounded classes, spans and byte counts). total_ms runs
// from the request headers (the body's arrival when they were not seen).
// A later call for the same turn logs nothing.
func emitRaylineARCTurnTiming(
	ctx *RequestContext,
	surface, routeID string,
	algorithm *config.AlgorithmConfig,
	err error,
) {
	if ctx == nil || ctx.RaylineARCTurnTiming == nil {
		return
	}
	started := ctx.StartTime
	if started.IsZero() {
		started = ctx.ProcessingStartTime
	}
	var total time.Duration
	if !started.IsZero() {
		total = time.Since(started)
	}
	fields := ctx.RaylineARCTurnTiming.Seal(total)
	if fields == nil {
		return
	}
	fields["request_id"] = ctx.RequestID
	fields["surface"] = surface
	if routeID != "" {
		fields["routing_decision_id"] = routeID
	}
	fields["outcome"] = "selected"
	if err != nil {
		fields["outcome"] = "failed"
		if algorithm != nil {
			fields["failure_class"] = authoritativeSelectionFailureClass(algorithm, err)
		}
	}
	logging.ComponentEvent("extproc", "rayline_arc_turn_timing", fields)
}
