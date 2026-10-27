package extproc

import (
	"errors"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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
func emitRaylineARCTurnTiming(ctx *RequestContext, algorithm *config.AlgorithmConfig, err error) {
	fields := sealRaylineARCTurnTiming(ctx)
	if fields == nil {
		return
	}
	fields["surface"] = raylineARCTurnSurfaceHosted
	fields["outcome"] = "selected"
	if err != nil {
		fields["outcome"] = "failed"
		var failure *modelSelectionFailure
		if errors.As(err, &failure) {
			fields["failure_class"] = failure.class
		} else if algorithm != nil {
			fields["failure_class"] = authoritativeSelectionFailureClass(algorithm, err)
		}
	}
	logging.ComponentEvent("extproc", "rayline_arc_turn_timing", fields)
}

// emitRaylineARCRoutesTurnTiming logs a /v1/routes lookup's line once its
// answer is built: outcome is selected for a 200, failed otherwise, with
// the answer's status.
func emitRaylineARCRoutesTurnTiming(ctx *RequestContext, routeID string, response *ext_proc.ProcessingResponse) {
	fields := sealRaylineARCTurnTiming(ctx)
	if fields == nil {
		return
	}
	fields["surface"] = raylineARCTurnSurfaceRoutes
	if routeID != "" {
		fields["routing_decision_id"] = routeID
	}
	status := 0
	if immediate := response.GetImmediateResponse(); immediate != nil && immediate.GetStatus() != nil {
		status = int(immediate.GetStatus().GetCode())
	}
	fields["status"] = status
	fields["outcome"] = "failed"
	if status == 200 {
		fields["outcome"] = "selected"
	}
	logging.ComponentEvent("extproc", "rayline_arc_turn_timing", fields)
}

// sealRaylineARCTurnTiming is the turn's fields with total_ms and the request
// id, nil for a turn without a timing or one already logged.
func sealRaylineARCTurnTiming(ctx *RequestContext) map[string]interface{} {
	if ctx == nil || ctx.RaylineARCTurnTiming == nil {
		return nil
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
	if fields != nil {
		fields["request_id"] = ctx.RequestID
	}
	return fields
}
