package extproc

import (
	"net/http"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// The helpers in this file were appended to processor_req_body.go and
// processor_req_body_routing.go on the fork. They are pure additions and are
// kept here so no upstream file carries them.
//
// TODO(vsr-next Bucket B): the callers that used to invoke these from the
// auto-routing dispatch path no longer exist upstream. Re-seat the fail-closed
// dispatch response and the artifact credential header on
// prepareProviderDispatch / appendProviderCredential, or delete them if VSR
// owns dispatch end to end.

func (r *OpenAIRouter) raylineARCDispatchFailureResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	return r.raylineARCDispatchFailureResponseFor(
		ctx,
		"request_shape",
	)
}

func (r *OpenAIRouter) raylineARCDispatchFailureResponseFor(
	ctx *RequestContext,
	failureClass string,
) *ext_proc.ProcessingResponse {
	logging.ComponentErrorEvent(
		"extproc",
		"rayline_arc_dispatch_failed",
		map[string]interface{}{
			"request_id":    ctx.RequestID,
			"failure_class": failureClass,
		},
	)
	metrics.RecordRaylineARCFailure("dispatch_" + failureClass)
	r.finalizeRaylineARCAbort(ctx, "dispatch_"+failureClass)
	return r.createErrorResponse(
		http.StatusServiceUnavailable,
		"Rayline ARC routing unavailable",
	)
}

//nolint:unused // TODO(vsr-next Bucket B): caller was resolveAutoRoutingTarget in the deleted dispatch path.
func (r *OpenAIRouter) selectionDispatchFailureResponseFor(
	ctx *RequestContext,
	failureClass string,
) *ext_proc.ProcessingResponse {
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	logging.ComponentErrorEvent(
		"extproc",
		configRaylineARC+"_dispatch_failed",
		map[string]interface{}{
			"request_id":    requestID,
			"failure_class": failureClass,
		},
	)
	metrics.RecordRaylineARCFailure("dispatch_" + failureClass)
	finalizeSelectionAbort(ctx, "dispatch_"+failureClass)
	return r.createErrorResponse(
		http.StatusServiceUnavailable,
		selectionUnavailableMessage(ctx),
	)
}

func raylineARCDispatchAllowed(ctx *RequestContext) bool {
	return ctx != nil &&
		ctx.RaylineARCTransaction != nil &&
		ctx.RaylineARCTransaction.dispatchAllowed()
}

// appendRaylineARCCredentialHeader injects the artifact-owned provider
// credential for the selected worker and overwrites any inbound value so a
// caller-supplied key can never reach the provider.
func (r *OpenAIRouter) appendRaylineARCCredentialHeader(
	state *routeHeaderState,
	model string,
	authHeader string,
	authPrefix string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	accessKey := ""
	for _, endpoint := range r.Config.GetEndpointsForModel(model) {
		if endpoint.APIKeyEnvName == ctx.RaylineARCDispatch.APIKeyEnv {
			accessKey = endpoint.APIKey
			break
		}
	}
	if accessKey == "" {
		return r.raylineARCDispatchFailureResponse(ctx)
	}
	value := accessKey
	if authPrefix != "" {
		value = authPrefix + " " + accessKey
	}
	state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      authHeader,
			RawValue: []byte(value),
		},
		// Explicit overwrite: Envoy's default append action would combine
		// this with any inbound value of the same header.
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
	ctx.RaylineARCAuthHeader = authHeader
	logging.ComponentDebugEvent("extproc", "provider_auth_injected", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       model,
		"header_name": authHeader,
		"source":      "rayline_arc_artifact",
	})
	return nil
}

// enforceRaylineARCCredentialHeader keeps exactly one value for the artifact
// credential header and drops any deletion that would strip it.
func enforceRaylineARCCredentialHeader(
	state *routeHeaderState,
	ctx *RequestContext,
) {
	if ctx == nil || ctx.RaylineARCDispatch == nil || state == nil {
		return
	}
	authHeader := ctx.RaylineARCAuthHeader
	if authHeader == "" {
		return
	}
	artifactIndex := -1
	for index, option := range state.setHeaders {
		if strings.EqualFold(option.GetHeader().GetKey(), authHeader) {
			artifactIndex = index
			break
		}
	}
	retained := make([]*core.HeaderValueOption, 0, len(state.setHeaders))
	for index, option := range state.setHeaders {
		if index != artifactIndex &&
			strings.EqualFold(option.GetHeader().GetKey(), authHeader) {
			continue
		}
		retained = append(retained, option)
	}
	state.setHeaders = retained
	state.removeHeaders = removeARCAuthHeaderDeletions(
		state.removeHeaders,
		authHeader,
	)
}

func removeARCAuthHeaderDeletions(removeHeaders []string, authHeader string) []string {
	retained := make([]string, 0, len(removeHeaders))
	for _, name := range removeHeaders {
		if strings.EqualFold(name, authHeader) {
			continue
		}
		retained = append(retained, name)
	}
	return retained
}
