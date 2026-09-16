package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func (r *OpenAIRouter) validateRequestHeaders(method string, path string) *ext_proc.ProcessingResponse {
	normalizedPath := normalizeRequestPath(path)

	switch normalizedPath {
	case "/v1/chat/completions":
		return validateAllowedMethod(r, method, "POST")
	case "/v1/messages":
		return validateAllowedMethod(r, method, "POST")
	case "/v1/models":
		return validateAllowedMethod(r, method, "GET")
	case "/v1/responses":
		return r.validateResponseAPICollectionMethod(method)
	case raylineRoutesAPIPath:
		return r.validateRaylineRoutesMethod(method)
	}

	if extractResponseIDFromInputItemsPath(normalizedPath) != "" {
		return validateAllowedMethod(r, method, "GET")
	}

	if extractResponseIDFromPath(normalizedPath) != "" {
		return r.validateResponseAPIItemMethod(method)
	}

	if normalizedPath == routerReplayAPIBasePath || strings.HasPrefix(normalizedPath, routerReplayAPIBasePath+"/") {
		return r.createErrorResponse(404, "endpoint not found")
	}

	if normalizedPath == "/v1" || strings.HasPrefix(normalizedPath, "/v1/") {
		return r.createErrorResponse(404, "endpoint not found")
	}

	return nil
}

// validateRaylineRoutesMethod keeps the endpoint invisible where it is not
// configured: a disabled cell answers 404 for every method, so probing it
// cannot distinguish "off here" from "never existed".
func (r *OpenAIRouter) validateRaylineRoutesMethod(method string) *ext_proc.ProcessingResponse {
	if !r.raylineRoutesAPIEnabled() {
		return r.createErrorResponse(404, "endpoint not found")
	}
	return validateAllowedMethod(r, method, "POST")
}

func (r *OpenAIRouter) validateResponseAPICollectionMethod(method string) *ext_proc.ProcessingResponse {
	if r.ResponseAPIFilter == nil || !r.ResponseAPIFilter.IsEnabled() {
		return r.createErrorResponse(404, "endpoint not found")
	}

	return validateAllowedMethod(r, method, "POST")
}

func (r *OpenAIRouter) validateResponseAPIItemMethod(method string) *ext_proc.ProcessingResponse {
	if method == "GET" || method == "DELETE" {
		return nil
	}

	return r.createErrorResponse(405, "method not allowed")
}

func validateAllowedMethod(r *OpenAIRouter, method string, allowed string) *ext_proc.ProcessingResponse {
	if method == allowed {
		return nil
	}
	return r.createErrorResponse(405, "method not allowed")
}

func normalizeRequestPath(path string) string {
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}
	if len(path) > 1 {
		path = strings.TrimSuffix(path, "/")
	}
	return path
}
