package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// publicUpstreamError is the provider error a client is shown (semantic-router
// #222). The provider's own text can name the Router's provider account, so it
// is replaced or redacted (llmprotocol.PublicUpstreamError). The log line keeps
// what was withheld correlatable without writing provider text, which the
// content-logging policy forbids (logging/content.go).
func publicUpstreamError(
	ctx *RequestContext,
	protocolError *llmprotocol.ProtocolError,
	status int,
	path string,
) *llmprotocol.ProtocolError {
	public := llmprotocol.PublicUpstreamError(protocolError, status)
	if public == nil || (public.Message == protocolError.Message && public.Code == protocolError.Code) {
		return public
	}
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	logging.ComponentEvent("extproc", "upstream_error_redacted", map[string]interface{}{
		"request_id":        requestID,
		"path":              path,
		"upstream_status":   status,
		"upstream_category": string(protocolError.Category),
		"upstream_detail":   providerStreamFailureDetail(protocolError),
		"upstream_message":  logging.ContentDescriptor(protocolError.Message),
		"public_code":       public.Code,
	})
	return public
}
