package extproc

import (
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// publicUpstreamError is the provider error a client is shown (semantic-router
// #222). The provider's own text can name the Router's provider account, so it
// is replaced or redacted (llmprotocol.PublicUpstreamError). The log line keeps
// what was withheld correlatable without writing provider text, which the
// content-logging policy forbids (logging/content.go). The descriptor is taken
// after redaction: a credential the provider echoed must not be logged even as
// a hash.
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
	redacted := llmprotocol.RedactProviderText(protocolError.Message)
	fields := map[string]interface{}{
		"request_id":        requestID,
		"path":              path,
		"upstream_status":   status,
		"upstream_category": string(protocolError.Category),
		"upstream_detail":   providerStreamFailureDetail(protocolError),
		"upstream_message":  logging.ContentDescriptor(redacted),
		"public_code":       public.Code,
	}
	// Which request member the provider rejected, as contract paths only
	// (upstreamErrorFields): the message itself stays withheld.
	if named := upstreamErrorFields(redacted); len(named) > 0 {
		fields["upstream_error_fields"] = named
	}
	logging.ComponentEvent("extproc", "upstream_error_redacted", fields)
	return public
}
