package extproc

import (
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// logIngressProtocolError writes the one record of a refused request that
// survives it. The client is told only what its request got wrong in general
// terms, the body is never stored, and the counters carry no member name, so
// without this line a 400 at the protocol boundary is undiagnosable after the
// fact. The cause carries member names, Go types and JSON offsets and never a
// decoded value, which is what makes it safe to write down.
func logIngressProtocolError(ctx *RequestContext, err error) {
	fields := map[string]interface{}{"code": "invalid_request"}
	if ctx != nil {
		fields["request_id"] = ctx.RequestID
		fields["format"] = string(ctx.SourceFormat)
		// The beta set is the effective protocol version of the refused
		// request. Without it a refusal names a member and leaves the reader
		// to guess which opt-in introduced it.
		if betas := anthropicBetaSet(ctx.Headers); len(betas) > 0 {
			fields["anthropic_betas"] = strings.Join(betas, ",")
		}
	}
	var protocolError *llmprotocol.ProtocolError
	if errors.As(err, &protocolError) {
		fields["code"] = protocolError.Code
		fields["category"] = string(protocolError.Category)
		if protocolError.Cause != nil {
			fields["detail"] = protocolError.Cause.Error()
		}
	}
	logging.ComponentWarnEvent("extproc", "ingress_request_refused", fields)
}
