package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// thinkingMarkerFamily is the family whose marker signs unsigned thinking on
// this turn's Messages output (semantic-router #191), or empty when the
// Router mints none: minting is off, or the client does not speak Messages.
// The family names the served worker, so the request pre-pass can log where
// a resent marker came from.
func (r *OpenAIRouter) thinkingMarkerFamily(ctx *RequestContext) string {
	if r == nil || r.Config == nil || !r.Config.ThinkingMarkerMint || ctx == nil ||
		ctx.SourceFormat != llmprotocol.AnthropicMessagesV1 {
		return ""
	}
	model := ctx.VSRSelectedModel
	if model == "" {
		model = ctx.RequestModel
	}
	return protocolcodec.ThinkingMarkerFamily(model)
}
