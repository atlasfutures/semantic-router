package extproc

import (
	"bytes"
	"fmt"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// stabilizeBillingAttribution removes the per-turn fields of Claude Code's
// billing attribution line from a Messages body bound for a worker Anthropic
// does not publish. Any other body is returned unchanged.
//
// The codec drops the line for Chat and Responses targets and carries it on
// Messages, where an Anthropic host reads it as metadata. An open-weight
// worker behind OpenRouter's Messages API reads it as prompt text, and its cch
// changes on every turn, so the provider's prefix cache stops at the line: a
// 120k-token Claude Code session on glm-5.3-flash (Together) read 640 cached
// tokens per turn, and the same bodies without the per-turn fields cached
// 98%. Only the line's text changes; every other byte stays as rendered, and
// the selector reads the client's own body, so its turn signals are intact.
func (r *OpenAIRouter) stabilizeBillingAttribution(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	if dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		!bytes.Contains(body, []byte("x-anthropic-billing-header:")) ||
		r.armFamily(dispatch.logicalModel) == anthropicFamily {
		return body, nil
	}
	system := gjson.GetBytes(body, "system")
	if !system.IsArray() {
		return body, nil
	}
	out := body
	rewritten := 0
	for index, block := range system.Array() {
		text := block.Get("text")
		if block.Get("type").String() != "text" || text.Type != gjson.String {
			continue
		}
		stable, isLine := protocolcodec.StableBillingAttributionLine(text.String())
		if !isLine || stable == text.String() {
			continue
		}
		var err error
		out, err = sjson.SetBytes(out, fmt.Sprintf("system.%d.text", index), stable)
		if err != nil {
			return nil, fmt.Errorf("failed to stabilize the billing attribution line: %w", err)
		}
		rewritten++
	}
	if rewritten > 0 && ctx != nil {
		logging.ComponentEvent("extproc", "billing_attribution_stabilized", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      dispatch.logicalModel,
			"blocks":     rewritten,
		})
	}
	return out, nil
}
