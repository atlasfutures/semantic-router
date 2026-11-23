package extproc

import (
	"fmt"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// vertexAnthropicVersion is the anthropic_version Vertex AI's Anthropic
// publisher models take in the request body.
const vertexAnthropicVersion = "vertex-2023-10-16"

func providerIsVertexAI(profile *config.ProviderProfile) bool {
	return profile != nil && strings.EqualFold(strings.TrimSpace(profile.Type), "vertex-ai")
}

// adaptVertexMessagesRequest turns an Anthropic Messages body into the body
// Vertex AI's rawPredict / streamRawPredict take: the model is named in the
// URL, so the body carries no model member, and the API version moves from
// the anthropic-version header into the body as anthropic_version. Every
// other member, stream included, is sent as rendered, and the response is
// the Anthropic Messages response (an Anthropic SSE stream when streaming).
// Any other dispatch is returned unchanged.
func adaptVertexMessagesRequest(body []byte, dispatch *providerDispatch) ([]byte, error) {
	if dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		!providerIsVertexAI(dispatch.profile) {
		return body, nil
	}
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return nil, fmt.Errorf("messages body is not a JSON object")
	}
	out := body
	if gjson.GetBytes(out, "model").Exists() {
		stripped, err := sjson.DeleteBytes(out, "model")
		if err != nil {
			return nil, fmt.Errorf("remove model: %w", err)
		}
		out = stripped
	}
	out, err := sjson.SetBytes(out, "anthropic_version", vertexAnthropicVersion)
	if err != nil {
		return nil, fmt.Errorf("set anthropic_version: %w", err)
	}
	return out, nil
}

// bodyRequestsStream reports the stream flag of the body actually sent. A
// Messages body omits it when false.
func bodyRequestsStream(body []byte) bool {
	return gjson.GetBytes(body, "stream").Bool()
}

// resetVertexRequestPath sets a Vertex AI dispatch's :path from the body that
// is sent rather than from what the client asked: Vertex AI names the
// streaming operation in the URL, and a request parameter policy can drop a
// client's stream flag from the body. Any other dispatch is left alone.
func resetVertexRequestPath(mutation *ext_proc.HeaderMutation, dispatch *providerDispatch, body []byte) {
	if mutation == nil || dispatch == nil || !providerIsVertexAI(dispatch.profile) {
		return
	}
	kept := mutation.SetHeaders[:0]
	for _, option := range mutation.SetHeaders {
		if option.GetHeader().GetKey() != ":path" {
			kept = append(kept, option)
		}
	}
	mutation.SetHeaders = kept
	setProviderRequestPath(&mutation.SetHeaders, dispatch.profile, dispatch.targetFormat,
		dispatch.upstreamModel, bodyRequestsStream(body))
}
