package extproc

import (
	"fmt"
	"net/http"
	"sort"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// vertexAnthropicVersion is the anthropic_version Vertex AI's Anthropic
// publisher models take in the request body.
const vertexAnthropicVersion = "vertex-2023-10-16"

// vertexMessagesFields are the top-level Messages members Vertex AI's
// rawPredict / streamRawPredict accept without a beta. Vertex AI rejects any
// other member ("Extra inputs are not permitted"), and a beta-gated member
// such as context_management is rejected unless its beta arrives in the
// anthropic-beta header, which a Vertex AI dispatch does not forward from
// the client (vertexAnthropicBeta). The list is an allowlist, so a member a
// client adds later is dropped rather than failing the call.
var vertexMessagesFields = map[string]struct{}{
	"anthropic_version": {},
	"messages":          {},
	"max_tokens":        {},
	"system":            {},
	"metadata":          {},
	"stop_sequences":    {},
	"stream":            {},
	"temperature":       {},
	"top_p":             {},
	"top_k":             {},
	"tools":             {},
	"tool_choice":       {},
	"thinking":          {},
	"output_config":     {},
	"cache_control":     {},
}

func providerIsVertexAI(profile *config.ProviderProfile) bool {
	return profile != nil && strings.EqualFold(strings.TrimSpace(profile.Type), "vertex-ai")
}

func dispatchIsVertexMessages(dispatch *providerDispatch) bool {
	return dispatch != nil && dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 &&
		providerIsVertexAI(dispatch.profile)
}

// adaptVertexMessagesRequest turns an Anthropic Messages body into the body
// Vertex AI's rawPredict / streamRawPredict take: the model is named in the
// URL, so the body carries no model member; the API version moves from the
// anthropic-version header into the body as anthropic_version; and every
// top-level member outside vertexMessagesFields is dropped. It returns the
// names of the dropped members (model and anthropic_version excepted),
// sorted. The response is the Anthropic Messages response (an Anthropic SSE
// stream when streaming). Any other dispatch is returned unchanged.
func adaptVertexMessagesRequest(body []byte, dispatch *providerDispatch) ([]byte, []string, error) {
	if !dispatchIsVertexMessages(dispatch) {
		return body, nil, nil
	}
	if !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return nil, nil, fmt.Errorf("messages body is not a JSON object")
	}
	var remove, dropped []string
	gjson.ParseBytes(body).ForEach(func(key, _ gjson.Result) bool {
		name := key.String()
		if name == "model" {
			remove = append(remove, name)
		} else if _, accepted := vertexMessagesFields[name]; !accepted {
			remove = append(remove, name)
			dropped = append(dropped, name)
		}
		return true
	})
	out := body
	for _, name := range remove {
		stripped, err := sjson.DeleteBytes(out, gjsonEscapePath(name))
		if err != nil {
			return nil, nil, fmt.Errorf("remove %s: %w", name, err)
		}
		out = stripped
	}
	out, err := sjson.SetBytes(out, "anthropic_version", vertexAnthropicVersion)
	if err != nil {
		return nil, nil, fmt.Errorf("set anthropic_version: %w", err)
	}
	sort.Strings(dropped)
	return out, dropped, nil
}

// gjsonEscapePath escapes a member name for use as a one-level gjson/sjson
// path, so a name with a path character is deleted as itself.
func gjsonEscapePath(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch r {
		case '.', '*', '?', '|', '#', '@', '\\', '!', '=', '<', '>', '%', ':':
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

// dispatchedAnthropicBeta is the anthropic-beta the dispatch itself sets (a
// provider profile's header or a decision's header_mutation), the last one
// set, read before anything merges the client's value into it.
func dispatchedAnthropicBeta(setHeaders []*core.HeaderValueOption) string {
	value := ""
	for _, header := range setHeaders {
		if header.GetHeader() != nil && strings.EqualFold(header.Header.Key, anthropicBetaHeader) {
			value = string(header.Header.RawValue)
			if value == "" {
				value = header.Header.Value
			}
		}
	}
	return value
}

// vertexAnthropicBeta is the anthropic-beta a Vertex AI Messages dispatch
// sends: the betas the dispatch sets itself, plus the extended-cache-ttl
// beta when the Router generated a 1h cache ttl, and never the client's.
// The body keeps no beta-gated member (vertexMessagesFields), so a client's
// beta has nothing to enable, and Vertex AI refuses a beta name it does not
// know, which a client written for the Anthropic API can send. On an ARC
// turn the server-side fallback betas are taken out, as on every dispatch.
func vertexAnthropicBeta(own string, ctx *RequestContext) string {
	value := own
	if auto := ctx.DispatchAutoCache; auto != nil && auto.ttl == anthropicLongestCacheTTL {
		value = mergeAnthropicBetas(value, extendedCacheTTLBeta)
	}
	if ctx.VSRRaylineARC != nil {
		value, _ = withoutServerSideFallbackBetas(value)
	}
	return mergeAnthropicBetas(value, "")
}

// limitVertexAnthropicBeta replaces whatever anthropic-beta a Vertex AI
// Messages dispatch would send with vertexAnthropicBeta's value, removing
// the header when that is empty, so the client's header is not forwarded.
// own is dispatchedAnthropicBeta, read before the beta merges ran. Any other
// dispatch is left alone.
func limitVertexAnthropicBeta(mutation *ext_proc.HeaderMutation, dispatch *providerDispatch, ctx *RequestContext, own string) {
	if mutation == nil || !dispatchIsVertexMessages(dispatch) {
		return
	}
	kept := mutation.SetHeaders[:0]
	for _, option := range mutation.SetHeaders {
		if option.GetHeader() == nil || !strings.EqualFold(option.Header.Key, anthropicBetaHeader) {
			kept = append(kept, option)
		}
	}
	mutation.SetHeaders = kept
	if value := vertexAnthropicBeta(own, ctx); value != "" {
		mutation.SetHeaders = append(mutation.SetHeaders, overwriteRequestHeader(anthropicBetaHeader, value))
		return
	}
	for _, name := range mutation.RemoveHeaders {
		if strings.EqualFold(name, anthropicBetaHeader) {
			return
		}
	}
	mutation.RemoveHeaders = append(mutation.RemoveHeaders, anthropicBetaHeader)
}

// limitVertexAnthropicBetaHeader is limitVertexAnthropicBeta for a re-send's
// HTTP headers.
func limitVertexAnthropicBetaHeader(headers http.Header, dispatch *providerDispatch, ctx *RequestContext, own string) {
	if !dispatchIsVertexMessages(dispatch) {
		return
	}
	headers.Del(anthropicBetaHeader)
	if value := vertexAnthropicBeta(own, ctx); value != "" {
		headers.Set(anthropicBetaHeader, value)
	}
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
