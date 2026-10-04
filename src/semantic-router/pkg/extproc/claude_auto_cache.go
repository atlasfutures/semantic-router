package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Where an automatic-cache directive the Router supplies for a Claude worker
// comes from: the client named a cache shard with prompt_cache_key, or it
// stated nothing and relies on the automatic prefix caching an OpenAI model
// gives it by default.
const (
	autoCacheSourcePromptCacheKey = "prompt_cache_key"
	autoCacheSourceDefault        = "default"
)

// dispatchAutoCache is the automatic-cache directive the Router supplies for
// one dispatch, and why. encodeDispatchRequest sets it on the dispatched copy
// of the request only, so the client's own request -- which the response
// cache and retained Responses state read -- keeps what the client sent.
type dispatchAutoCache struct {
	directive llmprotocol.CacheDirective
	source    string
}

// claudeAutoCache decides whether a turn going to a Claude worker over
// Messages gets the automatic-cache directive the client did not state.
//
// An OpenAI-shaped client (Chat or Responses) sends no cache_control: it
// relies on the prefix caching an OpenAI model applies on its own, and may
// name a shard with prompt_cache_key. Anthropic caches only at explicit
// breakpoints, so the same turn routed to Claude would be billed uncached
// every time. The directive makes the Messages codec place the breakpoint
// automatic caching would (placeAutoCacheBreakpoint), as OpenRouter's
// automatic-caching mode does, so the client's cache intent is not silently
// lost.
//
// It applies only where the family and the wire are both known, which is
// here: a non-Claude worker and a Chat or Responses target get nothing (a
// Claude worker reached over Chat is a separate gap). A Messages client is
// left as it is: it caches explicitly, or chooses not to. A client's own
// top-level directive wins, and a client's per-block breakpoints are kept by
// the placement, which adds one only when the last cacheable block holds
// none and the four-breakpoint limit leaves room. No client format defines an
// opt-out of automatic caching, so there is none to honour. Nor does the
// Router configuration have a prompt-cache setting to hang a switch on; the
// rule is unconditional, as the caching the client gets from an OpenAI model
// is.
//
// The directive is constant and its placement depends only on the request,
// so the same request dispatches the same bytes and a growing conversation
// keeps a cache-stable prefix across turns.
func (r *OpenAIRouter) claudeAutoCache(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	source llmprotocol.WireFormat,
) *dispatchAutoCache {
	if request == nil || dispatch == nil || request.AutoCache != nil ||
		dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		(source != llmprotocol.OpenAIChatV1 && source != llmprotocol.OpenAIResponsesV1) ||
		r.armFamily(dispatch.logicalModel) != anthropicFamily {
		return nil
	}
	origin := autoCacheSourceDefault
	// The codecs carry prompt_cache_key only when it holds a value.
	if carrier := request.Unmodeled; carrier != nil {
		if _, named := carrier.Fields["prompt_cache_key"]; named {
			origin = autoCacheSourcePromptCacheKey
		}
	}
	return &dispatchAutoCache{
		directive: llmprotocol.CacheDirective{Type: "ephemeral"},
		source:    origin,
	}
}

// applyDispatchAutoCache sets the directive claudeAutoCache planned for this
// dispatch on the dispatched copy, and records a diagnostic naming its
// source, which the translation-warning counter, the protocol warnings header
// and the debug log all read. The codec's placement then adds its own
// diagnostic for where the breakpoint went, or why none was added. A
// Messages client is never given the directive, so the replay of its own
// bytes is untouched.
func applyDispatchAutoCache(request *llmprotocol.Request, ctx *RequestContext) {
	auto := ctx.DispatchAutoCache
	if auto == nil {
		return
	}
	directive := auto.directive
	request.AutoCache = &directive
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
		Source: ctx.SourceFormat,
		Target: llmprotocol.AnthropicMessagesV1,
		Field:  "cache_control",
		Action: llmprotocol.DiagnosticGenerated,
		Reason: "automatic_cache_" + auto.source,
	})
}
