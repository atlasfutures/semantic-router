package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Where an automatic-cache directive the Router supplies for a Claude worker
// comes from: the client named a cache shard with prompt_cache_key, or it
// stated nothing and relies on the automatic prefix caching an OpenAI model
// gives it by default.
const (
	autoCacheSourcePromptCacheKey = "prompt_cache_key"
	autoCacheSourceDefault        = "default"
)

// dispatchAutoCache marks a dispatch that may be given the automatic-cache
// directive: a Claude worker over Messages, reached from an OpenAI-shaped
// client. Whether it is given one is decided at encode, on the request that
// is actually dispatched (applyDispatchAutoCache).
type dispatchAutoCache struct {
	logicalModel string
}

// claudeAutoCacheDispatch marks the dispatches the automatic-cache rule
// covers. It runs in finalizeProviderDispatchResponse, which every dispatch
// path reaches with the worker's family and the wire known.
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
// A non-Claude worker and a Chat or Responses target are not covered (a
// Claude worker reached over Chat is a separate gap). A Messages client is
// left as it is: it caches explicitly, or chooses not to. No client format
// defines an opt-out of automatic caching, so there is none to honour. Nor
// does the Router configuration have a prompt-cache setting to hang a switch
// on; the rule is unconditional, as the caching the client gets from an
// OpenAI model is.
func (r *OpenAIRouter) claudeAutoCacheDispatch(dispatch *providerDispatch, ctx *RequestContext) *dispatchAutoCache {
	if dispatch == nil || ctx == nil ||
		dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		(ctx.SourceFormat != llmprotocol.OpenAIChatV1 && ctx.SourceFormat != llmprotocol.OpenAIResponsesV1) ||
		r.armFamily(dispatch.logicalModel) != anthropicFamily {
		return nil
	}
	return &dispatchAutoCache{logicalModel: dispatch.logicalModel}
}

// applyDispatchAutoCache gives a covered dispatch the automatic-cache
// directive {"type":"ephemeral"} (the 5m default) when the client stated
// none. It runs in encodeDispatchRequest on the dispatched copy, after every
// late mutation -- tool selection, the thinking lever, reasoning rewrites,
// the output bound -- so the decision and the codec's placement read the same
// final request, and the client's own request, which the response cache and
// retained Responses state read, keeps what the client sent.
//
// Any directive the client stated wins: a top-level one, or a breakpoint of
// its own on a tool, a system block, a message block or a block inside a
// tool result. A client that placed breakpoints has said exactly what to
// cache, and one more from the Router would change what it pays to write the
// cache, so none is added and the skip is logged. Breakpoints are counted on
// what is dispatched: one on a tool the decision's tools plugin removed is
// not sent, so it no longer stops the directive.
//
// A directive supplied is recorded as a diagnostic naming its source, which
// the translation-warning counter, the protocol warnings header and the debug
// log all read; the codec's placement adds its own for where the breakpoint
// went. The directive is constant and its placement depends only on the
// request, so the same request dispatches the same bytes and a growing
// conversation keeps a cache-stable prefix across turns. A Messages client is
// never covered, so the replay of its own bytes is untouched.
func applyDispatchAutoCache(request *llmprotocol.Request, ctx *RequestContext) {
	auto := ctx.DispatchAutoCache
	if auto == nil || request.AutoCache != nil {
		return
	}
	if breakpoints := protocolcodec.CountCacheBreakpoints(*request); breakpoints > 0 {
		logging.ComponentEvent("extproc", "auto_cache_skipped", map[string]interface{}{
			"request_id":  ctx.RequestID,
			"model":       auto.logicalModel,
			"reason":      "client_breakpoints",
			"breakpoints": breakpoints,
		})
		return
	}
	origin := autoCacheSourceDefault
	// The codecs carry prompt_cache_key only when it holds a value.
	if carrier := request.Unmodeled; carrier != nil {
		if _, named := carrier.Fields["prompt_cache_key"]; named {
			origin = autoCacheSourcePromptCacheKey
		}
	}
	request.AutoCache = &llmprotocol.CacheDirective{Type: "ephemeral"}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
		Source: ctx.SourceFormat,
		Target: llmprotocol.AnthropicMessagesV1,
		Field:  "cache_control",
		Action: llmprotocol.DiagnosticGenerated,
		Reason: "automatic_cache_" + origin,
	})
}
