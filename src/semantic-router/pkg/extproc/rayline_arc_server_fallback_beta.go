package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// serverSideFallbackBetaPrefix names Anthropic's server-side fallback betas
// (server-side-fallback-2026-06-01 and later). On an ARC turn the policy
// decides every model that serves (ADR 0145), and the cell re-serves a
// refused turn itself, so a client cannot turn on Anthropic's own fallback
// through the cell: it would serve a model the policy did not choose
// (pathfinder#6066 §6).
const serverSideFallbackBetaPrefix = "server-side-fallback-"

// withoutServerSideFallbackBetas is an anthropic-beta value without the
// server-side fallback betas, and whether it had any.
func withoutServerSideFallbackBetas(value string) (string, bool) {
	kept := []string{}
	removed := false
	for _, beta := range strings.Split(value, ",") {
		beta = strings.TrimSpace(beta)
		if beta == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(beta), serverSideFallbackBetaPrefix) {
			removed = true
			continue
		}
		kept = append(kept, beta)
	}
	return strings.Join(kept, ","), removed
}

// stripServerSideFallbackBeta takes the server-side fallback betas out of an
// ARC dispatch's anthropic-beta, the dispatch's own value if it set one, else
// the client's. The client's header is updated too, so a re-send carries the
// stripped value. A request that declared none is left as it was.
func stripServerSideFallbackBeta(mutation *ext_proc.HeaderMutation, ctx *RequestContext) {
	if mutation == nil || ctx == nil || ctx.VSRRaylineARC == nil {
		return
	}
	var dispatched *core.HeaderValueOption
	for _, header := range mutation.SetHeaders {
		if header.GetHeader() != nil && strings.EqualFold(header.Header.Key, anthropicBetaHeader) {
			dispatched = header
		}
	}
	current, declared := lookupHeaderIgnoringCase(ctx.Headers, anthropicBetaHeader)
	if dispatched != nil {
		current, declared = string(dispatched.Header.RawValue), true
		if current == "" {
			current = dispatched.Header.Value
		}
	}
	if !declared {
		return
	}
	stripped, removed := withoutServerSideFallbackBetas(current)
	if !removed {
		return
	}
	for key := range ctx.Headers {
		if strings.EqualFold(key, anthropicBetaHeader) {
			ctx.Headers[key] = stripped
		}
	}
	switch {
	case dispatched != nil:
		dispatched.Header.RawValue, dispatched.Header.Value = []byte(stripped), ""
	case stripped == "":
		mutation.RemoveHeaders = append(mutation.RemoveHeaders, anthropicBetaHeader)
	default:
		mutation.SetHeaders = append(mutation.SetHeaders, overwriteRequestHeader(anthropicBetaHeader, stripped))
	}
}
