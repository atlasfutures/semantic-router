package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// upstreamSessionIDField is OpenRouter's sticky routing key: a top-level
// string on the Chat Completions body, at most 256 characters. When it is
// present OpenRouter sends every request in the session to the same provider
// endpoint, so the provider-side prompt cache survives the turn boundary.
// Documented at https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion
// and https://openrouter.ai/docs/guides/best-practices/prompt-caching, read
// 2026-09-03.
const upstreamSessionIDField = "session_id"

// applyUpstreamSessionID gives the provider the router's own episode id so a
// conversation keeps one upstream endpoint.
//
// The router already derives a per-conversation episode id from the session
// header and hashes it; that hash is what travels. The raw header is
// userId:conversationId, so sending it would hand the provider an account
// identifier for nothing: the provider needs a key that is stable and unique,
// not a key it can read.
//
// Three conditions gate it, and all three are load-bearing. The decision must
// ask for it, because it is a provider-specific hint and not every decision
// dispatches to a provider that reads one. The backend must be OpenRouter,
// because every other OpenAI-compatible backend would see an unknown member.
// And the request must have an episode, because a request with no session
// header has no conversation to pin.
func applyUpstreamSessionID(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	episodeIDHash, wanted := upstreamSessionIDForDispatch(dispatch, ctx)
	if !wanted {
		return body, nil
	}
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	encoded, err := json.Marshal(episodeIDHash)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize upstream session id: %w", err)
	}
	requestMap[upstreamSessionIDField] = encoded
	mutated, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	decisionKey := config.RoutingDecisionKey(ctx.Routing.RecipeName(), dispatch.decisionName)
	metrics.RecordUpstreamSessionIDApplied(decisionKey)
	logging.Infof("Sent an upstream session id for provider affinity on decision %q", decisionKey)
	return mutated, nil
}

// upstreamSessionIDForDispatch answers the three gates in one place, so the
// mutation above reads as a mutation and the policy stays testable on its own.
func upstreamSessionIDForDispatch(
	dispatch *providerDispatch,
	ctx *RequestContext,
) (string, bool) {
	if dispatch == nil || ctx == nil || ctx.VSRSelectedDecision == nil {
		return "", false
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
	if params == nil || !params.SendUpstreamSessionID {
		return "", false
	}
	if !resolveOpenAIBackendDialect(dispatch.profile).usesUpstreamSessionID() {
		return "", false
	}
	if ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.EpisodeIDHash == "" {
		return "", false
	}
	return ctx.VSRRaylineARC.EpisodeIDHash, true
}
