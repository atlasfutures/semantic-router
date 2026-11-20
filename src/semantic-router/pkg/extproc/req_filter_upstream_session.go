package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// upstreamSessionIDField is OpenRouter's sticky routing key: a top-level
// string on the Chat Completions and Messages bodies, at most 256 characters. When it is
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
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) ([]byte, error) {
	encoded, err := upstreamSessionIDMember(dispatch, routerConfig, ctx)
	if err != nil || encoded == nil {
		return body, err
	}
	mutated, err := setTopLevelMembers(body, map[string]json.RawMessage{upstreamSessionIDField: encoded})
	if err != nil {
		return nil, err
	}
	recordUpstreamSessionIDApplied(dispatch, ctx)
	return mutated, nil
}

// upstreamSessionIDMember is the session_id member a dispatch carries, nil
// when it carries none.
func upstreamSessionIDMember(
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) (json.RawMessage, error) {
	sessionID, wanted := upstreamSessionIDForDispatch(dispatch, routerConfig, ctx)
	if !wanted {
		return nil, nil
	}
	encoded, err := json.Marshal(sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize upstream session id: %w", err)
	}
	return encoded, nil
}

func recordUpstreamSessionIDApplied(dispatch *providerDispatch, ctx *RequestContext) {
	decisionKey := config.RoutingDecisionKey(ctx.Routing.RecipeName(), dispatch.decisionName)
	metrics.RecordUpstreamSessionIDApplied(decisionKey)
	logging.Infof("Sent an upstream session id for provider affinity on decision %q", decisionKey)
}

// applyUpstreamRoutingMembers is applyUpstreamSessionID then
// applyProviderPreferences in one rewrite of the body. Each rewrite decodes
// and re-encodes the whole transcript to set one top-level member, and the
// encoding is canonical (sorted members, compact values), so setting both in
// one pass writes the bytes the two passes wrote.
func applyUpstreamRoutingMembers(
	body []byte,
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) ([]byte, error) {
	session, err := upstreamSessionIDMember(dispatch, routerConfig, ctx)
	if err != nil {
		return nil, err
	}
	provider, err := providerPreferencesMember(dispatch, routerConfig, ctx)
	if err != nil {
		return nil, err
	}
	members := make(map[string]json.RawMessage, 2)
	if session != nil {
		members[upstreamSessionIDField] = session
	}
	if provider != nil {
		members[providerPreferencesField] = provider
	}
	if len(members) == 0 {
		return body, nil
	}
	mutated, err := setTopLevelMembers(body, members)
	if err != nil {
		return nil, err
	}
	if session != nil {
		recordUpstreamSessionIDApplied(dispatch, ctx)
	}
	return mutated, nil
}

// setTopLevelMembers sets members on the body's top-level object, re-encoding
// it as encoding/json does a map: members sorted, values compact.
func setTopLevelMembers(body []byte, members map[string]json.RawMessage) ([]byte, error) {
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	for name, value := range members {
		requestMap[name] = value
	}
	mutated, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	return mutated, nil
}

// upstreamSessionIDForDispatch answers the three gates in one place, so the
// mutation above reads as a mutation and the policy stays testable on its own,
// and picks the key's scope by the worker's family.
//
// For an OpenAI worker OpenRouter also scopes OpenAI's prompt cache by the
// session_id: a new session_id never reads a prefix another session wrote,
// even byte for byte the same and seconds old, while the same session_id
// always does. A per-conversation key therefore made every first turn of a
// conversation on a GPT worker a full cache write, where an Anthropic worker
// reads the shared system and tools prefix. An OpenAI worker is sent the per-user key
// instead, so one user's conversations share their common prefix and users
// stay apart, or no session_id when the episode names no user. Other
// families keep the per-conversation key.
func upstreamSessionIDForDispatch(
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) (string, bool) {
	if dispatch == nil || ctx == nil || ctx.VSRSelectedDecision == nil {
		return "", false
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
	if params == nil || !params.SendUpstreamSessionID {
		return "", false
	}
	if !providerIsOpenRouter(dispatch.profile) {
		return "", false
	}
	if ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.EpisodeIDHash == "" {
		return "", false
	}
	if modelFamily(routerConfig, dispatch.logicalModel) == openAIFamily {
		return ctx.RaylineARCUserKey, ctx.RaylineARCUserKey != ""
	}
	return ctx.VSRRaylineARC.EpisodeIDHash, true
}
