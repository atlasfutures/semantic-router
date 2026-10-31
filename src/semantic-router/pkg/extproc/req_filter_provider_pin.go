package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// providerPreferencesField is OpenRouter's provider-routing instruction: a
// top-level object on the Chat Completions and Messages bodies naming which providers may
// serve the request and whether OpenRouter may fall back past them.
// Documented at https://openrouter.ai/docs/features/provider-routing, read
// 2026-09-05.
const providerPreferencesField = "provider"

// applyProviderPreferences pins the arm to the providers its model card names.
//
// OpenRouter picks a provider per request from its own price and uptime
// ranking, so two turns on the same arm can be served by providers that differ
// in quantization, tokenizer and context handling. An arm measured on one of
// them is not the arm the next turn gets, and a latency or quality number
// carries the provider it happened to land on rather than the arm.
//
// Which providers may serve an arm is configuration, not routing: the decision
// already chose the arm, and this states how that arm reaches a machine. So the
// pin is read off the arm's model card and nothing here consults the request.
//
// A private turn (rayline_arc_privacy.go) is the one exception: its data mode
// is added to the pin, or sent alone when the arm pins nothing, and either way
// replaces the provider object the request carries.
//
// An ARC turn on an arm that pins more than one endpoint is pinned to one of
// them, picked by its session (rayline_arc_endpoint_pick.go), so the
// conversation keeps one endpoint's cache and the endpoint that served is
// known.
//
// Two conditions gate it. The arm must carry the key, because an unpinned arm
// keeps the bytes it sent before the key existed. And the backend must be
// OpenRouter, because every other OpenAI-compatible backend would see an
// unknown member.
func applyProviderPreferences(
	body []byte,
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) ([]byte, error) {
	encoded, err := providerPreferencesMember(dispatch, routerConfig, ctx)
	if err != nil || encoded == nil {
		return body, err
	}
	return setTopLevelMembers(body, map[string]json.RawMessage{providerPreferencesField: encoded})
}

// providerPreferencesMember is the provider member a dispatch carries, nil
// when it carries none, and records the endpoint it pins.
func providerPreferencesMember(
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) (json.RawMessage, error) {
	privacyMode := ""
	if ctx != nil {
		privacyMode = ctx.RaylineARCPrivacyMode
	}
	resetDispatchedEndpoint(ctx)
	pin, openRouter := providerPreferencesForDispatch(dispatch, routerConfig, privacyMode)
	if !openRouter || (pin == nil && privacyMode == "") {
		return nil, nil
	}
	sent := privacyProviderPreferences(pickArmEndpoint(ctx, dispatch, pin), privacyMode)
	if ctx != nil {
		ctx.RaylineARCDispatchedEndpoint = dispatchedEndpoint(ctx, sent, dispatch.upstreamModel)
	}
	encoded, err := json.Marshal(sent)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize provider preferences: %w", err)
	}
	return encoded, nil
}

// providerPreferencesForDispatch answers both gates in one place, so the
// mutation above reads as a mutation and the policy stays testable on its own:
// whether the dispatch reaches OpenRouter, and the arm's pin for the turn's
// privacy mode (armProviderPin), nil when it pins nothing.
func providerPreferencesForDispatch(
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	privacyMode string,
) (*config.OpenRouterProviderPreferences, bool) {
	if dispatch == nil || routerConfig == nil || !providerIsOpenRouter(dispatch.profile) {
		return nil, false
	}
	return armProviderPin(routerConfig, dispatch.logicalModel, privacyMode), true
}

// recordDispatchedProviderPin names the providers the request actually pinned,
// read back off the rendered body rather than remembered from the arm's
// configuration. The routing record then cannot disagree with the wire, which
// is the rule the reasoning controls already follow.
func recordDispatchedProviderPin(ctx *RequestContext, body []byte) {
	if ctx == nil {
		return
	}
	ctx.DispatchedProviderOrder = nil
	ctx.DispatchedProviderPrivacy = dispatchedProviderPrivacy{}

	var wire dispatchedProviderWire
	if json.Unmarshal(body, &wire) != nil {
		return
	}
	wire.record(ctx)
}

// dispatchedProviderWire is the body's provider object, as far as the record
// reads it.
type dispatchedProviderWire struct {
	Provider struct {
		Order          []string `json:"order"`
		DataCollection string   `json:"data_collection"`
		ZDR            *bool    `json:"zdr"`
		AllowFallbacks *bool    `json:"allow_fallbacks"`
	} `json:"provider"`
}

func (wire *dispatchedProviderWire) record(ctx *RequestContext) {
	ctx.DispatchedProviderOrder = wire.Provider.Order
	ctx.DispatchedProviderPrivacy = dispatchedProviderPrivacy{
		dataCollection: wire.Provider.DataCollection,
		zdr:            wire.Provider.ZDR,
		allowFallbacks: wire.Provider.AllowFallbacks,
	}
}

// dispatchedProviderPrivacy is the provider object's data-use members as the
// rendered upstream body carries them (llm_usage provider_data_collection,
// provider_zdr, provider_allow_fallbacks): empty and nil where the body sent
// none.
type dispatchedProviderPrivacy struct {
	dataCollection string
	zdr            *bool
	allowFallbacks *bool
}
