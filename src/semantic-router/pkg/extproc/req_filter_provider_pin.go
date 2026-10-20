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
// Two conditions gate it. The arm must carry the key, because an unpinned arm
// keeps the bytes it sent before the key existed. And the backend must be
// OpenRouter, because every other OpenAI-compatible backend would see an
// unknown member.
func applyProviderPreferences(
	body []byte,
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	privacyMode string,
) ([]byte, error) {
	pin, openRouter := providerPreferencesForDispatch(dispatch, routerConfig)
	if !openRouter || (pin == nil && privacyMode == "") {
		return body, nil
	}
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	encoded, err := json.Marshal(privacyProviderPreferences(pin, privacyMode))
	if err != nil {
		return nil, fmt.Errorf("failed to serialize provider preferences: %w", err)
	}
	requestMap[providerPreferencesField] = encoded
	mutated, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	return mutated, nil
}

// providerPreferencesForDispatch answers both gates in one place, so the
// mutation above reads as a mutation and the policy stays testable on its own:
// whether the dispatch reaches OpenRouter, and the arm's pin, nil when it pins
// nothing.
func providerPreferencesForDispatch(
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
) (*config.OpenRouterProviderPreferences, bool) {
	if dispatch == nil || routerConfig == nil || !providerIsOpenRouter(dispatch.profile) {
		return nil, false
	}
	return routerConfig.ProviderPreferencesForModel(dispatch.logicalModel), true
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

	var wire struct {
		Provider struct {
			Order []string `json:"order"`
		} `json:"provider"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return
	}
	ctx.DispatchedProviderOrder = wire.Provider.Order
}
