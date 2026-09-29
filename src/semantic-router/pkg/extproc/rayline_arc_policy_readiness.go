package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// raylineARCPolicyDispatchReady is the policy-service mode's share of the
// artifact dispatch contract: the checks that do not need an artifact
// manifest. Every worker must reach its provider over https with a credential
// the router owns -- present, from its declared environment variable, never
// inline -- in the provider's default auth shape, and carry USD pricing with a
// cache-write rate, so its usage is accounted like the artifact mode's.
//
// The credential check is what keeps an absent or wrong-tenant key from
// letting a caller-supplied credential header reach the provider.
func raylineARCPolicyDispatchReady(cfg *config.RouterConfig, decision *config.Decision) bool {
	if cfg == nil || decision == nil {
		return false
	}
	for _, modelRef := range decision.ModelRefs {
		if !raylineARCPolicyWorkerReady(cfg, modelRef.Model) {
			return false
		}
	}
	return true
}

func raylineARCPolicyWorkerReady(cfg *config.RouterConfig, worker string) bool {
	pricing, ok := cfg.GetFullModelPricing(worker)
	if !ok || pricing.Currency != "USD" || pricing.CacheWritePer1M == nil {
		return false
	}
	endpoints := cfg.GetEndpointsForModel(worker)
	if len(endpoints) == 0 {
		return false
	}
	for index := range endpoints {
		endpoint := &endpoints[index]
		if endpoint.APIKeyEnvName == "" || endpoint.APIKey == "" || endpoint.APIKeyInline {
			return false
		}
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
		if err != nil || profile == nil || profile.ChatPath != "" {
			return false
		}
		parsed, err := url.Parse(profile.BaseURL)
		if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" {
			return false
		}
		switch {
		case raylineARCOpenAICompatibleProviderType(endpoint.Type) && raylineARCOpenAICompatibleProviderType(profile.Type):
			if !raylineARCAuthShapeMatches(profile) {
				return false
			}
		case strings.EqualFold(endpoint.Type, "anthropic") && strings.EqualFold(profile.Type, "anthropic"):
			if !raylineARCProviderDefaultAuth(profile) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// raylineARCProviderDefaultAuth reports whether a profile keeps its provider
// type's own credential header and prefix, as the catalog defines them, with
// no override that could let another header stand in for the credential.
func raylineARCProviderDefaultAuth(profile *config.ProviderProfile) bool {
	resolved, err := profile.ResolveAuth()
	if err != nil {
		return false
	}
	catalogDefault, err := (&config.ProviderProfile{Type: profile.Type}).ResolveAuth()
	return err == nil && strings.EqualFold(resolved.Header, catalogDefault.Header) &&
		resolved.Prefix == catalogDefault.Prefix
}
