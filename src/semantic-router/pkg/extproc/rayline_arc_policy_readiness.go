package extproc

import (
	"context"
	"errors"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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

// raylineARCPolicyServiceRequirements is what the cell's opt-in request
// members need the policy service to advertise (pathfinder#5454): each
// Decisions API v2 member and exploration its capability, and every cost
// weight the cell sends a place inside the served package's bounds. A cell
// with none of them needs nothing, so it still arms against a service that
// predates the capabilities listing.
func raylineARCPolicyServiceRequirements(policy *config.RaylineARCPolicyServiceConfig) raylinearc.PolicyServiceRequirements {
	var required raylinearc.PolicyServiceRequirements
	if policy == nil {
		return required
	}
	if v2 := policy.DecisionsV2; v2 != nil {
		if v2.Name != "" || v2.CostWeight != nil || len(v2.TierPoints) > 0 || len(v2.ResponseFields) > 0 {
			required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityDecisionsV2)
		}
		if len(v2.TierPoints) > 0 {
			required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityTierPoints)
		}
		if len(v2.ResponseFields) > 0 {
			required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityResponseFields)
		}
		if v2.CostWeight != nil {
			required.CostWeights = append(required.CostWeights, *v2.CostWeight)
		}
		for _, point := range v2.TierPoints {
			if point.CostWeight != nil {
				required.CostWeights = append(required.CostWeights, *point.CostWeight)
			}
		}
	}
	if policy.Exploration != nil {
		required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityExploration)
	}
	return required
}

// requireRaylineARCPolicyServing fails unless the service has loaded the
// cell's package and serves every request member the cell sends. A service
// that lacks one leaves the cell not ready, with the missing capability or
// cost weight logged, rather than arming a cell whose every turn the service
// would refuse or whose members it would ignore.
func requireRaylineARCPolicyServing(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	policy *config.RaylineARCPolicyServiceConfig,
) error {
	// The listing's fallback pair, when a service still states one, is not
	// read: no model serves a turn the policy did not decide.
	listing, loaded, err := client.ServedPackage(ctx, policy.PackageAlias, policy.PackageSHA256)
	if err != nil {
		return err
	}
	err = raylinearc.CheckPolicyServiceRequirements(listing, loaded, raylineARCPolicyServiceRequirements(policy))
	var failure *raylinearc.PolicyServiceError
	if errors.As(err, &failure) {
		fields := map[string]interface{}{"package_alias": policy.PackageAlias, "class": failure.Class}
		for key, value := range failure.Detail {
			fields[key] = value
		}
		logging.ComponentErrorEvent("extproc", "rayline_arc_policy_service_unsupported", fields)
	}
	return err
}
