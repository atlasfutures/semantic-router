package extproc

import (
	"context"
	"errors"
	"net/url"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
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
	dispatchedOn := raylineARCPolicyDispatchTargets(cfg, decision)
	for _, modelRef := range decision.ModelRefs {
		if raylineARCPolicyWorkerOutOfService(cfg, modelRef.Model, dispatchedOn) {
			continue
		}
		if !raylineARCPolicyWorkerReady(cfg, modelRef.Model) {
			return false
		}
	}
	return true
}

// raylineARCPolicyDispatchTargets names the workers an in-service arm is sent
// on through dispatch_model_overrides.
func raylineARCPolicyDispatchTargets(cfg *config.RouterConfig, decision *config.Decision) map[string]bool {
	targets := map[string]bool{}
	if decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil ||
		decision.Algorithm.RaylineARC.PolicyService == nil {
		return targets
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	for from, to := range policy.DispatchModelOverrides {
		if params, ok := cfg.ModelConfig[strings.TrimSpace(from)]; ok && params.IsDisabled() {
			continue
		}
		targets[strings.TrimSpace(to)] = true
	}
	return targets
}

// raylineARCPolicyWorkerOutOfService reports a worker no turn is sent to: its
// card is disabled, so the policy's hard mask never offers it, and no
// in-service arm is dispatched on it. Its provider contract is then not part
// of what readiness proves; an install that does not serve a model (an
// upstream that lacks it) carries no credential for it.
func raylineARCPolicyWorkerOutOfService(cfg *config.RouterConfig, worker string, dispatchedOn map[string]bool) bool {
	worker = strings.TrimSpace(worker)
	params, ok := cfg.ModelConfig[worker]
	return ok && params.IsDisabled() && !dispatchedOn[worker]
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
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
		if err != nil || profile == nil || profile.ChatPath != "" {
			return false
		}
		parsed, err := url.Parse(profile.BaseURL)
		if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" {
			return false
		}
		if raylineARCAmbientCredentialEndpoint(cfg, endpoint, profile) {
			continue
		}
		if endpoint.APIKeyEnvName == "" || endpoint.APIKey == "" || endpoint.APIKeyInline {
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

// raylineARCAmbientCredentialEndpoint reports a Vertex AI endpoint whose
// credential is the router's own Google identity rather than a static key:
// the endpoint declares no key at all, the profile keeps the provider's
// default Bearer shape, and the authz chain carries the google-access-token
// provider, which is what supplies that credential. A key or a header the
// caller could inject never stands in for it: the chain is the configured
// one, and a static key on the endpoint takes it out of this path.
func raylineARCAmbientCredentialEndpoint(
	cfg *config.RouterConfig,
	endpoint *config.VLLMEndpoint,
	profile *config.ProviderProfile,
) bool {
	if !strings.EqualFold(strings.TrimSpace(endpoint.Type), "vertex-ai") || !providerIsVertexAI(profile) {
		return false
	}
	if endpoint.APIKeyEnvName != "" || endpoint.APIKey != "" || endpoint.APIKeyInline {
		return false
	}
	if !raylineARCProviderDefaultAuth(profile) {
		return false
	}
	return raylineARCAuthzServesAmbientCredential(cfg)
}

// raylineARCAuthzServesAmbientCredential reports a valid, fail-closed authz
// chain that includes the google-access-token provider and no provider that
// reads a credential from a caller's header. A fail-open chain would send a
// request on without the router's credential when no token is at hand, and
// with it any Authorization header the caller sent.
func raylineARCAuthzServesAmbientCredential(cfg *config.RouterConfig) bool {
	if cfg.Authz.FailOpen {
		return false
	}
	providers := cfg.Authz.Providers
	if len(validateAuthzProviders(providers)) > 0 {
		return false
	}
	ambient := false
	for _, provider := range providers {
		switch provider.Type {
		case authz.CloudTokenProviderType:
			ambient = true
		case "header-injection":
			return false
		}
	}
	return ambient
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
// Decisions API v2 member and an exploration block that any turn can carry
// its capability, and every cost
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
	// A cell that excludes every turn from exploration never sends the
	// member, so it needs no exploration capability.
	if explore := newPolicyExploration(policy); explore.mayExplore() {
		capability := raylinearc.PolicyCapabilityExploration
		if explore.temperature {
			capability = raylinearc.PolicyCapabilityExplorationTemperature
		}
		required.Capabilities = append(required.Capabilities, capability)
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
	routing *config.RaylineARCRoutesRoutingConfig,
	advertised func([]string),
) error {
	// The listing's fallback pair, when a service still states one, is not
	// read: no model serves a turn the policy did not decide.
	listing, loaded, err := client.ServedPackage(ctx, policy.PackageAlias, policy.PackageSHA256)
	if err != nil {
		return err
	}
	// The optional capabilities a request may use only where the service
	// states them (predictions), kept for the cell's requests.
	if advertised != nil && listing != nil {
		advertised(listing.Capabilities)
	}
	required := raylineARCPolicyServiceRequirements(policy)
	routesRequirements(&required, routing, policy.EffectiveValueUSDMax())
	generationRoutingRequirements(&required, policy)
	err = raylinearc.CheckPolicyServiceRequirements(listing, loaded, required)
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

// routesRequirements adds what a cell's Decisions routing controls need: the
// tier points every lookup asks (swift and sage), cost weights over the
// whole public V range [0.1, 5] (cost_weight [0.2, 10]) for any custom
// result, and the temperature shape when the cell seeds lookup draws.
func routesRequirements(
	required *raylinearc.PolicyServiceRequirements,
	routing *config.RaylineARCRoutesRoutingConfig,
	valueUSDMax float64,
) {
	if routing == nil {
		return
	}
	for _, capability := range []string{
		raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints, raylinearc.PolicyCapabilityCallerPrices,
	} {
		if !slices.Contains(required.Capabilities, capability) {
			required.Capabilities = append(required.Capabilities, capability)
		}
	}
	if routing.ExplorationSalt != "" && !slices.Contains(required.Capabilities, raylinearc.PolicyCapabilityExplorationTemperature) {
		required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityExplorationTemperature)
	}
	required.CostWeights = append(required.CostWeights, 1/valueUSDMax, 1/routesValueUSDMin)
	for _, preset := range routesPresets {
		required.CostWeights = append(required.CostWeights, 1/preset.valueUSD)
	}
}

// generationRoutingRequirements adds what configurable ARC's caller V needs:
// decisions_v2, and cost weights over the whole public V range [0.1, 5]
// (cost_weight [0.2, 10]).
func generationRoutingRequirements(required *raylinearc.PolicyServiceRequirements, policy *config.RaylineARCPolicyServiceConfig) {
	if policy == nil || policy.GenerationRouting == nil || !policy.GenerationRouting.CallerValue {
		return
	}
	if !slices.Contains(required.Capabilities, raylinearc.PolicyCapabilityDecisionsV2) {
		required.Capabilities = append(required.Capabilities, raylinearc.PolicyCapabilityDecisionsV2)
	}
	required.CostWeights = append(required.CostWeights, 1/policy.EffectiveValueUSDMax(), 1/routesValueUSDMin)
}
