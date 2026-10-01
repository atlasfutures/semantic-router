package extproc

import (
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineARCComponents holds the process-wide ARC resources the router owns
// for the life of one router generation.
type raylineARCComponents struct {
	episodeStore raylinearc.EpisodeStore
	sessionClose raylineARCSessionCloseFunc
	// recipeEpisodeStores is each named recipe's own episode store, for the
	// recipes whose ARC decision takes its decision from a policy service.
	recipeEpisodeStores map[config.RecipeName]raylinearc.EpisodeStore
}

// registerRaylineARCSelector installs the Rayline ARC selector on the default
// recipe registry and hands its episode store to the router's resource scope
// so the generation lifecycle closes it after in-flight streams drain.
//
// A named recipe gets its own ARC selector only in the policy-service mode:
// there the selector's resources -- a service client and an episode store --
// are its own, so several candidates can run side by side, one per recipe,
// chosen by the request's model name through entrypoints. The artifact mode
// owns process-wide resources (an encoder session pool) and stays on the
// default recipe until those are recipe-scoped.
func registerRaylineARCSelector(
	cfg *config.RouterConfig,
	registries map[config.RecipeName]*selection.Registry,
	resources *resourceScope,
) raylineARCComponents {
	components := registerRaylineARCRecipeSelector(cfg, registries[config.DefaultRecipeName], resources, "")
	prefixes := map[string]config.RecipeName{}
	if decisions := configuredRaylineARCDecisions(cfg); len(decisions) > 0 {
		if namespace := raylineARCEpisodeNamespace(decisions[0].Algorithm.RaylineARC.Episode); namespace != "" {
			prefixes[namespace] = config.DefaultRecipeName
		}
	}
	for index := range cfg.Recipes {
		recipe := &cfg.Recipes[index]
		if recipe.Name == config.DefaultRecipeName {
			continue
		}
		scoped := cfg.ConfigForRecipe(recipe)
		decisions := configuredRaylineARCDecisions(scoped)
		if len(decisions) == 0 {
			continue
		}
		arcConfig := decisions[0].Algorithm.RaylineARC
		namespace := raylineARCEpisodeNamespace(arcConfig.Episode)
		refusal := raylineARCRecipeRefusal(arcConfig, namespace, prefixes)
		if refusal != "" {
			logging.ComponentErrorEvent("extproc", "rayline_arc_component_readiness", map[string]interface{}{
				"ready": false, "recipe": string(recipe.Name), "failure_class": refusal, "reprobing": false,
			})
			continue
		}
		if namespace != "" {
			prefixes[namespace] = recipe.Name
		}
		registered := registerRaylineARCRecipeSelector(scoped, registries[recipe.Name], resources, recipe.Name)
		if registered.episodeStore != nil {
			if components.recipeEpisodeStores == nil {
				components.recipeEpisodeStores = map[config.RecipeName]raylinearc.EpisodeStore{}
			}
			components.recipeEpisodeStores[recipe.Name] = registered.episodeStore
		}
	}
	return components
}

// raylineARCRecipeRefusal says why a named recipe's ARC decision cannot get a
// selector of its own, or "" when it can: only the policy-service mode is
// recipe-scoped, and no two recipes may write one Redis episode namespace,
// or they would read and fence each other's episodes under a shared session
// id.
func raylineARCRecipeRefusal(
	arcConfig *config.RaylineARCAlgorithmConfig,
	namespace string,
	taken map[string]config.RecipeName,
) string {
	if arcConfig.PolicyService == nil {
		return "recipe_requires_policy_service"
	}
	if owner, ok := taken[namespace]; ok && namespace != "" {
		return "episode_namespace_shared_with_" + string(owner)
	}
	return ""
}

// raylineARCEpisodeNamespace identifies the keyspace an episode store writes:
// memory stores are private to their selector, Redis stores share a server
// and are told apart only by key prefix.
func raylineARCEpisodeNamespace(episode config.RaylineARCEpisodeConfig) string {
	if episode.Backend != config.RaylineARCBackendRedis {
		return ""
	}
	return episode.Redis.Address + "/" + strconv.Itoa(episode.Redis.DB) + "/" + episode.KeyPrefix
}

func raylineARCRecipeLabel(recipe config.RecipeName) string {
	if recipe == "" {
		return string(config.DefaultRecipeName)
	}
	return string(recipe)
}

func registerRaylineARCRecipeSelector(
	cfg *config.RouterConfig,
	registry *selection.Registry,
	resources *resourceScope,
	recipe config.RecipeName,
) raylineARCComponents {
	if registry == nil {
		return raylineARCComponents{}
	}
	var (
		arcSelector         selection.Selector
		episodeStore        raylinearc.EpisodeStore
		closeEpisodeStore   func() error
		closeEncoderSession raylineARCSessionCloseFunc
		readinessFailure    string
	)
	if recipe == "" {
		arcSelector, episodeStore, closeEpisodeStore, closeEncoderSession, readinessFailure = createRaylineARCSelector(cfg)
	} else {
		arcSelector, episodeStore, closeEpisodeStore, closeEncoderSession, readinessFailure = createRaylineARCRecipePolicySelector(cfg, recipe)
	}
	if arcSelector == nil {
		return raylineARCComponents{}
	}
	registry.Register(selection.MethodRaylineARC, arcSelector)
	if resources != nil {
		resources.add(closeEpisodeStore)
	}
	fields := map[string]interface{}{
		"ready":  readinessFailure == "",
		"recipe": raylineARCRecipeLabel(recipe),
	}
	if recipe == "" {
		metrics.SetRaylineARCComponentReady(readinessFailure == "")
		// Each component reports its own readiness. An encoder that is not
		// answering yet must not read as a broken episode store, or the two
		// gauges cannot tell an operator which dependency is late.
		metrics.SetRaylineARCNamedComponentReady("episode_store", raylineARCEpisodeStoreReady(episodeStore))
	}
	switch readinessFailure {
	case "":
		logging.ComponentEvent("extproc", "rayline_arc_component_readiness", fields)
	case raylineARCReadinessPendingClass:
		// The normal boot: registered, unarmed, probing. Not an error, and it
		// is the line that proves readiness ran at all.
		fields["failure_class"] = readinessFailure
		fields["reprobing"] = true
		logging.ComponentEvent("extproc", "rayline_arc_component_readiness", fields)
	default:
		fields["failure_class"] = readinessFailure
		fields["reprobing"] = false
		logging.ComponentErrorEvent("extproc", "rayline_arc_component_readiness", fields)
	}
	return raylineARCComponents{
		episodeStore: episodeStore,
		sessionClose: closeEncoderSession,
	}
}

// createRaylineARCRecipePolicySelector builds a named recipe's policy-service
// selector; its decisions must agree the way the default recipe's must.
func createRaylineARCRecipePolicySelector(
	cfg *config.RouterConfig,
	recipe config.RecipeName,
) (selection.Selector, raylinearc.EpisodeStore, func() error, raylineARCSessionCloseFunc, string) {
	decisions := configuredRaylineARCDecisions(cfg)
	for index := 1; index < len(decisions); index++ {
		if !sameRaylineARCSelectionConfig(decisions[0].Algorithm.RaylineARC, decisions[index].Algorithm.RaylineARC) ||
			!raylineARCDecisionsShareWorkers(decisions) {
			selector := newUnrecoverableRaylineARCSelector(decisions[0].Algorithm.RaylineARC.PolicyService.PackageSHA256)
			selector.recipe = recipe
			return selector, nil, nil, nil, "conflicting_config"
		}
	}
	return createRaylineARCPolicySelector(cfg, decisions[0], recipe)
}
