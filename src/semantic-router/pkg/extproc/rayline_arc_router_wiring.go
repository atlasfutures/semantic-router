package extproc

import (
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
}

// registerRaylineARCSelector installs the Rayline ARC selector on the default
// recipe registry and hands its episode store to the router's resource scope
// so the generation lifecycle closes it after in-flight streams drain.
//
// The selector owns process-wide resources (an episode store and an encoder
// session), so it is built for the default recipe only. Multi-recipe ARC is
// deferred until those resources are recipe-scoped.
func registerRaylineARCSelector(
	cfg *config.RouterConfig,
	registries map[config.RecipeName]*selection.Registry,
	resources *resourceScope,
) raylineARCComponents {
	registry := registries[config.DefaultRecipeName]
	if registry == nil {
		return raylineARCComponents{}
	}
	arcSelector, episodeStore, closeEpisodeStore, closeEncoderSession, readinessFailure := createRaylineARCSelector(cfg)
	if arcSelector == nil {
		return raylineARCComponents{}
	}
	registry.Register(selection.MethodRaylineARC, arcSelector)
	if resources != nil {
		resources.add(closeEpisodeStore)
	}
	metrics.SetRaylineARCComponentReady(readinessFailure == "")
	// Each component reports its own readiness. An encoder that is not
	// answering yet must not read as a broken episode store, or the two
	// gauges cannot tell an operator which dependency is late.
	metrics.SetRaylineARCNamedComponentReady("episode_store", episodeStore != nil)
	fields := map[string]interface{}{
		"ready": readinessFailure == "",
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
