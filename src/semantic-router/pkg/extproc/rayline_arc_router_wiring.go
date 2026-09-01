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
	metrics.SetRaylineARCNamedComponentReady(
		"episode_store",
		readinessFailure == "" && episodeStore != nil,
	)
	fields := map[string]interface{}{
		"ready": readinessFailure == "",
	}
	if readinessFailure != "" {
		fields["failure_class"] = readinessFailure
		logging.ComponentErrorEvent("extproc", "rayline_arc_component_readiness", fields)
	} else {
		logging.ComponentEvent("extproc", "rayline_arc_component_readiness", fields)
	}
	return raylineARCComponents{
		episodeStore: episodeStore,
		sessionClose: closeEncoderSession,
	}
}
