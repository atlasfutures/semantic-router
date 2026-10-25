package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// providerIsOpenRouter reports that a dispatch reaches OpenRouter, the one
// OpenAI-compatible backend that reads a top-level session_id, a provider
// object and a reasoning object; every other backend would see unknown
// members. The compiled catalog names it by provider id. A binding declared
// as an OpenAI-compatible provider with OpenRouter's base URL is the same
// backend, so the host decides when the id does not.
func providerIsOpenRouter(profile *config.ProviderProfile) bool {
	return profile.IsOpenRouter()
}

func providerIsOpenRouterByHost(profile *config.ProviderProfile) bool {
	return profile.HasOpenRouterHost()
}
