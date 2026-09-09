package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// providerIsOpenRouter reports that a dispatch reaches OpenRouter, the one
// OpenAI-compatible backend that reads a top-level session_id, a provider
// object and a reasoning object; every other backend would see unknown
// members. The compiled catalog names it by provider id. A binding declared
// as an OpenAI-compatible provider with OpenRouter's base URL is the same
// backend, so the host decides when the id does not.
func providerIsOpenRouter(profile *config.ProviderProfile) bool {
	if profile == nil {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), "openrouter") {
		return true
	}
	return providerIsOpenRouterByHost(profile)
}

func providerIsOpenRouterByHost(profile *config.ProviderProfile) bool {
	if profile == nil || profile.BaseURL == "" {
		return false
	}
	parsed, err := url.Parse(profile.BaseURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	return host == "openrouter.ai" || strings.HasSuffix(host, ".openrouter.ai")
}
