package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// providerIsBedrock reports an Amazon Bedrock provider profile.
func providerIsBedrock(profile *config.ProviderProfile) bool {
	return profile != nil && strings.EqualFold(strings.TrimSpace(profile.Type), "bedrock")
}
