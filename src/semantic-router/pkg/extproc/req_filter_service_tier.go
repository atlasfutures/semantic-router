package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// serviceTierField is the top-level request member OpenRouter reads the
// service tier from.
const serviceTierField = "service_tier"

// applyServiceTier sends the tier an arm is configured for
// (providers.models[].service_tier) on an OpenRouter dispatch.
//
// The arm's price is defined for that tier, so the configured value replaces
// any tier the client sent: an arm priced at default must not be served at
// priority because a client asked. The client's own value is kept on the
// request context so a re-send on a twin without a configured tier can put
// it back. An arm without the key, or a non-OpenRouter backend, keeps the
// body exactly as it was.
func applyServiceTier(
	body []byte,
	dispatch *providerDispatch,
	routerConfig *config.RouterConfig,
	ctx *RequestContext,
) ([]byte, error) {
	if dispatch == nil || routerConfig == nil || !providerIsOpenRouter(dispatch.profile) {
		return body, nil
	}
	tier := routerConfig.ServiceTierForModel(dispatch.logicalModel)
	if tier == "" {
		return body, nil
	}
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	if ctx != nil {
		ctx.ClientServiceTier = append([]byte(nil), requestMap[serviceTierField]...)
	}
	encoded, err := json.Marshal(tier)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize service tier: %w", err)
	}
	requestMap[serviceTierField] = encoded
	mutated, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	return mutated, nil
}

// withResendServiceTier sets a standard re-send's tier from the twin's own
// configuration, never the discount attempt's. A twin without a configured
// tier gets back what the client sent (or no member) when the discount arm
// had replaced it; otherwise the body is left as dispatched.
func withResendServiceTier(body []byte, twinTier string, discountTier string, clientTier json.RawMessage) ([]byte, bool) {
	if twinTier == "" && discountTier == "" {
		return body, true
	}
	var members map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil {
		return nil, false
	}
	switch {
	case twinTier != "":
		encoded, err := json.Marshal(twinTier)
		if err != nil {
			return nil, false
		}
		members[serviceTierField] = encoded
	case len(clientTier) > 0:
		members[serviceTierField] = clientTier
	default:
		delete(members, serviceTierField)
	}
	replaced, err := json.Marshal(members)
	if err != nil {
		return nil, false
	}
	return replaced, true
}

// dispatchedServiceTier reads the service tier a rendered body carries, so
// the usage record names the bytes that travelled rather than the config.
func dispatchedServiceTier(body []byte) string {
	var wire struct {
		ServiceTier string `json:"service_tier"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return ""
	}
	return wire.ServiceTier
}
