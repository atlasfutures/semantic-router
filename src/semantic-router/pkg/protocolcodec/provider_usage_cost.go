package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// providerUsageCostWire is the charge OpenRouter adds to the usage object of
// every protocol it serves (Chat, Messages and Responses alike). It is embedded
// in each usage wire so the members are decoded rather than pruned, and the
// Router can account for the call with the provider's own number.
//
// Encoding never sets it: the charge is the Router's accounting evidence, not
// part of any client contract.
type providerUsageCostWire struct {
	Cost        *float64                      `json:"cost,omitempty"`
	IsBYOK      *bool                         `json:"is_byok,omitempty"`
	CostDetails *providerUsageCostDetailsWire `json:"cost_details,omitempty"`
}

// providerUsageCostDetailsWire names only the member the Router carries. The
// per-direction splits stay unmodeled, so pruning still names them.
type providerUsageCostDetailsWire struct {
	UpstreamInferenceCost *float64 `json:"upstream_inference_cost,omitempty"`
}

func (wire providerUsageCostWire) decode() llmprotocol.ProviderCost {
	cost := llmprotocol.ProviderCost{
		Charged: cloneFloat64(wire.Cost),
		BYOK:    cloneBool(wire.IsBYOK),
	}
	if wire.CostDetails != nil {
		cost.UpstreamInference = cloneFloat64(wire.CostDetails.UpstreamInferenceCost)
	}
	return cost
}

// mergeProviderCost keeps the latest stated value of each member. A streamed
// usage update that does not restate a charge leaves the earlier one in place.
func mergeProviderCost(current, update llmprotocol.ProviderCost) llmprotocol.ProviderCost {
	if update.Charged != nil {
		current.Charged = update.Charged
	}
	if update.UpstreamInference != nil {
		current.UpstreamInference = update.UpstreamInference
	}
	if update.BYOK != nil {
		current.BYOK = update.BYOK
	}
	return current
}

func cloneFloat64(value *float64) *float64 {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}
