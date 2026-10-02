package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// providerUsageCostWire is the charge OpenRouter adds to the usage object of
// every protocol it serves (Chat, Messages and Responses alike). It is embedded
// in each usage wire so the members are read rather than pruned, and the
// Router can account for the call with the provider's own number.
//
// Each member is held raw and read best-effort. The charge is evidence about
// the completion, not part of it: before it was read these members were
// pruned, and a completion the user already paid for must not fail because a
// provider spelled its charge in a shape the Router does not expect. A member
// that does not read as its expected type is left unknown and named once in
// upstream_usage_unreconciled.
//
// Encoding never sets it: the charge is the Router's accounting evidence, not
// part of any client contract.
type providerUsageCostWire struct {
	Cost        json.RawMessage `json:"cost,omitempty"`
	IsBYOK      json.RawMessage `json:"is_byok,omitempty"`
	CostDetails json.RawMessage `json:"cost_details,omitempty"`
}

func (wire providerUsageCostWire) decode() llmprotocol.ProviderCost {
	cost := llmprotocol.ProviderCost{
		Charged: lenientJSONValue[float64](wire.Cost, "usage.cost"),
		BYOK:    lenientJSONValue[bool](wire.IsBYOK, "usage.is_byok"),
	}
	if details := lenientJSONValue[map[string]json.RawMessage](wire.CostDetails, "usage.cost_details"); details != nil {
		cost.UpstreamInference = lenientJSONValue[float64](
			(*details)["upstream_inference_cost"], "usage.cost_details.upstream_inference_cost",
		)
	}
	return cost
}

// lenientJSONValue reads raw as a T. Absent and null are unknown; any other
// value that is not a T is unknown too, and named.
func lenientJSONValue[T any](raw json.RawMessage, path string) *T {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var value T
	if err := json.Unmarshal(trimmed, &value); err != nil {
		reportUnreconciledUsage(path)
		return nil
	}
	return &value
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
