package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
)

// PricingSnapshotPrefix marks a snapshot id as a digest of the rate table.
const PricingSnapshotPrefix = "sha256:"

// pricingSnapshotEntry is everything GetFullModelPricing reads for one model:
// the name it is keyed by, the external ids and LoRA names that resolve to it,
// and its rates. Two configs with equal snapshots price every model alike.
type pricingSnapshotEntry struct {
	Model            string            `json:"model"`
	ExternalModelIDs map[string]string `json:"external_model_ids,omitempty"`
	LoRAs            []string          `json:"loras,omitempty"`
	Currency         string            `json:"currency,omitempty"`
	PromptPer1M      string            `json:"prompt_per_1m"`
	CompletionPer1M  string            `json:"completion_per_1m"`
	CachedInputPer1M string            `json:"cached_input_per_1m"`
	CacheWritePer1M  *string           `json:"cache_write_per_1m"`
}

// snapshotRate writes a rate in its shortest round-tripping form. A string
// rather than a JSON number keeps a non-finite rate a YAML file can spell
// encodable, so the snapshot is always computable.
func snapshotRate(rate float64) string {
	return strconv.FormatFloat(rate, 'g', -1, 64)
}

// PricingSnapshotID names the rate table this config prices with, as
// "sha256:" and the digest of its canonical form. A usage record that carries
// it can be repriced or audited against exactly the rates it was priced from,
// and it changes whenever any rate, alias or LoRA mapping changes.
//
// A config with no model table still has a snapshot: the empty one, under
// which every model is unpriced.
func (c *RouterConfig) PricingSnapshotID() string {
	entries := make([]pricingSnapshotEntry, 0)
	if c != nil {
		for model, params := range c.ModelConfig {
			loras := make([]string, 0, len(params.LoRAs))
			for _, adapter := range params.LoRAs {
				loras = append(loras, adapter.Name)
			}
			sort.Strings(loras)
			entry := pricingSnapshotEntry{
				Model:            model,
				ExternalModelIDs: params.ExternalModelIDs,
				LoRAs:            loras,
				Currency:         params.Pricing.Currency,
				PromptPer1M:      snapshotRate(params.Pricing.PromptPer1M),
				CompletionPer1M:  snapshotRate(params.Pricing.CompletionPer1M),
				CachedInputPer1M: snapshotRate(params.Pricing.CachedInputPer1M),
			}
			if params.Pricing.CacheWritePer1M != nil {
				rate := snapshotRate(*params.Pricing.CacheWritePer1M)
				entry.CacheWritePer1M = &rate
			}
			entries = append(entries, entry)
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Model < entries[j].Model })
	// encoding/json writes map keys sorted, so equal tables encode to equal
	// bytes. Every member is a string, a string slice or a string map, which
	// cannot fail to encode.
	encoded, _ := json.Marshal(entries)
	digest := sha256.Sum256(encoded)
	return PricingSnapshotPrefix + hex.EncodeToString(digest[:])
}
