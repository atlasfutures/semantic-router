package cachecontext

import (
	"context"
	"crypto/sha256"
	"fmt"
)

// PromptIdentity qualifies the serving tokenizer and chat template, not the
// ARC encoder. RuntimeIdentity must pin their revisions and model configuration.
// It does not assert that a runtime's KV cache is resident.
type PromptIdentity struct {
	RuntimeIdentity string
	Model           string
	Protocol        string
	Units           string
}

// PreparedPrompt is the final provider-bound request, including private
// append, system and tools. A counter must return an unknown CarriedPrefixTokens
// unless it can identify the exact token boundary before the new user turn in
// the serving template. Counting just message text is not sufficient.
type PreparedPrompt struct {
	Identity     PromptIdentity
	ContextEpoch string
	Body         []byte
}

// PromptCounter is an explicitly installed, qualified serving counter. A
// provider token-count endpoint may return counts without token IDs. A local
// tokenizer must use the actual serving model/template and special tokens.
type PromptCounter interface {
	Identity() PromptIdentity
	Count(context.Context, PreparedPrompt) (PromptMeasurement, error)
}

type PromptMeasurement struct {
	PromptTokens        int
	CarriedPrefixTokens *int
	// TokenIDs, when known, are the full prompt in actual serving order.
	// A count-only provider leaves this nil; counts cannot prove overlap.
	TokenIDs   []int
	Provenance string // provider_reported or local_measured; estimates refused
}

// MeasuredPrompt is an immutable local checkpoint. It is staged before
// dispatch and may replace the previous checkpoint only on execution commit.
// Token IDs and body digests never belong in public execution context.
type MeasuredPrompt struct {
	identity   PromptIdentity
	epoch      string
	body       [sha256.Size]byte
	prefix     *int
	tokens     []int
	provenance string
	// total is the full prepared prompt count in the serving tokenizer.
	total int
}

type CurrentPrefix struct {
	Tokens        int    `json:"tokens"`
	OverlapTokens *int   `json:"overlap_tokens"`
	Units         string `json:"units"`
	Provenance    string `json:"provenance"`
}

func MeasurePrompt(ctx context.Context, counter PromptCounter, prompt PreparedPrompt) (*MeasuredPrompt, error) {
	if counter == nil || counter.Identity() != prompt.Identity {
		return nil, fmt.Errorf("prompt counter does not match the serving identity")
	}
	if prompt.Identity.RuntimeIdentity == "" || prompt.Identity.Model == "" || prompt.Identity.Protocol == "" || prompt.Identity.Units == "" || prompt.ContextEpoch == "" {
		return nil, fmt.Errorf("prompt counter identity and context epoch must be explicit")
	}
	body := sha256.Sum256(prompt.Body)
	// A counter must not alter the request that will be dispatched.
	prompt.Body = append([]byte(nil), prompt.Body...)
	measurement, err := counter.Count(ctx, prompt)
	if err != nil {
		return nil, err
	}
	if sha256.Sum256(prompt.Body) != body {
		return nil, fmt.Errorf("prompt counter changed the prepared request")
	}
	return admitMeasurement(prompt, body, measurement)
}

func admitMeasurement(prompt PreparedPrompt, body [sha256.Size]byte, measurement PromptMeasurement) (*MeasuredPrompt, error) {
	if measurement.Provenance != "provider_reported" && measurement.Provenance != "local_measured" {
		return nil, fmt.Errorf("prompt counter must provide measured serving tokens")
	}
	if measurement.PromptTokens < 0 || (measurement.CarriedPrefixTokens != nil && (*measurement.CarriedPrefixTokens < 0 || *measurement.CarriedPrefixTokens > measurement.PromptTokens)) {
		return nil, fmt.Errorf("prompt counter returned inconsistent counts")
	}
	if measurement.TokenIDs != nil && len(measurement.TokenIDs) != measurement.PromptTokens {
		return nil, fmt.Errorf("prompt token IDs do not match the full prompt count")
	}
	for _, token := range measurement.TokenIDs {
		if token < 0 {
			return nil, fmt.Errorf("prompt counter returned a negative token ID")
		}
	}
	result := &MeasuredPrompt{identity: prompt.Identity, epoch: prompt.ContextEpoch, body: body, provenance: measurement.Provenance, total: measurement.PromptTokens}
	if measurement.CarriedPrefixTokens != nil {
		count := *measurement.CarriedPrefixTokens
		result.prefix = &count
	}
	if measurement.TokenIDs != nil {
		result.tokens = append([]int{}, measurement.TokenIDs...)
	}
	return result, nil
}

// Matches permits reuse within one transaction without counting the same
// exact prepared request twice. Changes of action, template or epoch invalidate
// the measurement even when the public transcript is unchanged.
func (measured *MeasuredPrompt) Matches(prompt PreparedPrompt) bool {
	return measured != nil && measured.identity == prompt.Identity && measured.epoch == prompt.ContextEpoch && measured.body == sha256.Sum256(prompt.Body)
}

// PromptTokens is the full prepared prompt in the serving tokenizer's units,
// as counted by the qualified runtime. It is what capacity admission needs.
func (measured *MeasuredPrompt) PromptTokens() int { return measured.total }

// Units and Provenance identify the count's tokenizer and source.
func (measured *MeasuredPrompt) Units() string      { return measured.identity.Units }
func (measured *MeasuredPrompt) Provenance() string { return measured.provenance }

// Prefix compares serving token IDs only in the same qualified units, template
// and epoch. The result says nothing about physical residency or TTL. Unknown
// token IDs leave overlap unknown, including for provider-reported counts.
func (measured *MeasuredPrompt) Prefix(previous *MeasuredPrompt) *CurrentPrefix {
	if measured == nil || measured.prefix == nil {
		return nil
	}
	result := &CurrentPrefix{Tokens: *measured.prefix, Units: measured.identity.Units, Provenance: measured.provenance}
	if previous == nil || previous.identity != measured.identity || previous.epoch != measured.epoch || previous.tokens == nil || measured.tokens == nil {
		return result
	}
	overlap := min(*measured.prefix, TokenPrefixLength(previous.tokens, measured.tokens))
	result.OverlapTokens = &overlap
	return result
}

// TokenPrefixLength compares token IDs, not prompt bytes or character counts.
// The caller is responsible for establishing compatible tokenizer units first.
func TokenPrefixLength(previous, current []int) int {
	index := 0
	for index < min(len(previous), len(current)) && previous[index] == current[index] {
		index++
	}
	return index
}
