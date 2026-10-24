package cachecontext

import (
	"crypto/sha256"
	"testing"
)

func measuredTokens(t *testing.T, identity PromptIdentity, epoch string, tokens []int, prefix int) *MeasuredPrompt {
	t.Helper()
	result, err := admitMeasurement(PreparedPrompt{Identity: identity, ContextEpoch: epoch}, sha256.Sum256(nil), PromptMeasurement{PromptTokens: len(tokens), CarriedPrefixTokens: &prefix, TokenIDs: tokens, Provenance: "local_measured"})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestTokenPrefixUsesExactCommonTokensAndCarriedBoundary(t *testing.T) {
	identity := PromptIdentity{RuntimeIdentity: "model-and-template-r1", Model: "worker", Protocol: "chat", Units: "tokenizer:r1"}
	previous := measuredTokens(t, identity, "epoch-1", []int{101, 202, 303, 404}, 1)
	for _, test := range []struct {
		name    string
		tokens  []int
		prefix  int
		overlap int
	}{
		{"extended", []int{101, 202, 303, 404, 505, 606}, 5, 4},
		{"changed within prefix", []int{101, 202, 707, 404, 505}, 4, 2},
		{"changed first token", []int{999, 202}, 1, 0},
		{"new user boundary caps overlap", []int{101, 202, 303, 404}, 2, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			current := measuredTokens(t, identity, "epoch-1", test.tokens, test.prefix)
			got := current.Prefix(previous)
			if got.Tokens != test.prefix || got.OverlapTokens == nil || *got.OverlapTokens != test.overlap || got.Provenance != "local_measured" {
				t.Fatalf("prefix = %+v", got)
			}
		})
	}
}

func TestTokenPrefixUnknownAcrossRuntimeUnitsOrEpoch(t *testing.T) {
	identity := PromptIdentity{RuntimeIdentity: "runtime-r1", Model: "worker", Protocol: "chat", Units: "tokenizer:r1"}
	previous := measuredTokens(t, identity, "epoch-1", []int{1, 2, 3}, 2)
	for _, field := range []string{"runtime", "model", "protocol", "units", "epoch"} {
		t.Run(field, func(t *testing.T) {
			changed, epoch := identity, "epoch-1"
			switch field {
			case "runtime":
				changed.RuntimeIdentity = "runtime-r2"
			case "model":
				changed.Model = "other"
			case "protocol":
				changed.Protocol = "messages"
			case "units":
				changed.Units = "tokenizer:r2"
			case "epoch":
				epoch = "epoch-2"
			}
			got := measuredTokens(t, changed, epoch, []int{1, 2, 3, 4}, 3).Prefix(previous)
			if got.OverlapTokens != nil {
				t.Fatal("incompatible measurement claimed overlap")
			}
		})
	}
}

func TestCountOnlyEvidencePreservesProvenanceAndUnknownOverlap(t *testing.T) {
	prefix := 100
	measured, err := admitMeasurement(PreparedPrompt{}, sha256.Sum256(nil), PromptMeasurement{PromptTokens: 120, CarriedPrefixTokens: &prefix, Provenance: "provider_reported"})
	if err != nil {
		t.Fatal(err)
	}
	got := measured.Prefix(measured)
	if got.OverlapTokens != nil || got.Provenance != "provider_reported" {
		t.Fatalf("count-only evidence invented token overlap or provenance: %+v", got)
	}
	for _, measurement := range []PromptMeasurement{
		{PromptTokens: 120, CarriedPrefixTokens: &prefix, Provenance: "caller_estimated"},
		{PromptTokens: 99, CarriedPrefixTokens: &prefix, Provenance: "local_measured"},
		{PromptTokens: 2, TokenIDs: []int{1}, Provenance: "local_measured"},
		{PromptTokens: 1, TokenIDs: []int{-1}, Provenance: "local_measured"},
	} {
		if _, err := admitMeasurement(PreparedPrompt{}, sha256.Sum256(nil), measurement); err == nil {
			t.Fatal("invalid or estimated token evidence admitted")
		}
	}
}
