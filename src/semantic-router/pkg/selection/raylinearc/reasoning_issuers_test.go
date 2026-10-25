package raylinearc

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
)

const (
	issuerA = "aaaaaaaaaaaaaaaa"
	issuerB = "bbbbbbbbbbbbbbbb"
)

// The issuer set is v3 and written only when present, so an episode without
// one keeps the bytes every earlier build decodes, and an earlier record
// decodes with no issuers.
func TestEpisodeStateWireCarriesReasoningIssuersUnderV3(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(plain), "reasoning_issuers") || !strings.Contains(string(plain), "episode-state.v2") {
		t.Fatalf("an episode without issuers changed shape: %s", plain)
	}
	if decoded, _, err := unmarshalEpisodeState(plain, 1, now); err != nil || decoded.ReasoningIssuers != nil {
		t.Fatalf("an earlier record decoded as %v, %v", decoded, err)
	}

	state.ReasoningIssuers = []string{issuerA, reasoning.ReasoningIssuerUnknown}
	payload, err := marshalEpisodeState(state, 2, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v3"`) {
		t.Fatalf("issuers written without v3: %s", payload)
	}
	decoded, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.ReasoningIssuers, state.ReasoningIssuers) {
		t.Fatalf("issuers did not round-trip: %v", decoded.ReasoningIssuers)
	}
	v2 := strings.Replace(string(payload), "episode-state.v3", "episode-state.v2", 1)
	if _, _, err := unmarshalEpisodeState([]byte(v2), 1, now); err == nil {
		t.Fatal("a v2 record carrying issuers was accepted")
	}
}

func TestEpisodeStateRefusesInvalidReasoningIssuers(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	for name, issuers := range map[string][]string{
		"too many":     {issuerA, issuerB, reasoning.ReasoningIssuerUnknown},
		"duplicated":   {issuerA, issuerA},
		"not a digest": {"worker-name"},
		"empty":        {""},
	} {
		t.Run(name, func(t *testing.T) {
			state, err := NewEpisodeState(1)
			if err != nil {
				t.Fatal(err)
			}
			state.ReasoningIssuers = issuers
			if _, err := marshalEpisodeState(state, 1, now); err == nil {
				t.Fatalf("%v was stored", issuers)
			}
		})
	}
}
