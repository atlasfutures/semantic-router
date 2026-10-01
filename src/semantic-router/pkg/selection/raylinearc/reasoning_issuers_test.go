package raylinearc

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

const (
	issuerA = "aaaaaaaaaaaaaaaa"
	issuerB = "bbbbbbbbbbbbbbbb"
)

func TestNextReasoningIssuers(t *testing.T) {
	cases := map[string]struct {
		previous                []string
		held, forwarded, issues bool
		want                    []string
	}{
		// A turn that resends nothing restarts from what its reply can add.
		"nothing held, Responses target":    {[]string{issuerA, issuerB}, false, false, true, []string{issuerB}},
		"nothing held, other target":        {[]string{issuerA}, false, false, false, nil},
		"forwarded to its issuer":           {[]string{issuerB}, true, true, true, []string{issuerB}},
		"dropped on a Chat or Messages arm": {[]string{issuerA}, true, false, false, []string{issuerA}},
		"dropped on another Responses arm":  {[]string{issuerA}, true, false, true, []string{issuerA, issuerB}},
		"held blobs of no known issuer":     {nil, true, false, true, []string{ReasoningIssuerUnknown, issuerB}},
		"unknown blobs on a Chat arm":       {nil, true, false, false, []string{ReasoningIssuerUnknown}},
		"already mixed stays mixed":         {[]string{issuerA, ReasoningIssuerUnknown}, true, false, true, []string{issuerA, ReasoningIssuerUnknown}},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			issuer := issuerB
			got := NextReasoningIssuers(test.previous, test.held, test.forwarded, test.issues, issuer)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("next = %v, want %v", got, test.want)
			}
			if err := validateReasoningIssuers(got); err != nil {
				t.Fatalf("next set is not storable: %v", err)
			}
		})
	}
}

func TestReasoningIssuersAreExactlyOneTarget(t *testing.T) {
	for _, issuers := range [][]string{nil, {issuerA}, {issuerB, issuerA}, {ReasoningIssuerUnknown}} {
		if ReasoningIssuersAre(issuers, issuerB) {
			t.Fatalf("%v forwards to %s", issuers, issuerB)
		}
	}
	// The unknown issuer is no target, so its blobs never travel.
	if ReasoningIssuersAre([]string{ReasoningIssuerUnknown}, ReasoningIssuerUnknown) {
		t.Fatal("blobs of unknown provenance are forwarded")
	}
	if !ReasoningIssuersAre([]string{issuerB}, issuerB) {
		t.Fatal("the target's own blobs are not forwarded to it")
	}
}

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

	state.ReasoningIssuers = []string{issuerA, ReasoningIssuerUnknown}
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
		"too many":     {issuerA, issuerB, ReasoningIssuerUnknown},
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
