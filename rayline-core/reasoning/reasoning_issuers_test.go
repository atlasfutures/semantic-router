package reasoning

import (
	"reflect"
	"testing"
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
			if err := ValidateReasoningIssuers(got); err != nil {
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
