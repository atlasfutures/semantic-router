package raylinearc

import (
	"strings"
	"testing"
)

// Both exploration shapes reach the service with '<', '>' and '&' as sent,
// as the decide body's MarshalUnescaped writes every other field (GrayHorse
// follow-up on #303): the legacy wire was byte-identical before the
// two-shape MarshalJSON, and stays so.
func TestExplorationRequestIsWrittenWithoutHTMLEscaping(t *testing.T) {
	legacy := PolicyExplorationRequest{
		Tau: 0.02, Salt: "a<b>&c",
		Basis: PolicyExplorationBasis{R: 0.05, PackageSHA256: strings.Repeat("a", 64), ReadoutPin: "pin&<x>"},
	}
	temperature := PolicyExplorationRequest{Salt: "a<b>&c", Temperature: &PolicyExplorationTemperature{
		Temperature: 0.5, Surface: "hosted", Source: "default", TrainingData: "allowed",
		Calibration: &PolicyExplorationBasis{R: 0.05, PackageSHA256: strings.Repeat("a", 64), ReadoutPin: "pin&<x>"},
	}}
	for name, request := range map[string]PolicyExplorationRequest{"legacy": legacy, "temperature": temperature} {
		encoded, err := request.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(encoded), "\\u00") || !strings.Contains(string(encoded), `"a<b>&c"`) ||
			!strings.Contains(string(encoded), `"pin&<x>"`) {
			t.Fatalf("%s: %s", name, encoded)
		}
		// Through the whole decide body too.
		body, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{
			Selection: PolicySelection{AvailableActionIDs: []string{"x"}, Exploration: &request},
			Request:   PolicyClientRequest{Messages: []byte(`[{"role":"user","content":"hi"}]`)},
		})
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), "\\u00") || !strings.Contains(string(body), `"a<b>&c"`) {
			t.Fatalf("%s decide body: %s", name, body)
		}
	}
}
