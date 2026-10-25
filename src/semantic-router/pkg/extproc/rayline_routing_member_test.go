package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// The routing member leaves at the byte level: every other byte stays as the
// client sent it, duplicates of other members included, so the codec's
// duplicate_json_field check still sees them (GrayHorse P2 on #305).
func TestRemoveTopLevelMemberKeepsEveryOtherByte(t *testing.T) {
	for name, tc := range map[string]struct{ in, want string }{
		"first":                {`{"routing":{"value_usd":2},"model":"m","x":1}`, `{"model":"m","x":1}`},
		"middle":               {`{"model":"m","routing":{"value_usd":2},"x":1}`, `{"model":"m","x":1}`},
		"last":                 {`{"model":"m","x":1,"routing":{"value_usd":2}}`, `{"model":"m","x":1}`},
		"only":                 {`{"routing":{}}`, `{}`},
		"spaced":               {"{ \"model\" : \"m\" ,\n \"routing\" : null , \"x\" : 1 }", "{ \"model\" : \"m\" , \"x\" : 1 }"},
		"first, spaced":        {"{\n  \"routing\": {},\n  \"model\": \"m\"\n}", "{\n  \"model\": \"m\"\n}"},
		"nested is not top":    {`{"model":"m","messages":[{"routing":1}],"routing":{}}`, `{"model":"m","messages":[{"routing":1}]}`},
		"duplicate kept":       {`{"max_tokens":1,"routing":{},"max_tokens":2}`, `{"max_tokens":1,"max_tokens":2}`},
		"string mentions only": {`{"model":"routing","text":"\"routing\":{}"}`, `{"model":"routing","text":"\"routing\":{}"}`},
	} {
		t.Run(name, func(t *testing.T) {
			got, _, err := removeTopLevelMember([]byte(tc.in), "routing")
			if err != nil || string(got) != tc.want {
				t.Fatalf("got %q (%v), want %q", got, err, tc.want)
			}
		})
	}
	if _, _, err := removeTopLevelMember([]byte(`{"routing":{},"routing":{"value_usd":2}}`), "routing"); !errors.Is(err, errRoutingMemberRepeated) {
		t.Fatalf("a repeated routing member: %v", err)
	}
	if refusal := func() *routesContractError {
		_, r := stripRoutingMember([]byte(`{"routing":{},"routing":{}}`))
		return r
	}(); refusal == nil || refusal.code != "routing_invalid" {
		t.Fatalf("repeated routing refusal %+v", refusal)
	}
}

// A generation turn keeps the body's own duplicate fields after the strip.
func TestGenerationStripKeepsDuplicateFieldsForTheCodec(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":1,"routing":{"value_usd":2},"max_tokens":9,"messages":[{"role":"user","content":"x"}]}`)
	_, stripped, refusal := readRaylineGenerationRouting(body, generationPolicy(true, nil))
	if refusal != nil || string(stripped) != `{"model":"rayline/arc-dev","max_tokens":1,"max_tokens":9,"messages":[{"role":"user","content":"x"}]}` {
		t.Fatalf("stripped %s (%+v)", stripped, refusal)
	}
}

// value_usd is applied only when the decision was taken at it: a capacity or
// media hold decides nothing (GrayHorse P2 on #305).
func TestGenerationValueIsUnappliedOnAHeldTurn(t *testing.T) {
	value, weight := 2.0, 0.5
	routing := &raylineGenerationRouting{valueUSD: &value, costWeight: &weight}
	for name, tc := range map[string]struct {
		trace   selection.RaylineARCTrace
		applied bool
	}{
		"decided at the weight": {selection.RaylineARCTrace{PolicyActionID: "a", TurnCostWeightApplied: true}, true},
		"capacity hold":         {selection.RaylineARCTrace{PolicyActionID: "a", OfferKind: "capacity_hold"}, false},
		"media hold":            {selection.RaylineARCTrace{PolicyActionID: "a", OfferKind: "media_hold"}, false},
	} {
		trace := tc.trace
		in, out := raylineGenerationRoutingStatus(&RequestContext{RaylineGenerationRouting: routing, VSRRaylineARC: &trace})
		if (len(in) == 1 && in[0] == "value_usd") != tc.applied || (len(out) == 1) == tc.applied {
			t.Errorf("%s: applied %v unapplied %v", name, in, out)
		}
	}
	// A tier cell restates its own V: applied on a decided turn, not on a hold.
	tier := &raylineGenerationRouting{valueUSD: &value}
	decided := selection.RaylineARCTrace{PolicyActionID: "a"}
	if in, _ := raylineGenerationRoutingStatus(&RequestContext{RaylineGenerationRouting: tier, VSRRaylineARC: &decided}); len(in) != 1 {
		t.Errorf("a tier cell's decided turn did not apply its V: %v", in)
	}
	held := selection.RaylineARCTrace{PolicyActionID: "a", OfferKind: "capacity_hold"}
	if in, _ := raylineGenerationRoutingStatus(&RequestContext{RaylineGenerationRouting: tier, VSRRaylineARC: &held}); len(in) != 0 {
		t.Errorf("a tier cell's held turn applied its V: %v", in)
	}
}
