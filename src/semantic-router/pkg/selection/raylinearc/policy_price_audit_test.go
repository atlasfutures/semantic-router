/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

func testPriceAudit(t *testing.T) *PolicyPriceAudit {
	t.Helper()
	response, err := DecodePolicyDecisionResponse(readRoutingPriceFixture(t, "decision_response_live.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	cost := 0.007
	return NewPolicyPriceAudit(response.Package.PackageSHA256, response.RoutingPricesUsed, &cost)
}

func TestPolicyPriceAuditRecordsTheEcho(t *testing.T) {
	audit := testPriceAudit(t)
	want := &PolicyPriceAudit{
		PackageSHA256: "81e4a7af2c28c3be9ecbe850888c6773d196d58cad100474cbfb48622af7abe4",
		Mode:          RoutingPricesModeLive, StoreVersion: "v000000000042", AsOf: "2026-10-14T16:00:00Z",
		Digest:           "3b4c1f0e5a6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f",
		FallbackArms:     []string{"4951fcf2b0d2c10050e8d69681e8deda04f2e3a7b5c3e8b2945bf291be9b9279"},
		FallbackArmCount: 1, PredictedCostUSD: audit.PredictedCostUSD,
		FilledLegs: map[string][]string{"z-ai/glm-4.6": {"cache_read", "cache_write"}}, FilledModelCount: 1,
	}
	if !reflect.DeepEqual(audit, want) || *audit.PredictedCostUSD != 0.007 {
		t.Fatalf("audit = %+v", audit)
	}
	if NewPolicyPriceAudit(want.PackageSHA256, nil, nil) != nil {
		t.Fatal("a response with no echo produced an audit")
	}
	arms := make([]string, 20)
	for index := range arms {
		arms[index] = fmt.Sprintf("%064x", index)
	}
	many := NewPolicyPriceAudit(want.PackageSHA256, &PolicyRoutingPricesUsed{Mode: RoutingPricesModePackage, FallbackArms: arms}, nil)
	if len(many.FallbackArms) != maxPolicyAuditFallbackArms || many.FallbackArmCount != 20 {
		t.Fatalf("20 fallback arms recorded as %d, count %d", len(many.FallbackArms), many.FallbackArmCount)
	}
	if many.FilledLegs != nil || many.FilledModelCount != 0 {
		t.Fatalf("nothing filled recorded as %v, count %d", many.FilledLegs, many.FilledModelCount)
	}

	// Filled legs keep the first 8 models in id order and the true count; an
	// id past the ledger's bound is counted but not listed.
	filled := map[string][]string{"a/" + strings.Repeat("x", maxPolicyAuditFilledModelBytes): {"input"}}
	for index := 18; index >= 0; index-- {
		filled[fmt.Sprintf("vendor/model-%02d", index)] = []string{"cache_write"}
	}
	wide := NewPolicyPriceAudit(want.PackageSHA256, &PolicyRoutingPricesUsed{Mode: RoutingPricesModeLive, FilledLegs: filled}, nil)
	if len(wide.FilledLegs) != maxPolicyAuditFilledModels || wide.FilledModelCount != 20 ||
		wide.FilledLegs["vendor/model-00"] == nil || wide.FilledLegs["vendor/model-07"] == nil ||
		wide.FilledLegs["vendor/model-08"] != nil || wide.validate() != nil {
		t.Fatalf("20 filled models recorded as %v, count %d", wide.FilledLegs, wide.FilledModelCount)
	}
}

// Only the newest MaxPolicyPriceAudits entries keep their audit.
func TestPolicyLedgerKeepsOnlyTheNewestPriceAudits(t *testing.T) {
	audit := testPriceAudit(t)
	state := &PolicyEpisodeState{}
	var messages []json.RawMessage
	for turn := 0; turn < MaxPolicyPriceAudits+3; turn++ {
		messages = append(messages, json.RawMessage(user1))
		state = state.NextWithPrices(messages, policyTestActionA, "arm-a", audit)
		messages = append(messages, json.RawMessage(asst1))
	}
	state = state.NextWithPrices(messages, policyTestActionA, "arm-a", nil)
	if err := state.Validate(); err != nil {
		t.Fatal(err)
	}
	audited := 0
	for index, entry := range state.Ledger {
		if entry.Prices == nil {
			continue
		}
		audited++
		if index < len(state.Ledger)-1-MaxPolicyPriceAudits {
			t.Fatalf("entry %d of %d kept its audit", index, len(state.Ledger))
		}
	}
	if audited != MaxPolicyPriceAudits {
		t.Fatalf("%d audits kept, want %d", audited, MaxPolicyPriceAudits)
	}
}

// The audit round-trips through the persisted episode record, and a record
// written before live prices still reads and rewrites byte for byte.
func TestEpisodeStateRoundTripsPriceAuditsAndReadsOldRecords(t *testing.T) {
	now := time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)
	old := `{"schema_version":"rayline.arc.episode-state.v3","version":3,"previous_arm":null,"turn_index":0,` +
		`"warmth":[null],"encoder_owner":"","encoder_visited_owners":[],"policy":{"epoch":1,"epoch_start_turn":0,` +
		`"prefix_len":2,"prefix_digest":"` + strings.Repeat("a", 64) + `","ledger":[{"message":1,"action_id":"` +
		strings.Repeat("b", 64) + `","arm_id":"arm"}]}}`
	decoded, version, err := unmarshalEpisodeState([]byte(old), 1, now)
	if err != nil || version != 3 || decoded.Policy.Ledger[0].Prices != nil {
		t.Fatalf("old record: %+v, %v", decoded, err)
	}
	rewritten, err := marshalEpisodeState(decoded, 3, now)
	if err != nil || string(rewritten) != old {
		t.Fatalf("old record rewritten as\n%s\nwant\n%s (%v)", rewritten, old, err)
	}

	decoded.Policy.Ledger[0].Prices = testPriceAudit(t)
	payload, err := marshalEpisodeState(decoded, 4, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"prices":{"package_sha256":"81e4a7af`) {
		t.Fatalf("audit not persisted: %s", payload)
	}
	again, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(again.Policy, decoded.Policy) {
		t.Fatalf("audit did not round-trip: %+v", again.Policy.Ledger[0].Prices)
	}
}

// A full ledger of 64-hex arm ids with the largest audits it can keep still
// fits the episode size limit.
func TestFullLedgerWithPriceAuditsFitsTheEpisodeLimit(t *testing.T) {
	now := time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)
	arms := make([]string, maxPolicyAuditFallbackArms)
	for index := range arms {
		arms[index] = strings.Repeat("c", maxPolicyArmIDBytes)
	}
	cost := 1.0
	filled := map[string][]string{}
	for index := 0; index < maxPolicyAuditFilledModels; index++ {
		model := fmt.Sprintf("%d", index) + strings.Repeat("m", maxPolicyAuditFilledModelBytes-1)
		filled[model] = RoutingPriceLegs[:]
	}
	audit := &PolicyPriceAudit{
		PackageSHA256: strings.Repeat("a", 64), Mode: RoutingPricesModeLive,
		StoreVersion: strings.Repeat("v", maxRoutingPriceStoreVersionBytes),
		AsOf:         strings.Repeat("t", maxRoutingPriceTimestampBytes), Digest: strings.Repeat("d", 64),
		FallbackArms: arms, FallbackArmCount: maxRoutingPriceFallbackArms, PredictedCostUSD: &cost,
		FilledLegs: filled, FilledModelCount: maxRoutingPriceModels,
	}
	policy := &PolicyEpisodeState{PrefixLen: MaxPolicyLedgerEntries * 1000, PrefixDigest: strings.Repeat("a", 64)}
	for index := 0; index < MaxPolicyLedgerEntries; index++ {
		entry := PolicyLedgerEntry{Message: 100000 + index, ActionID: strings.Repeat("b", 64), ArmID: strings.Repeat("e", 64)}
		if index >= MaxPolicyLedgerEntries-MaxPolicyPriceAudits {
			entry.Prices = audit
		}
		policy.Ledger = append(policy.Ledger, entry)
	}
	state, err := NewEpisodeState(8)
	if err != nil {
		t.Fatal(err)
	}
	state.Policy = policy
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("full audited ledger: %d of %d bytes", len(payload), maxEpisodeStateBytes)
	if len(payload) > maxEpisodeStateBytes*7/8 {
		t.Fatalf("a full audited ledger takes %d of %d bytes", len(payload), maxEpisodeStateBytes)
	}
}

func TestPolicyStateRefusesMalformedPriceAudits(t *testing.T) {
	for name, mutate := range map[string]func(*PolicyPriceAudit){
		"an unknown mode":              func(a *PolicyPriceAudit) { a.Mode = "snapshot" },
		"a malformed package":          func(a *PolicyPriceAudit) { a.PackageSHA256 = "pkg" },
		"a malformed digest":           func(a *PolicyPriceAudit) { a.Digest = "digest" },
		"a count below the list":       func(a *PolicyPriceAudit) { a.FallbackArmCount = 0 },
		"an empty fallback arm":        func(a *PolicyPriceAudit) { a.FallbackArms = []string{""} },
		"an overlong store label":      func(a *PolicyPriceAudit) { a.StoreVersion = strings.Repeat("v", 300) },
		"a filled count below the map": func(a *PolicyPriceAudit) { a.FilledModelCount = 0 },
		"an unknown filled leg":        func(a *PolicyPriceAudit) { a.FilledLegs = map[string][]string{"m": {"tokens"}} },
		"an empty filled model":        func(a *PolicyPriceAudit) { a.FilledLegs = map[string][]string{"": {"input"}} },
	} {
		audit := *testPriceAudit(t)
		mutate(&audit)
		state := &PolicyEpisodeState{
			PrefixLen: 2, PrefixDigest: policyTestActionA,
			Ledger: []PolicyLedgerEntry{{Message: 1, ActionID: policyTestActionA, ArmID: "arm", Prices: &audit}},
		}
		if err := state.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	audit := testPriceAudit(t)
	state := &PolicyEpisodeState{PrefixLen: 2, PrefixDigest: policyTestActionA}
	for index := 0; index <= MaxPolicyPriceAudits; index++ {
		state.Ledger = append(state.Ledger, PolicyLedgerEntry{Message: 1, ActionID: policyTestActionA, ArmID: "arm", Prices: audit})
	}
	if err := state.Validate(); err == nil {
		t.Error("a ledger keeping too many audits was accepted")
	}
}
