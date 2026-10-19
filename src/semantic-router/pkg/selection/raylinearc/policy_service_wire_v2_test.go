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
	"testing"
)

// pathfinder#5443: price_per_mtok, thinking and tier_picks decode strictly,
// and a null predicted-usage leg stays nil rather than becoming 0.
func TestDecodePolicyDecisionResponseDecisionsV2(t *testing.T) {
	response, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response_decisions_v2.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Decision.TierPicks) != 3 || response.Decision.StageOne != nil {
		t.Fatalf("tier picks %v, stage one %v", response.Decision.TierPicks, response.Decision.StageOne)
	}
	sage := response.Decision.TierPicks["sage"]
	if sage.Model != "claude-opus-5" || sage.SelectedArmID == "" || sage.Thinking.Effort == nil || *sage.Thinking.Effort != "high" ||
		sage.Thinking.BudgetTokens != nil || sage.Thinking.ControlID != nil {
		t.Fatalf("sage tier pick %+v", sage)
	}
	unsupported := response.Actions[2]
	usage := unsupported.PredictedUsage
	if usage == nil || usage.CacheWrite != nil || usage.UncachedInput == nil || *usage.UncachedInput != 100000 {
		t.Fatalf("predicted usage %+v", usage)
	}
	price := unsupported.PricePerMtok
	if price == nil || price.CacheWrite == nil || *price.CacheWrite != 6.25 || unsupported.Thinking == nil {
		t.Fatalf("price %+v, thinking %+v", price, unsupported.Thinking)
	}
}

// pathfinder#5436: decision.stage_one.exploration decodes strictly, with its
// nullable members (a null score, a null greedy/chosen) kept nil.
func TestDecodePolicyDecisionResponseExploration(t *testing.T) {
	response, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response_exploration.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if response.Decision.StageOne == nil || response.Decision.StageOne.Exploration == nil {
		t.Fatal("no stage-one exploration record")
	}
	record := response.Decision.StageOne.Exploration
	if !record.Applied || record.Reason != nil || record.Tau != 0.02 || record.U == nil || !record.Explored ||
		record.Basis.ReadoutPin == "" || record.DecisionKey == nil || record.OriginDecisionKey != nil ||
		record.Override == nil || *record.Override != "proxy_remap" {
		t.Fatalf("exploration record %+v", record)
	}
	if len(record.Offered) != 3 || len(record.Scores) != 3 || len(record.Pi) != 3 || record.Scores[2] != nil {
		t.Fatalf("offered %v, scores %v, pi %v", record.Offered, record.Scores, record.Pi)
	}
	if effort := record.Chosen["effort"]; effort != nil || *record.Chosen["model"] != "glm-5.3-flash" {
		t.Fatalf("chosen %v", record.Chosen)
	}
	if record.Propensity != record.Pi[1] {
		t.Fatalf("propensity %v, pi %v", record.Propensity, record.Pi)
	}
	if response.Actions[2].PredictedUsage != nil {
		t.Fatal("a null predicted_usage decoded to a value")
	}

	// A caller-held greedy record: null greedy, chosen, u and override.
	held, err := DecodePolicyDecisionResponse(fixtureWith(t, "decision_response_exploration.v1.json", func(body map[string]any) {
		stageOne := body["decision"].(map[string]any)["stage_one"].(map[string]any)
		exploration := stageOne["exploration"].(map[string]any)
		for _, key := range []string{"greedy", "chosen", "u", "override", "decision_key"} {
			exploration[key] = nil
		}
		exploration["reason"] = "caller_held"
	}))
	if err != nil {
		t.Fatal(err)
	}
	greedy := held.Decision.StageOne.Exploration
	if greedy.Greedy != nil || greedy.Chosen != nil || greedy.U != nil || greedy.Override != nil ||
		greedy.DecisionKey != nil || greedy.Reason == nil || *greedy.Reason != "caller_held" {
		t.Fatalf("greedy record %+v", greedy)
	}
}

// A v1 response carries none of the opt-in blocks.
func TestDecodePolicyDecisionResponseV1HasNoOptInBlocks(t *testing.T) {
	response, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if response.Decision.TierPicks != nil || response.Decision.StageOne != nil {
		t.Fatal("a v1 response decoded an opt-in decision block")
	}
	for _, action := range response.Actions {
		if action.PricePerMtok != nil || action.Thinking != nil {
			t.Fatalf("action %s decoded an opt-in action block", action.ActionID)
		}
	}
}

// The strict decoder still refuses an unknown member, including inside each
// new block.
func TestDecodePolicyDecisionResponseRefusesUnknownMembersInOptInBlocks(t *testing.T) {
	cases := map[string]struct {
		fixture string
		edit    func(map[string]any)
	}{
		"top level": {"decision_response_decisions_v2.v1.json", func(body map[string]any) { body["surprise"] = 1 }},
		"price_per_mtok": {"decision_response_decisions_v2.v1.json", func(body map[string]any) {
			action := body["actions"].([]any)[0].(map[string]any)
			action["price_per_mtok"].(map[string]any)["surprise"] = 1
		}},
		"thinking": {"decision_response_decisions_v2.v1.json", func(body map[string]any) {
			action := body["actions"].([]any)[0].(map[string]any)
			action["thinking"].(map[string]any)["surprise"] = 1
		}},
		"predicted_usage": {"decision_response_decisions_v2.v1.json", func(body map[string]any) {
			action := body["actions"].([]any)[0].(map[string]any)
			action["predicted_usage"].(map[string]any)["surprise"] = 1
		}},
		"tier pick": {"decision_response_decisions_v2.v1.json", func(body map[string]any) {
			picks := body["decision"].(map[string]any)["tier_picks"].(map[string]any)
			picks["swift"].(map[string]any)["surprise"] = 1
		}},
		"tier pick thinking": {"decision_response_decisions_v2.v1.json", func(body map[string]any) {
			picks := body["decision"].(map[string]any)["tier_picks"].(map[string]any)
			picks["swift"].(map[string]any)["thinking"].(map[string]any)["surprise"] = 1
		}},
		"stage_one": {"decision_response_exploration.v1.json", func(body map[string]any) {
			body["decision"].(map[string]any)["stage_one"].(map[string]any)["surprise"] = 1
		}},
		"exploration": {"decision_response_exploration.v1.json", func(body map[string]any) {
			stageOne := body["decision"].(map[string]any)["stage_one"].(map[string]any)
			stageOne["exploration"].(map[string]any)["surprise"] = 1
		}},
		"exploration basis": {"decision_response_exploration.v1.json", func(body map[string]any) {
			stageOne := body["decision"].(map[string]any)["stage_one"].(map[string]any)
			stageOne["exploration"].(map[string]any)["basis"].(map[string]any)["surprise"] = 1
		}},
	}
	for name, tc := range cases {
		if _, err := DecodePolicyDecisionResponse(fixtureWith(t, tc.fixture, tc.edit)); err == nil {
			t.Errorf("%s: a response with an unknown member was accepted", name)
		}
	}
}
