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

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A call whose wire states no TTL the Router reads takes its serving
// provider's documented lifetime, labelled provider_policy; a wire TTL, an
// unlisted provider, a body asking for another lifetime, or a cell without a
// policy leaves it as it was.
func TestProviderCacheTTLFallsBackToTheDocumentedLifetimeOnly(t *testing.T) {
	policy := &config.RaylineARCPolicyServiceConfig{CacheContext: true, CacheTTLPolicy: []config.RaylineARCCacheTTLPolicyRow{
		{Provider: "Moonshot AI", TTLSeconds: 300, Source: "kimi docs 5m idle, read 2026-10-21"},
	}}
	cases := []struct {
		name      string
		policy    *config.RaylineARCPolicyServiceConfig
		wireTTL   int
		body      bool
		bodyTTL   int
		provider  string
		wantTTL   int
		wantFound bool
	}{
		{"documented, no body directive", policy, 0, false, 0, "Moonshot AI", 300, true},
		{"documented, body states the same lifetime", policy, 0, true, 300, "Moonshot AI", 300, true},
		{"body asks for another lifetime", policy, 0, true, 3600, "Moonshot AI", 0, false},
		{"body directive the Router cannot read", policy, 0, true, 0, "Moonshot AI", 0, false},
		{"undocumented provider", policy, 0, false, 0, "Modal", 0, false},
		{"provider unreported", policy, 0, false, 0, "", 0, false},
		{"wire TTL kept", policy, 3600, false, 0, "Moonshot AI", 3600, false},
		{"cell sends no cache_context", nil, 0, false, 0, "Moonshot AI", 0, false},
	}
	for _, tc := range cases {
		transaction := &raylineARCEpisodeTransaction{cachePolicy: tc.policy, servedModel: "kimi-k3"}
		if tc.wireTTL > 0 {
			transaction.stageWireCacheTTL(tc.wireTTL, "anthropic cache_control ephemeral 1h")
		} else {
			transaction.stageWireCacheTTL(0, "")
			transaction.stageBodyCacheControl(tc.body, tc.bodyTTL)
		}
		transaction.stageProviderCacheTTL(tc.provider)
		call := transaction.call
		if call.TTLSeconds != tc.wantTTL {
			t.Fatalf("%s: ttl = %d, want %d", tc.name, call.TTLSeconds, tc.wantTTL)
		}
		documented := strings.HasPrefix(call.TTLSource, raylinearc.ProviderPolicyTTLSourcePrefix)
		if documented != tc.wantFound {
			t.Fatalf("%s: source %q", tc.name, call.TTLSource)
		}
		if documented && call.TTLSource != "provider_policy: kimi docs 5m idle, read 2026-10-21" {
			t.Fatalf("%s: source %q", tc.name, call.TTLSource)
		}
	}
	var none *raylineARCEpisodeTransaction
	none.stageProviderCacheTTL("Moonshot AI")
	none.stageBodyCacheControl(true, 300)
}

// The cell's user_turn_cache travels as selection.user_turn_cache, and an
// empty one is omitted so the package's own mode decides.
func TestPolicySelectionCarriesTheCellsUserTurnCache(t *testing.T) {
	for _, mode := range []string{"", config.RaylineARCUserTurnCacheRepartitionTTL} {
		body, err := json.Marshal(raylinearc.PolicySelection{AvailableActionIDs: []string{"a"}, UserTurnCache: mode})
		if err != nil {
			t.Fatal(err)
		}
		var members map[string]json.RawMessage
		if err := json.Unmarshal(body, &members); err != nil {
			t.Fatal(err)
		}
		sent, present := members["user_turn_cache"]
		if present != (mode != "") || present && string(sent) != `"`+mode+`"` {
			t.Fatalf("mode %q: user_turn_cache %s present=%v", mode, sent, present)
		}
	}
}

// A cell's user_turn_cache reaches the decide request.
func TestPolicyDecideSendsTheCellsUserTurnCache(t *testing.T) {
	fixture := sendingCacheContext(newPolicySelectorFixture(t, ""))
	fixture.decision.Algorithm.RaylineARC.PolicyService.UserTurnCache = config.RaylineARCUserTurnCacheRepartitionTTL
	armed := *fixture.selector.armed.Load()
	armed.scorer = newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	fixture.selector.arm(&armed)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	state, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select: %v", err)
	}
	if got := fixture.fake.received()[0].Selection.UserTurnCache; got != config.RaylineARCUserTurnCacheRepartitionTTL {
		t.Fatalf("selection.user_turn_cache = %q", got)
	}
}
