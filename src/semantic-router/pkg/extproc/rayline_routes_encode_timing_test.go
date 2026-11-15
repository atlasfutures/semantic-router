/*
Copyright 2025 The vLLM Semantic Router Authors.

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
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A two-stage package answers timing_ms.encode null: the decide still
// carries the service's own total, which includes the encode, and the
// cell's round trip; no encode time is invented.
func TestPolicyDecideWithoutAnEncodeTimeKeepsTheServiceTotal(t *testing.T) {
	for _, tc := range []struct {
		name       string
		unreported bool
		encode     time.Duration
	}{
		{name: "single-stage", encode: 41500 * time.Microsecond},
		{name: "two-stage", unreported: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
			fixture.fake.encodeUnreported = tc.unreported
			advertiseLookupSessions(fixture)
			result, err := fixture.scopedLookupBy(t, lookupScopeTestScope, time.Now().Add(5*time.Second))
			if err != nil {
				t.Fatal(err)
			}
			trace := result.RaylineARC
			if trace.EncoderLatency != tc.encode || trace.EncoderLatencyUnknown != tc.unreported {
				t.Fatalf("encoder latency = %v (unknown %t), want %v (unknown %t)",
					trace.EncoderLatency, trace.EncoderLatencyUnknown, tc.encode, tc.unreported)
			}
			// decision_response.v1.json: timing_ms.total 47.2.
			if trace.PolicyServiceTotal != 47200*time.Microsecond || trace.PolicyLatency <= 0 {
				t.Fatalf("service total = %v, round trip = %v", trace.PolicyServiceTotal, trace.PolicyLatency)
			}
		})
	}
}

// The lookup's routing_decision carries the decide's round trip and the
// service's total beside the encode time, and the encode time only when
// the encoder measured one.
func TestRouteDecisionCarriesThePolicyTimings(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.policyLatency, fixture.selector.serviceTotal = 640*time.Millisecond, 610*time.Millisecond
	decision := fixture.decide(t, routerruntime.RouteDecisionRequest{
		Surface: routerruntime.RouteDecisionSurfaceRoutes, Ephemeral: true, SessionID: "conv-1", LookupScope: lookupScopeTestScope,
	})
	if decision.Encode.PolicyRoundTrip != 640*time.Millisecond || decision.Encode.ServiceTotal != 610*time.Millisecond ||
		decision.Encode.Latency != 0 {
		t.Fatalf("encode = %+v", decision.Encode)
	}
	fields := map[string]interface{}{}
	addRaylineRoutesEncodeFields(fields, decision.Encode)
	if fields["policy_ms"] != int64(640) || fields["service_ms"] != int64(610) || fields["session_action"] != "appended" {
		t.Fatalf("fields = %v", fields)
	}
	if _, ok := fields["encode_ms"]; ok {
		t.Fatalf("encode_ms logged without a measured encode: %v", fields)
	}
	fields = map[string]interface{}{}
	addRaylineRoutesEncodeFields(fields, routerruntime.RouteEncode{Latency: 42 * time.Millisecond})
	if fields["encode_ms"] != int64(42) {
		t.Fatalf("fields = %v", fields)
	}
	if _, ok := fields["policy_ms"]; ok {
		t.Fatalf("policy_ms logged outside a policy decide: %v", fields)
	}
}
