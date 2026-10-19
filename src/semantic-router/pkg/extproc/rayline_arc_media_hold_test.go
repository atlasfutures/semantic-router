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
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// mediaRefusal makes every later decide call answer the projection's typed
// media refusal (pathfinder#5281).
func mediaRefusal(fixture *policySelectorFixture, subReason any) {
	fixture.fake.mu.Lock()
	defer fixture.fake.mu.Unlock()
	fixture.fake.failWith = "unsupported_request"
	fixture.fake.failDetail = map[string]any{"reason": "media_refused", "sub_reason": subReason}
}

// pathfinder#5281: media the projection cannot read stays in the
// conversation, so the episode keeps its last served action instead of
// failing this and every later turn, and the turn says so.
func TestMediaRefusalHoldsTheEpisodesServedAction(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	mediaRefusal(fixture, "undecodable_image")
	logs := captureLogs(t)
	result, err := fixture.selectOn(t, state, nextTurn(t))
	if err != nil {
		t.Fatalf("media refusal with a held action failed the turn: %v", err)
	}
	if result.RaylineARC.OfferKind != "media_hold" || len(result.RaylineARC.MaskedArms) != len(result.RaylineARC.ExcludedArms) {
		t.Fatalf("media hold offer_kind %q masked %v, want media_hold with a mask per arm",
			result.RaylineARC.OfferKind, result.RaylineARC.MaskedArms)
	}
	if result.RaylineARC.PolicyActionID != bindings[1].ActionID || !strings.Contains(result.Reasoning, "media_hold") {
		t.Fatalf("served %s (%s), want the held %s", result.RaylineARC.PolicyActionID, result.Reasoning, bindings[1].ActionID)
	}
	if next := result.RaylineARC.PolicyNextState; next == nil || next.Ledger[len(next.Ledger)-1].ActionID != bindings[1].ActionID {
		t.Fatalf("the held turn did not commit to the ledger: %+v", next)
	}
	if result.RaylineARC.RawScores != nil || !result.RaylineARC.TokenCountsUnknown {
		t.Fatalf("media hold scores %v, counts unknown %v: want no scores and unknown counts",
			result.RaylineARC.RawScores, result.RaylineARC.TokenCountsUnknown)
	}
	hold := findLogEvent(t, logs, "rayline_arc_media_hold")
	if hold["request_id"] != "req-policy-test" || hold["sub_reason"] != "undecodable_image" || hold["action_id"] != bindings[1].ActionID {
		t.Fatalf("media hold line = %v, want the request id, the bounded sub_reason and the held action", hold)
	}
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, result.RaylineARC)
	if selected := findLogEvent(t, logs, "rayline_arc_selection"); selected["offer_kind"] != "media_hold" {
		t.Fatalf("selection offer_kind = %v, want media_hold", selected["offer_kind"])
	}
}

// With nothing served yet a media refusal fails the turn as it does today:
// an unsupported_request the caller sees as request_shape.
func TestMediaRefusalWithNothingHeldIsARequestShapeFailure(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	mediaRefusal(fixture, "document_unsupported")
	logs := captureLogs(t)
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_service_unsupported_request" ||
		publicSelectionFailureClass(failure.class) != selectionFailureRequestShape || !selectionFailureIsCallerError(failure.class) {
		t.Fatalf("first call: error = %v, want policy_service_unsupported_request (request_shape)", err)
	}
	failed := findLogEvent(t, logs, "rayline_arc_policy_service_failed")
	if failed["class"] != "unsupported_request" || failed["reason"] != "media_refused" || failed["sub_reason"] != "document_unsupported" {
		t.Fatalf("failure line = %v, want unsupported_request with the bounded media sub_reason", failed)
	}
}

// Every other unsupported_request -- runtime_refused from a service that
// predates the typed reason among them -- still fails a held episode's turn.
func TestOtherUnsupportedRequestsStillFailAHeldEpisode(t *testing.T) {
	for _, detail := range []map[string]any{{"reason": "runtime_refused"}, {"reason": "projection_refused"}, {}} {
		fixture := newPolicySelectorFixture(t, "")
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		state, _ := raylinearc.NewEpisodeState(2)
		servedTurn(t, fixture, state, bindings[1].ActionID)
		fixture.fake.mu.Lock()
		fixture.fake.failWith, fixture.fake.failDetail = "unsupported_request", detail
		fixture.fake.mu.Unlock()
		_, err := fixture.selectOn(t, state, nextTurn(t))
		var failure *raylineARCSelectionFailure
		if !errors.As(err, &failure) || failure.class != "policy_service_unsupported_request" {
			t.Fatalf("detail %v: error = %v, want policy_service_unsupported_request", detail, err)
		}
	}
}
