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
	"context"
	"errors"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Every selection failure tells the caller why, in the public vocabulary, so
// an eval harness can wait for its own in-flight turn on session_busy, back off
// on capacity and warm up on not_ready instead of treating one 429 or 503 as
// all of them.
func TestSelectionFailureCarriesPublicClass(t *testing.T) {
	router := &OpenAIRouter{}
	tests := []struct {
		class      string
		wantStatus int
		wantClass  string
	}{
		{"episode_timeout", 429, selectionFailureSessionBusy},
		{"policy_service_session_busy", 429, selectionFailureSessionBusy},
		{"episode_capacity", 429, selectionFailureCapacity},
		{arcEncoderFailureClassAdmission, 429, selectionFailureCapacity},
		{"policy_service_session_capacity", 429, selectionFailureCapacity},
		{"not_ready", 503, selectionFailureNotReady},
		{arcFailureMissingEpisodeID, 400, selectionFailureMissingSession},
		// A private turn with no allowed arm is refused identically every
		// time, so it is never a replayable 503.
		{arcFailureNoPrivateArm, 422, selectionFailureNoPrivateArm},
		// A decision service that cannot represent what the request carries
		// refused the request, not the turn: no arm was dialled, so the caller
		// is told 400 and never that a model service was unavailable.
		{"policy_service_unsupported_request", 400, selectionFailureRequestShape},
		// But not invalid_request: the router builds most of the decision
		// envelope, so that code can name a router-side integration fault.
		// Reporting it as the caller's 400 would blame it for a fault it
		// cannot fix and would hide a router bug.
		{"policy_service_invalid_request", 503, selectionFailureUnavailable},
		{arcEncoderFailureClass(raylinearc.EncoderFailureTransport), 503, selectionFailureUnavailable},
		{"policy_service_auth", 503, selectionFailureUnavailable},
	}
	for _, test := range tests {
		t.Run(test.class, func(t *testing.T) {
			response := router.authoritativeSelectionFailureResponse(
				&modelSelectionFailure{algorithm: configRaylineARC, class: test.class},
				&RequestContext{},
			)
			immediate := response.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("response = %#v", response)
			}
			if got := int(immediate.GetStatus().GetCode()); got != test.wantStatus {
				t.Fatalf("status = %d, want %d", got, test.wantStatus)
			}
			if got := immediateHeaderValue(response, selectionFailureHeader); got != test.wantClass {
				t.Fatalf("%s = %q, want %q", selectionFailureHeader, got, test.wantClass)
			}
		})
	}
}

// A contended class must never be published as a class that tells the caller
// waiting will not help, or the header would contradict the 429 it rides on.
func TestContendedClassesMapToRetryableVocabulary(t *testing.T) {
	for _, class := range []string{
		"episode_timeout",
		"episode_capacity",
		arcEncoderFailureClassAdmission,
		"policy_service_session_busy",
		"policy_service_session_capacity",
	} {
		if !selectionFailureIsContended(class) {
			t.Fatalf("%q is no longer contended; update this test and the vocabulary together", class)
		}
		switch got := publicSelectionFailureClass(class); got {
		case selectionFailureSessionBusy, selectionFailureCapacity:
		default:
			t.Fatalf("contended class %q published as %q", class, got)
		}
	}
}

// A request a decision service cannot represent is the caller's to fix and
// nobody's outage. The arm is never dialled, so reporting it as a model
// service failure blames a component that was not involved, invites a retry
// that is refused identically every time, and -- because monitoring reads a
// 503 as provider downtime -- reports an outage that did not happen
// (memex-desktop#7328).
func TestRequestShapeRefusalIsACallerErrorAndNeverUnavailable(t *testing.T) {
	router := &OpenAIRouter{}
	for class := range selectionShapeRefusalClasses {
		t.Run(class, func(t *testing.T) {
			if !selectionFailureIsCallerError(class) {
				t.Fatalf("%q is not a caller error", class)
			}
			if selectionFailureIsContended(class) {
				t.Fatalf("%q is contended: resending it unchanged cannot succeed", class)
			}
			if got := publicSelectionFailureClass(class); got != selectionFailureRequestShape {
				t.Fatalf("published %q, want %q", got, selectionFailureRequestShape)
			}
			// The turn is classed for the logs as the request-shape refusal it
			// is, not as one of the "unavailable" refinements.
			if got := selectionTurnFailureClass(class); got == selectionFailureUnavailable {
				t.Fatalf("turn class = %q, want anything but %q", got, selectionFailureUnavailable)
			}
			// It is not the backend's error, so it must not count against the
			// load-balancing window for an arm that was never called.
			if turnFailureIsBackendError(selectionTurnFailureClass(class)) {
				t.Fatalf("%q counted as a backend error", class)
			}
			ctx := &RequestContext{}
			response := router.authoritativeSelectionFailureResponse(
				&modelSelectionFailure{algorithm: configRaylineARC, class: class}, ctx,
			)
			if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != 400 {
				t.Fatalf("status = %d, want 400", got)
			}
			// The body is re-encoded from the carried protocol error, so the
			// client is told its request was invalid rather than a canned
			// message for the status.
			if ctx.ImmediateProtocolError == nil {
				t.Fatal("no protocol error carried: the refusal would read as a canned message")
			}
			if got := ctx.ImmediateProtocolError.Category; got != llmprotocol.ErrorInvalidRequest {
				t.Fatalf("category = %q, want %q", got, llmprotocol.ErrorInvalidRequest)
			}
			if got := ctx.ImmediateProtocolError.Code; got != selectionFailureRequestShape {
				t.Fatalf("code = %q, want %q", got, selectionFailureRequestShape)
			}
		})
	}
}

// A genuine arm outage still reads as one: the shape refusal must not widen
// into the classes that do mean the router or a dependency is unwell.
func TestGenuineSelectionBreakageStillAnswersUnavailable(t *testing.T) {
	router := &OpenAIRouter{}
	for _, class := range []string{
		"policy_service_auth",
		"policy_service_failed",
		arcEncoderFailureClass(raylinearc.EncoderFailureTransport),
	} {
		t.Run(class, func(t *testing.T) {
			if selectionFailureIsCallerError(class) {
				t.Fatalf("%q became a caller error", class)
			}
			response := router.authoritativeSelectionFailureResponse(
				&modelSelectionFailure{algorithm: configRaylineARC, class: class}, &RequestContext{},
			)
			if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != 503 {
				t.Fatalf("status = %d, want 503", got)
			}
			if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureUnavailable {
				t.Fatalf("%s = %q, want %q", selectionFailureHeader, got, selectionFailureUnavailable)
			}
		})
	}
}

// The published value is the public vocabulary, never the internal class,
// which names private components.
func TestPublicClassNeverLeaksInternalClass(t *testing.T) {
	for _, class := range []string{
		"policy_service_session_busy",
		"policy_service_auth",
		arcEncoderFailureClass(raylinearc.EncoderFailureTimeout),
	} {
		if got := publicSelectionFailureClass(class); got == class {
			t.Fatalf("internal class %q published verbatim", class)
		}
	}
}

// A lease lost at the dispatch gate is not known to be contention: renewal
// also fails on transport errors and timeouts. Publishing session_busy there
// would send the caller waiting for an in-flight turn that may not exist.
func TestDispatchGateLeaseLossIsUnavailable(t *testing.T) {
	ctx := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			&recordingSelectionTransaction{validateErr: errors.New("renewal failed")},
		),
	}
	response := (&OpenAIRouter{}).selectionDispatchGateResponse(ctx)
	if response.GetImmediateResponse() == nil {
		t.Fatalf("response = %#v", response)
	}
	if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureUnavailable {
		t.Fatalf("%s = %q, want %q", selectionFailureHeader, got, selectionFailureUnavailable)
	}
}

// A prepare that ran out of time is contention only if the store saw another
// owner's lease. A timeout inside a stalled store call is not, and must not
// tell the caller to wait for an in-flight turn that does not exist.
func TestPrepareTimeoutIsContentionOnlyWhenTheLeaseWasHeld(t *testing.T) {
	held := errors.Join(raylinearc.ErrEpisodeLeaseHeld, context.DeadlineExceeded)
	if got := boundedARCPrepareFailure(held); got != "episode_timeout" {
		t.Fatalf("held timeout class = %q", got)
	}
	if got := publicSelectionFailureClass(boundedARCPrepareFailure(held)); got != selectionFailureSessionBusy {
		t.Fatalf("held timeout published as %q", got)
	}
	stalled := boundedARCPrepareFailure(context.DeadlineExceeded)
	if stalled != "episode_store_timeout" {
		t.Fatalf("stalled timeout class = %q", stalled)
	}
	if selectionFailureIsContended(stalled) {
		t.Fatalf("a stalled store must not answer 429")
	}
	if got := publicSelectionFailureClass(stalled); got != selectionFailureUnavailable {
		t.Fatalf("stalled timeout published as %q", got)
	}
}

// A selector still being probed may arm later, so its refusal is a warm-up. One
// that construction left unarmed for good never will, and must not tell the
// caller to wait for a warm-up that cannot complete.
func TestUnarmedSelectorDistinguishesWarmUpFromPermanentFailure(t *testing.T) {
	probing := newRaylineARCSelector(nil, nil, nil, "rev")
	_, _, _, err := probing.prepareSelection(nil, nil)
	if got := publicSelectionFailureClass(selectionFailureClassOf(t, err)); got != selectionFailureNotReady {
		t.Fatalf("probing selector published %q", got)
	}
	failed := newUnrecoverableRaylineARCSelector("rev")
	_, _, _, err = failed.prepareSelection(nil, nil)
	class := selectionFailureClassOf(t, err)
	if selectionFailureIsContended(class) {
		t.Fatalf("an unrecoverable selector must not answer 429")
	}
	if got := publicSelectionFailureClass(class); got != selectionFailureUnavailable {
		t.Fatalf("unrecoverable selector published %q", got)
	}
}

func selectionFailureClassOf(t *testing.T, err error) string {
	t.Helper()
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) {
		t.Fatalf("error %v is not a selection failure", err)
	}
	return failure.class
}

// A turn the policy would not decide is answered as undecidable on every
// entrypoint: a 422 with its own public class, a turn class of its own that
// counts against no backend, and the decision-only surfaces' undecidable
// error -- never a served default and never a 503 that invites a retry.
func TestUndecidableIsA422AndNeverUnavailable(t *testing.T) {
	class := arcFailurePolicyUndecidable
	if selectionFailureIsContended(class) || selectionFailureIsCallerError(class) {
		t.Fatalf("%q is classed as contention or a missing session", class)
	}
	if got := publicSelectionFailureClass(class); got != selectionFailureUndecidable {
		t.Fatalf("published %q, want %q", got, selectionFailureUndecidable)
	}
	if got := selectionTurnFailureClass(class); got != turnFailureUndecidable || turnFailureIsBackendError(got) {
		t.Fatalf("turn class = %q, want %q and not a backend error", got, turnFailureUndecidable)
	}
	ctx := &RequestContext{}
	response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: configRaylineARC, class: class}, ctx,
	)
	if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != 422 {
		t.Fatalf("status = %d, want 422", got)
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != selectionFailureUndecidable {
		t.Fatalf("protocol error = %+v, want code %q", ctx.ImmediateProtocolError, selectionFailureUndecidable)
	}
	if err := selectionFailureError(&raylineARCSelectionFailure{class: class}); !errors.Is(err, routerruntime.ErrRouteDecisionUndecidable) {
		t.Fatalf("the decision-only error %v is not undecidable", err)
	}
}
