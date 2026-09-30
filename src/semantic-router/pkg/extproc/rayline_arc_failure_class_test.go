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
