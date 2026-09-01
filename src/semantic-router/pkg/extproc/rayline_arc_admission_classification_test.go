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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The admission gate exists to shed, and a shed is back-pressure from a
// healthy router. It must reach the caller as 429 with a Retry-After, never as
// the 503 that sends a caller into fallback and reads as an outage. This is
// the link between the gate's own failure class and that status: the adapter
// answers 429 for exactly the failures marked contended here. See
// pkg/apiserver/route_decision.go for the status mapping itself.
func TestAdmissionShedIsClassifiedAsContention(t *testing.T) {
	shed := &raylineARCSelectionFailure{
		class: arcEncoderFailureClass(raylinearc.EncoderFailureAdmission),
	}

	if !errors.Is(
		selectionFailureError(shed),
		routerruntime.ErrRouteDecisionContended,
	) {
		t.Fatalf("an admission shed lost its contention class")
	}
}

// Every real encoder failure keeps its 503. Waiting does not fix a transport
// error, a timeout or an unclassified failure, so inviting a retry would turn
// one broken consult into a retry storm.
func TestEncoderFailuresStayUnavailable(t *testing.T) {
	for _, class := range []raylinearc.EncoderFailureClass{
		raylinearc.EncoderFailureTransport,
		raylinearc.EncoderFailureTimeout,
		raylinearc.EncoderFailureRequest,
		raylinearc.EncoderFailureStatus,
		raylinearc.EncoderFailureDecode,
		raylinearc.EncoderFailureContract,
	} {
		failure := &raylineARCSelectionFailure{
			class: arcEncoderFailureClass(class),
		}
		if errors.Is(
			selectionFailureError(failure),
			routerruntime.ErrRouteDecisionContended,
		) {
			t.Fatalf("encoder failure %q was classified as contention", class)
		}
	}
}
