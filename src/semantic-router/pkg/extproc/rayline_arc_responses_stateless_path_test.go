package extproc

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// A Responses turn naming previous_response_id is refused with its own bounded
// class, not with the one the projection path produces when the router fails
// to materialize history it does hold. The two share a status today only by
// accident of the string, and they must not: one is the caller's to fix (400),
// the other is the router's own breakage (503).
func TestResponsesPreviousIDRefusalHasItsOwnClass(t *testing.T) {
	if arcFailureResponsesPreviousID == "turns_stored_history" {
		t.Fatalf("class %q collides with the projection's stored-history failure",
			arcFailureResponsesPreviousID)
	}
	if selectionFailureIsCallerError("turns_stored_history") {
		t.Fatal("a stored-history materialization failure is the router's, not the caller's")
	}
	if !selectionFailureIsCallerError(arcFailureResponsesPreviousID) {
		t.Fatalf("%s must be a caller error", arcFailureResponsesPreviousID)
	}
	if selectionFailureIsContended(arcFailureResponsesPreviousID) {
		t.Fatal("the refusal is permanent, so it must never read as back-pressure")
	}
}

// The refusal reaches the client as a 400 whose body names both the reason and
// the remedy, in the client's own wire format. The old answer was a 503, which
// reads as an outage and invites a retry that cannot succeed
// (memex-desktop#7484). Encoding is where the message is decided, so this goes
// through it.
func TestResponsesPreviousIDRefusalSurvivesClientEncoding(t *testing.T) {
	for name, format := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
		"responses": llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(name, func(t *testing.T) {
			router, requestContext, algorithm := missingSessionRequestContext(t, "episode-stateless-path")
			router.Config = &config.RouterConfig{}
			requestContext.SourceFormat = format
			router.buildRaylineARCSelectionContext(
				algorithm, requestContext, missingSessionModelRefs(), raylineARCEpisodeRequired,
			)

			err := selectionFailureForAlgorithm(algorithm, arcFailureResponsesPreviousID)
			response := router.authoritativeSelectionFailureResponse(err, requestContext)
			if response == nil {
				t.Fatal("no admission response")
			}
			if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureStatelessPath {
				t.Fatalf("%s = %q, want %q", selectionFailureHeader, got, selectionFailureStatelessPath)
			}
			encoded := router.encodeImmediateResponseForClient(response, requestContext)
			immediate := encoded.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("response is not an immediate refusal: %+v", encoded)
			}
			if got := int(immediate.GetStatus().GetCode()); got != 400 {
				t.Fatalf("status = %d, want 400", got)
			}
			message, errorType := immediateErrorBody(t, immediate.GetBody())
			if message != statelessPathMessage {
				t.Fatalf("message = %q, want %q", message, statelessPathMessage)
			}
			if errorType != "invalid_request_error" {
				t.Fatalf("error type = %q, want invalid_request_error", errorType)
			}
		})
	}
}

// The message has to carry the remedy, not just the refusal: a client that is
// only told "no" cannot act, and replaying the items is the one thing that
// works. It must also name what the caller sent and no private component.
func TestStatelessPathMessageNamesTheRemedy(t *testing.T) {
	if !strings.Contains(statelessPathMessage, "previous_response_id") {
		t.Fatalf("message does not name the member the caller sent: %q", statelessPathMessage)
	}
	if !strings.Contains(statelessPathMessage, "input") {
		t.Fatalf("message does not say to resend the items: %q", statelessPathMessage)
	}
	for _, private := range []string{"ARC", "policy", "pathfinder", "cell", "encoder"} {
		if strings.Contains(statelessPathMessage, private) {
			t.Fatalf("message names the private component %q: %q", private, statelessPathMessage)
		}
	}
}

// Both entrypoints must answer this identically. The decision API reads
// sentinel errors rather than class strings, so without one attached the same
// turn would be a 400 through ExtProc and a 503 through POST /v1/routes --
// which is the inconsistency the ticket is about.
func TestResponsesPreviousIDRefusalIsInvalidBodyOnTheDecisionAPI(t *testing.T) {
	err := prepareFailureError(arcFailureResponsesPreviousID)
	if !errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody) {
		t.Fatalf("error = %v, want it to carry ErrRouteDecisionInvalidBody", err)
	}
	if !errors.Is(err, errRaylineRoutesStatelessPath) {
		t.Fatalf("error = %v, want it to carry errRaylineRoutesStatelessPath", err)
	}
	// Not the body-shape cause: that one answers a different message, and a
	// caller whose body parsed fine must not be told it did not.
	if errors.Is(err, errRaylineRoutesRequestShape) {
		t.Fatal("the refusal must not claim the request body could not be represented")
	}
	if errors.Is(err, routerruntime.ErrRouteDecisionContended) {
		t.Fatal("the refusal is permanent, so it must never read as contention")
	}
}

// The sentinel is a tag for errors.Is, and the const is what the caller reads.
// Nothing should make the sentinel's own text client-visible, so it must not be
// a second copy of the message: two copies drift, and whoever edits one would
// have no way to know the other exists.
func TestStatelessPathSentinelIsNotASecondCopyOfTheMessage(t *testing.T) {
	if errRaylineRoutesStatelessPath.Error() == statelessPathMessage {
		t.Fatal("the sentinel duplicates the client-visible message; it is a tag, not a message")
	}
	// The remedy is the half a caller acts on, and it belongs to the message
	// alone. In the sentinel it would read as text meant to be shown.
	if strings.Contains(errRaylineRoutesStatelessPath.Error(), "Resend") {
		t.Fatal("the sentinel carries the remedy, which only the message should")
	}
}

// A genuine failure to materialize stored history keeps answering 503. It is
// the router failing at something it undertook, so it must not be reclassified
// as the caller's fault by the change above.
func TestStoredHistoryMaterializationStaysUnavailable(t *testing.T) {
	err := prepareFailureError("turns_stored_history")
	if errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody) {
		t.Fatalf("error = %v, want a plain failure that reads as 503", err)
	}
	if got := publicSelectionFailureClass("turns_stored_history"); got != selectionFailureUnavailable {
		t.Fatalf("public class = %q, want %q", got, selectionFailureUnavailable)
	}
}

// The public vocabulary is a contract the header, the docs and the callers
// share, so the new value must be distinct from every existing one.
func TestStatelessPathClassIsDistinct(t *testing.T) {
	for _, existing := range []string{
		selectionFailureSessionBusy,
		selectionFailureCapacity,
		selectionFailureNotReady,
		selectionFailureMissingSession,
		selectionFailureRequestShape,
		selectionFailureUndecidable,
		selectionFailureUnavailable,
	} {
		if selectionFailureStatelessPath == existing {
			t.Fatalf("stateless_path collides with the existing class %q", existing)
		}
	}
	if got := publicSelectionFailureClass(arcFailureResponsesPreviousID); got != selectionFailureStatelessPath {
		t.Fatalf("public class = %q, want %q", got, selectionFailureStatelessPath)
	}
}
