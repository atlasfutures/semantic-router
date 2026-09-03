package extproc

import (
	"encoding/json"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func immediateErrorBody(t *testing.T, body []byte) (message string, errorType string) {
	t.Helper()
	var decoded struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("refusal body is not JSON: %s", body)
	}
	return decoded.Error.Message, decoded.Error.Type
}

// The refusal is re-encoded into the client's own format before it leaves, and
// that step replaces the message with a canned one unless the request carries
// the protocol error it should use. A test that stops at the admission
// contract cannot see that, which is why this one goes through the encode.
func TestMissingSessionHeaderMessageSurvivesClientEncoding(t *testing.T) {
	for name, format := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) {
			router, requestContext, algorithm := missingSessionRequestContext(t, "")
			router.Config = &config.RouterConfig{}
			requestContext.SourceFormat = format
			router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())

			err := selectionFailureForAlgorithm(algorithm, arcFailureMissingEpisodeID)
			response := router.authoritativeSelectionFailureResponse(err, requestContext)
			if response == nil {
				t.Fatal("no admission response")
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
			want := "This request needs the " + testEpisodeIDHeader + " header."
			if message != want {
				t.Fatalf("message = %q, want %q", message, want)
			}
			if errorType != "invalid_request_error" {
				t.Fatalf("error type = %q, want invalid_request_error", errorType)
			}
		})
	}
}

// One refused request is one failure. The bounded class is counted where it is
// constructed, so counting it again where it is answered reports two.
func TestMissingSessionHeaderCountsOnceForOneRequest(t *testing.T) {
	router, requestContext, algorithm := missingSessionRequestContext(t, "")
	router.Config = &config.RouterConfig{}
	router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())
	before := testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues(arcFailureMissingEpisodeID))

	// The selector constructs the bounded failure, and the request path then
	// maps it onto the admission contract. Both happen once per request.
	selectorErr := arcSelectionFailure(arcFailureMissingEpisodeID)
	class := authoritativeSelectionFailureClass(algorithm, selectorErr)
	router.authoritativeSelectionFailureResponse(
		selectionFailureForAlgorithm(algorithm, class), requestContext,
	)

	after := testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues(arcFailureMissingEpisodeID))
	if after != before+1 {
		t.Fatalf("counter moved %v to %v, want exactly one increment per refused request", before, after)
	}
}

// The failure event is only keyable if it names the bounded class. A model
// selection failure already carries one, so falling back to the generic
// transaction class throws away the answer.
func TestSelectionFailureClassNamesTheBoundedClass(t *testing.T) {
	err := selectionFailureForAlgorithm(raylineARCAlgorithmConfigForTest(), arcFailureMissingEpisodeID)
	if got := boundedSelectionTransactionFailure(err); got != arcFailureMissingEpisodeID {
		t.Fatalf("failure class = %q, want %q", got, arcFailureMissingEpisodeID)
	}
}
