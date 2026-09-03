package extproc

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const testEpisodeIDHeader = "x-rayline-session"

// missingSessionModelRefs stands in for the decision's arm list. The episode
// header is read before any arm is looked at, so the names carry no meaning
// here beyond being a well-formed candidate set.
func missingSessionModelRefs() []config.ModelRef {
	refs := make([]config.ModelRef, 0, 7)
	for index := 0; index < 7; index++ {
		refs = append(refs, config.ModelRef{Model: "arm-" + strconv.Itoa(index)})
	}
	return refs
}

// raylineARCAlgorithmConfigForTest is the fail-closed ARC algorithm the
// bounded-class assertions need, without an episode store behind it.
func raylineARCAlgorithmConfigForTest() *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type:       config.RaylineARCAlgorithmType,
		OnError:    "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{},
	}
}

func missingSessionRequestContext(t *testing.T, header string) (*OpenAIRouter, *RequestContext, *config.AlgorithmConfig) {
	t.Helper()
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 4, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	headers := map[string]string{}
	if header != "" {
		headers[testEpisodeIDHeader] = header
	}
	requestContext := &RequestContext{
		Headers:      headers,
		SourceFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "public test turn"}},
			}},
		},
		TraceContext: context.Background(),
	}
	algorithm := &config.AlgorithmConfig{
		Type:    config.RaylineARCAlgorithmType,
		OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{
				IDHeader:              testEpisodeIDHeader,
				CloseHeader:           "x-rayline-episode-close",
				AcquireTimeoutSeconds: 1,
			},
		},
	}
	return &OpenAIRouter{RaylineARCEpisodeStore: store}, requestContext, algorithm
}

// An absent header and a header that holds only whitespace are the same
// omission, and both must be recognised before any episode work begins.
func TestRaylineARCMissingSessionHeaderIsPreparationFailure(t *testing.T) {
	for name, header := range map[string]string{
		"absent":     "",
		"whitespace": "   ",
	} {
		t.Run(name, func(t *testing.T) {
			router, requestContext, algorithm := missingSessionRequestContext(t, header)
			arcContext := router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())
			if arcContext == nil {
				t.Fatal("no ARC selection context was built")
			}
			if arcContext.PreparationFailure != arcFailureMissingEpisodeID {
				t.Fatalf("preparation failure = %q, want %q", arcContext.PreparationFailure, arcFailureMissingEpisodeID)
			}
			if requestContext.RaylineARCEpisodeIDHeader != testEpisodeIDHeader {
				t.Fatalf("header name = %q, want the configured header so the refusal can name it",
					requestContext.RaylineARCEpisodeIDHeader)
			}
		})
	}
}

// The caller omitted a header. Answering 503 tells them to retry something
// that can never succeed; a 400 naming the header tells them what to send.
func TestRaylineARCMissingSessionHeaderAnswers400(t *testing.T) {
	router, requestContext, algorithm := missingSessionRequestContext(t, "")
	router.buildRaylineARCSelectionContext(algorithm, requestContext, missingSessionModelRefs())
	before := testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues(arcFailureMissingEpisodeID))

	err := selectionFailureForAlgorithm(algorithm, arcFailureMissingEpisodeID)
	response := router.authoritativeSelectionFailureResponse(err, requestContext)
	if response == nil {
		t.Fatal("a missing session header produced no admission response")
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("response is not an immediate refusal: %+v", response)
	}
	if got := int(immediate.GetStatus().GetCode()); got != 400 {
		t.Fatalf("status = %d, want 400", got)
	}
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    int    `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.GetBody(), &body); err != nil {
		t.Fatalf("refusal body is not JSON: %s", immediate.GetBody())
	}
	if body.Error.Type != "invalid_request_error" {
		t.Fatalf("error type = %q, want invalid_request_error", body.Error.Type)
	}
	if !bytesContainsHeaderName(body.Error.Message) {
		t.Fatalf("message %q does not name %s", body.Error.Message, testEpisodeIDHeader)
	}
	after := testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues(arcFailureMissingEpisodeID))
	if after != before+1 {
		t.Fatalf("failure counter moved %v to %v, want one increment", before, after)
	}
}

func bytesContainsHeaderName(message string) bool {
	for index := 0; index+len(testEpisodeIDHeader) <= len(message); index++ {
		if message[index:index+len(testEpisodeIDHeader)] == testEpisodeIDHeader {
			return true
		}
	}
	return false
}
