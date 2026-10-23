package extproc

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A policy-service cell does not run the Go turn projection: the service
// projects and encodes the raw request itself, so a turn only the Go
// projection would refuse is prepared for the service. The image fact the
// vision mask reads is still taken off the conversation. An artifact-mode
// cell, whose encoder reads the projected turns, still refuses it.
func TestPolicyServiceTurnSkipsTheGoProjection(t *testing.T) {
	for _, policy := range []bool{true, false} {
		router, reqCtx, algorithm := missingSessionRequestContext(t, "episode-no-go-projection")
		reqCtx.RaylineARCRawBody = []byte(`{"model":"auto","messages":[]}`)
		reqCtx.SemanticRequest.Messages[0].Content = append(
			reqCtx.SemanticRequest.Messages[0].Content,
			llmprotocol.Content{Kind: llmprotocol.ContentKind("future_kind_from_a_later_release")},
			llmprotocol.Content{Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: "iVBORw0KGgo="},
		)
		if policy {
			algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "http://policy.invalid"}
		}
		arc := router.buildRaylineARCSelectionContext(algorithm, reqCtx, missingSessionModelRefs(), raylineARCEpisodeRequired)
		if !policy {
			if arc.PreparationFailure != "turns_unknown_item" {
				t.Fatalf("artifact mode: failure = %q, want turns_unknown_item", arc.PreparationFailure)
			}
			continue
		}
		if arc.PreparationFailure != "" {
			t.Fatalf("policy mode: failure = %q, want the turn prepared for the service", arc.PreparationFailure)
		}
		if arc.Turns != nil {
			t.Fatalf("policy mode projected turns: %+v", arc.Turns)
		}
		if !arc.ImageBearing {
			t.Fatal("policy mode lost the image fact the vision mask reads")
		}
		if string(arc.RawRequest) != string(reqCtx.RaylineARCRawBody) {
			t.Fatalf("policy mode raw request = %s", arc.RawRequest)
		}
	}
}
