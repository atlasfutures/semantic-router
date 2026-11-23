package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A worker moved to Vertex AI under the same name and format does not resume
// the placer it had on another endpoint: that placer's ledger can hold
// instructions the Vertex AI cell refuses.
func TestRaylineARCPlacementKeySeparatesVertexAI(t *testing.T) {
	openRouter := &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}
	vertex := &config.ProviderProfile{Type: "vertex-ai", BaseURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global"}
	if got := raylineARCPlacementKey("w", "chat", openRouter); got != "w|chat" {
		t.Fatalf("an OpenRouter dispatch's key = %q, want the plain worker|format key", got)
	}
	if got := raylineARCPlacementKey("w", "chat", nil); got != "w|chat" {
		t.Fatalf("a dispatch with no profile's key = %q", got)
	}
	if got := raylineARCPlacementKey("w", "chat", vertex); got == "w|chat" {
		t.Fatal("a Vertex AI dispatch resumes the worker's other placer")
	}
}
