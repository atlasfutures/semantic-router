package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A model whose second accepted format carries a capability its first lacks
// stays eligible at selection: audio input is carried by Chat, not Messages.
// With only its first format the same demand excludes it.
func TestDemandOnlySelectionAdmitsAnyAcceptedFormat(t *testing.T) {
	demand := selection.CandidateDemand{
		Known:           true,
		MaxOutputTokens: llmprotocol.Int64(512),
		Capabilities:    llmprotocol.Capabilities(llmprotocol.CapabilityText, llmprotocol.CapabilityAudioInput),
	}
	for name, tc := range map[string]struct {
		accepted []string
		eligible bool
	}{
		"second format carries it": {[]string{config.APIFormatAnthropic, config.APIFormatOpenAI}, true},
		"first format only":        {nil, false},
	} {
		t.Run(name, func(t *testing.T) {
			r := strictCandidateRouter()
			r.Config.ModelConfig["dual"] = config.ModelParams{
				APIFormat: config.APIFormatAnthropic, AcceptedFormats: tc.accepted,
				Capabilities: []string{"chat"}, ContextWindowSize: 20000, MaxOutputTokens: 1024,
			}
			refs, err := r.eligibleDemandModelRefs(r.Config.CandidateRequirements, []config.ModelRef{{Model: "dual"}}, demand)
			if got := err == nil && len(refs) == 1; got != tc.eligible {
				t.Fatalf("eligible = %v (refs=%v err=%v), want %v", got, refs, err, tc.eligible)
			}
		})
	}
}
