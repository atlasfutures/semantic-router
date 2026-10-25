package execution

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// Opt-in real-runtime qualification. This test counts only; it never starts
// generation, downloads a model, or claims that the runtime's KV is resident.
func TestLlamaCPPLiveQualificationAndPrefix(t *testing.T) {
	path := os.Getenv("RAYLINE_LLAMA_DEPLOYMENT_JSON")
	if path == "" {
		t.Skip("set RAYLINE_LLAMA_DEPLOYMENT_JSON for the qualified external runtime")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var deployment Deployment
	if err := json.Unmarshal(raw, &deployment); err != nil {
		t.Fatal(err)
	}
	counter, err := newLlamaCPPCounter(t.Context(), deployment)
	if err != nil {
		t.Fatal(err)
	}
	tools := []any{map[string]any{"type": "function", "function": map[string]any{"name": "read_file", "description": "Read a file", "parameters": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]string{"type": "string"}}, "required": []string{"path"}}}}}
	messages := []any{map[string]any{"role": "system", "content": "Follow the task and report tool results accurately."}, map[string]any{"role": "user", "content": "Read example.txt. Until the next steering instruction, reason more thoroughly before acting."}}
	measure := func(messages []any) *cachecontext.MeasuredPrompt {
		t.Helper()
		body, err := json.Marshal(map[string]any{"model": deployment.WireModel, "tools": tools, "messages": messages})
		if err != nil {
			t.Fatal(err)
		}
		if deployment.InbandSystem == thinkingcontrol.InbandSystemFold {
			body, err = thinkingcontrol.FoldInbandSystem(body, thinkingcontrol.FormatChat)
			if err != nil {
				t.Fatal(err)
			}
		}
		measurement, err := cachecontext.MeasurePrompt(t.Context(), counter, cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "live-qualified-epoch", Body: body})
		if err != nil {
			t.Fatal(err)
		}
		return measurement
	}
	if deployment.InbandSystem == thinkingcontrol.InbandSystemFold {
		messages = append(messages, map[string]any{"role": "system", "content": []any{map[string]any{"type": "text", "text": "Environment for the current turn.", "cache_control": map[string]string{"type": "ephemeral"}}}})
	}
	previous := measure(messages)
	messages = append(messages,
		map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": "call-read", "type": "function", "function": map[string]string{"name": "read_file", "arguments": `{"path":"example.txt"}`}}}},
		map[string]any{"role": "tool", "tool_call_id": "call-read", "content": "the file contains a test sentence"},
		map[string]any{"role": "assistant", "content": "The file contains a test sentence."},
		map[string]any{"role": "user", "content": "Summarize that sentence."})
	current := measure(messages)
	prefix := current.Prefix(previous)
	if prefix == nil || prefix.Tokens <= 0 || prefix.OverlapTokens == nil || *prefix.OverlapTokens <= 0 || prefix.Provenance != "provider_reported" {
		t.Fatalf("qualified runtime did not provide an exact carried prefix with overlap: %+v", prefix)
	}
	t.Logf("actual runtime qualified: build=%s prefix_tokens=%d overlap_tokens=%d units=%s provenance=%s", deployment.PromptCounter.BuildInfo, prefix.Tokens, *prefix.OverlapTokens, prefix.Units, prefix.Provenance)
	changedMessages := append([]any(nil), messages...)
	changedMessages[0] = map[string]any{"role": "system", "content": "A changed system instruction invalidates the prior prefix."}
	changed := measure(changedMessages).Prefix(previous)
	if changed == nil || changed.OverlapTokens == nil || *changed.OverlapTokens >= *prefix.OverlapTokens {
		t.Fatal("changed system text did not reduce the measured common prefix")
	}
	t.Logf("actual changed-system overlap_tokens=%d (previous=%d); no cache-residency assertion", *changed.OverlapTokens, *prefix.OverlapTokens)
	// Optionally qualify a captured synthetic harness request with its complete
	// instructions and tool schemas. The artifact remains outside this repository.
	if requestPath := os.Getenv("RAYLINE_LLAMA_REQUEST_JSON"); requestPath != "" {
		body, err := os.ReadFile(requestPath)
		if err != nil {
			t.Fatal(err)
		}
		if deployment.InbandSystem == thinkingcontrol.InbandSystemFold {
			body, err = thinkingcontrol.FoldInbandSystem(body, thinkingcontrol.FormatChat)
			if err != nil {
				t.Fatal(err)
			}
		}
		measured, err := counter.Count(t.Context(), cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "captured-harness", Body: body})
		if err != nil {
			t.Fatal(err)
		}
		if measured.PromptTokens == 0 {
			t.Fatal("captured harness request had no counted tokens")
		}
		t.Logf("captured harness complete prompt qualified: %d tokens", measured.PromptTokens)
	}
}
