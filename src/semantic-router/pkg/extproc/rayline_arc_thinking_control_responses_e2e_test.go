//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// v5OpusOnOpenAIResponses moves the v5 config's Opus worker to a Responses
// backend on a direct OpenAI-type provider, whose issuer the episode can
// track (an OpenRouter Responses target never forwards encrypted reasoning).
func v5OpusOnOpenAIResponses(t *testing.T) func(string) string {
	return func(rendered string) string {
		edited := strings.NewReplacer(
			"      provider_model_id: anthropic/claude-opus-5\n      api_format: anthropic\n",
			"      provider_model_id: anthropic/claude-opus-5\n      api_format: responses\n",
			"        - name: openrouter-opus\n          base_url: https://openrouter.ai/api/v1\n          provider: openrouter\n",
			"        - name: openai-opus\n          base_url: https://api.openai.com/v1\n          provider: openai\n",
		).Replace(rendered)
		if strings.Count(edited, "api_format: responses") != 1 || !strings.Contains(edited, "name: openai-opus") {
			t.Fatal("the v5 config template changed; update the Opus Responses edit")
		}
		return edited
	}
}

// v5CodexTurn routes one recorded Codex Responses turn to action and commits
// it as a 200.
func v5CodexTurn(t *testing.T, router *OpenAIRouter, fake *fakePolicyService, action, episode, fixture string) (string, *RequestContext) {
	t.Helper()
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
	body, ctx := codexTurn(t, router, episode, fixture)
	finishCodexTurn(t, router, ctx, "200")
	return body, ctx
}

// A v5 action is served on a Responses worker where the registry admits its
// control. Every Responses cell admits only the native-default control, whose
// base wire is empty: a Codex turn reaches the worker without the client's
// own reasoning controls, under the served model id, and the episode still
// forwards encrypted reasoning back to the worker that issued it.
func TestV5NativeControlServesAResponsesWorker(t *testing.T) {
	router, fake := v5RouterWith(t, "openai", v5OpusOnOpenAIResponses(t))
	const episode = "v5-responses"

	body, ctx := v5CodexTurn(t, router, fake, v5OpusAction, episode, "turn1-request.json")
	var provider map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &provider); err != nil {
		t.Fatal(err)
	}
	if string(provider["model"]) != `"anthropic/claude-opus-5"` {
		t.Fatalf("model = %s", provider["model"])
	}
	if _, present := provider["input"]; !present {
		t.Fatalf("the body is not a Responses body: %s", body)
	}
	if reasoning, present := provider["reasoning"]; present {
		t.Fatalf("the client's reasoning controls reached the provider: %s", reasoning)
	}
	trace := ctx.RaylineARCThinking
	if trace == nil || trace.Lever != thinkingControlLever || !strings.HasPrefix(trace.ControlInForce, "628d2855") ||
		trace.LevelInForce != "" || trace.Written != "" {
		t.Fatalf("trace = %+v", trace)
	}

	body, _ = v5CodexTurn(t, router, fake, v5OpusAction, episode, "turn2-request.json")
	if !strings.Contains(body, codexBlob) || !strings.Contains(body, `"type":"reasoning"`) {
		t.Fatalf("the blob did not reach the worker that issued it: %s", body)
	}
	if strings.Contains(body, `"reasoning":{`) {
		t.Fatalf("the client's reasoning controls reached the provider: %s", body)
	}
}
