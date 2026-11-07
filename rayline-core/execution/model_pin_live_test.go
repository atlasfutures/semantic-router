package execution

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
)

// Opt-in check that the real serving gateway preserves the two shapes
// validateModelPinBody used to refuse. It answers the question a httptest
// upstream cannot: whether the shape survives the gateway's own Messages
// adapter, not merely whether it reaches one.
//
// Set RAYLINE_MODEL_PIN_LIVE_URL to the gateway's Messages endpoint and
// RAYLINE_MODEL_PIN_LIVE_KEY to a credential for it;
// RAYLINE_MODEL_PIN_LIVE_MODEL names the concrete pinned model. It generates a
// few tokens per case.
func TestModelPinLiveGatewayPreservesRefusedShapes(t *testing.T) {
	url, key := os.Getenv("RAYLINE_MODEL_PIN_LIVE_URL"), os.Getenv("RAYLINE_MODEL_PIN_LIVE_KEY")
	model := os.Getenv("RAYLINE_MODEL_PIN_LIVE_MODEL")
	if url == "" || key == "" || model == "" {
		t.Skip("set RAYLINE_MODEL_PIN_LIVE_URL, RAYLINE_MODEL_PIN_LIVE_KEY and RAYLINE_MODEL_PIN_LIVE_MODEL")
	}
	ask := func(t *testing.T, body map[string]any) string {
		t.Helper()
		body["model"] = model
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		request, err := http.NewRequest(http.MethodPost, url, strings.NewReader(string(raw)))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+key)
		request.Header.Set("Content-Type", "application/json")
		// The transport's own pin header, as upstream() sets it.
		request.Header.Set("X-Workshop-Required-Model", model)
		response, err := http.DefaultClient.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		var answer struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.NewDecoder(response.Body).Decode(&answer); err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK || answer.Error != nil {
			message := ""
			if answer.Error != nil {
				message = answer.Error.Message
			}
			t.Fatalf("gateway refused the shape: HTTP %d %s", response.StatusCode, message)
		}
		var text strings.Builder
		for _, block := range answer.Content {
			if block.Type == "text" {
				text.WriteString(block.Text)
			}
		}
		return text.String()
	}

	t.Run("inBandSystemRowKeepsItsRole", func(t *testing.T) {
		// A folded row would arrive as user content, so the model would not
		// report a system role carrying the sentinel.
		answer := ask(t, map[string]any{
			"max_tokens": 200,
			"messages": []any{
				map[string]any{"role": "user", "content": "Hello."},
				map[string]any{"role": "system", "content": "SENTINEL_ABC"},
				map[string]any{"role": "user", "content": `Which role carried the text SENTINEL_ABC? Answer with one word: system, user, or absent.`},
			},
		})
		if !strings.Contains(strings.ToLower(answer), "system") {
			t.Fatalf("in-band system row did not survive the gateway adapter: %q", answer)
		}
	})

	t.Run("literalXhighInProseIsNotCoerced", func(t *testing.T) {
		answer := ask(t, map[string]any{
			"max_tokens": 120,
			"messages": []any{
				map[string]any{"role": "user", "content": "Repeat the following token exactly once, with no other words: xhigh"},
			},
		})
		if !strings.Contains(answer, "xhigh") {
			t.Fatalf("literal xhigh did not survive the gateway adapter: %q", answer)
		}
	})
}
