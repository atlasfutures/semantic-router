package execution

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
)

// Opt-in real-runtime capacity check against a caller-owned qualified llama.cpp
// (the same deployment JSON as TestLlamaCPPLiveQualificationAndPrefix). It uses
// the runtime's real tokenizer and its real context refusal; Decisions and the
// cloud target are local fixtures. It generates at most a few tokens locally.
func TestLlamaCPPLiveCapacityFactsAndRefusal(t *testing.T) {
	path := os.Getenv("RAYLINE_LLAMA_CAPACITY_LIVE_JSON")
	if path == "" {
		t.Skip("set RAYLINE_LLAMA_CAPACITY_LIVE_JSON for the qualified external runtime")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var local Deployment
	if err := json.Unmarshal(raw, &local); err != nil {
		t.Fatal(err)
	}
	if local.AuthEnv != "" {
		t.Setenv(local.AuthEnv, os.Getenv(local.AuthEnv))
	}
	cloud := &upstreamFixture{}
	cloudURL := chatUpstream(t, cloud, true).URL + "/v1/chat/completions"
	choose := func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}
	run := func(t *testing.T, configured int64, words int) ([]v2Lookup, *upstreamFixture) {
		t.Helper()
		config := hybridV2Config(t, local.URL, cloudURL)
		deployment := local
		deployment.ID = config.Deployments[0].ID
		deployment.ContextWindow = configured
		config.Deployments[0] = deployment
		fixture := &v2Decisions{choose: choose}
		server, httpServer := v2Server(t, config, fixture)
		if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
			t.Fatal(err)
		}
		prompt := "Reply with the single word ready."
		if words > 0 {
			prompt = strings.Repeat("lorem ", words) + prompt
		}
		body, _ := json.Marshal(map[string]any{"model": "arc-1.0", "max_tokens": 16, "messages": []any{map[string]any{"role": "user", "content": prompt}}})
		before := cloud.count()
		status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", string(body), t.Name())
		if status != http.StatusOK {
			t.Fatalf("status %d: %s", status, response)
		}
		lookups := fixture.snapshot()
		for _, lookup := range lookups {
			capacity := lookup.Execution.Targets[0].Capacity
			t.Logf("attempt=%d reconsult=%v local capacity=%s", lookup.Execution.Turn.Attempt, stringOrNil(lookup.Execution.Turn.Reconsult), mustJSON(capacity))
		}
		t.Logf("cloud dispatches=%d", cloud.count()-before)
		return lookups, cloud
	}
	t.Run("fits", func(t *testing.T) {
		lookups, _ := run(t, local.ContextWindow, 0)
		capacity := lookups[0].Execution.Targets[0].Capacity
		if len(lookups) != 1 || capacity == nil || !capacity.Fits || *capacity.Required <= 16 || *capacity.Units != local.TokenUnits {
			t.Fatalf("small prompt capacity %s", mustJSON(capacity))
		}
	})
	t.Run("configured spill", func(t *testing.T) {
		before := cloud.count()
		lookups, _ := run(t, local.ContextWindow, 20000)
		capacity := lookups[0].Execution.Targets[0].Capacity
		if len(lookups) != 1 || capacity == nil || capacity.Fits || *capacity.Required <= local.ContextWindow || cloud.count() != before+1 {
			t.Fatalf("spill capacity %s", mustJSON(capacity))
		}
	})
	t.Run("runtime refusal", func(t *testing.T) {
		// Configured larger than the real allocation: the core believes it
		// fits, the runtime refuses before output, the core re-consults.
		before := cloud.count()
		lookups, _ := run(t, 4*local.ContextWindow, 20000)
		if len(lookups) != 2 || *lookups[1].Execution.Turn.Reconsult != "capacity" || lookups[1].Execution.Targets[0].Capacity.Fits || cloud.count() != before+1 {
			t.Fatalf("runtime refusal lookups %d", len(lookups))
		}
	})
}

func mustJSON(value any) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}
