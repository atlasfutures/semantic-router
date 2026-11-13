//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// v5GLMNative is a GLM action with no instruction, at the default base: the
// shape the shared fixture gives Opus. A package that offers it beside the
// GLM steering actions switches a GLM transcript between lever presences.
const v5GLMNative = "9c6b2f0d4e0a4f8f8f5e3a1d2c7b6a5948372615a4b3c2d1e0f9e8d7c6b5a493"

// withGLMNativeAction adds v5GLMNative to the fixture's manifest, under the
// registry's native-only default control, the one the fixture's Opus action
// names.
func withGLMNativeAction(manifest []byte) ([]byte, []string) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(manifest, &doc); err != nil {
		panic(err)
	}
	var actions []map[string]json.RawMessage
	if err := json.Unmarshal(doc["actions"], &actions); err != nil {
		panic(err)
	}
	var native map[string]json.RawMessage
	for _, action := range actions {
		if string(action["action_id"]) == `"`+v5OpusAction+`"` {
			native = map[string]json.RawMessage{}
			for key, value := range action {
				native[key] = value
			}
		}
	}
	native["action_id"] = json.RawMessage(`"` + v5GLMNative + `"`)
	native["model"] = json.RawMessage(`"glm-5.3-flash"`)
	actions = append(actions, native)
	encoded, err := json.Marshal(actions)
	if err != nil {
		panic(err)
	}
	doc["actions"] = encoded
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		panic(err)
	}
	return out, []string{v5GLMNative}
}

// bindGLMNative binds v5GLMNative to the GLM worker, on the rendered config.
func bindGLMNative(config string) string {
	up := "              - action_id: " + v5GLMUp + "\n                worker: glm\n"
	return strings.Replace(config, up, up+"              - action_id: "+v5GLMNative+"\n                worker: glm\n", 1)
}

// providerMessages is a provider body's messages, each re-encoded without
// cache_control: the client moves its cache breakpoints every turn, and
// a breakpoint is not part of the cached prefix.
func providerMessages(t *testing.T, body []byte) []string {
	t.Helper()
	var provider struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &provider); err != nil {
		t.Fatal(err)
	}
	out := make([]string, 0, len(provider.Messages))
	for _, message := range provider.Messages {
		delete(message, "cache_control")
		if blocks, ok := message["content"].([]any); ok {
			for _, block := range blocks {
				if object, ok := block.(map[string]any); ok {
					delete(object, "cache_control")
				}
			}
		}
		encoded, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, string(encoded))
	}
	return out
}

// A worker's provider-bound transcript only ever grows, whichever of the
// worker's actions the policy draws: every item an earlier turn wrote is
// replayed at its anchor, byte for byte, so the provider's prompt cache
// survives every turn boundary. The policy here switches one GLM
// conversation between steering actions and a native-only one, as the prod
// package switches Opus between its effort and instruction shapes.
func TestV5TranscriptStaysAppendOnlyAcrossActionShapes(t *testing.T) {
	router, fake := v5RouterWithManifest(t, "anthropic", bindGLMNative, withGLMNativeAction)
	router.Config.KeepUnservedClaudeThinking = true
	schedule := []string{v5GLMUp, v5GLMNative, v5GLMNone, v5GLMUp, v5GLMNative, v5GLMNative, v5GLMNone}
	var history []string
	var previous []string
	for turn, action := range schedule {
		history = append(history, fmt.Sprintf(`{"role":"user","content":"step %d"}`, turn))
		client := `{"model":"auto","max_tokens":1024,"messages":[` + strings.Join(history, ",") + `]}`
		body, ctx := v5Turn(t, router, fake, action, "v5-append-only", client)
		current := providerMessages(t, body)
		if len(current) < len(previous) {
			t.Fatalf("turn %d (%s): the transcript shrank from %d to %d messages", turn, action[:8], len(previous), len(current))
		}
		for index := range previous {
			if current[index] != previous[index] {
				t.Fatalf("turn %d (%s): message %d changed, so the provider's cached prefix ends there\n was %s\n now %s",
					turn, action[:8], index, previous[index], current[index])
			}
		}
		if trace := ctx.RaylineARCThinking; trace != nil && trace.ResetReason != "" {
			t.Fatalf("turn %d (%s): the epoch reset (%s) on a transcript the client only appended to", turn, action[:8], trace.ResetReason)
		}
		previous = append(current, fmt.Sprintf(`{"content":"reply %d","role":"assistant"}`, turn))
		history = append(history, fmt.Sprintf(`{"role":"assistant","content":"reply %d"}`, turn))
	}
	if !bytes.Contains([]byte(strings.Join(previous, "")), []byte(v5UpText)) {
		t.Fatal("the schedule never steered, so it proves nothing")
	}
}
