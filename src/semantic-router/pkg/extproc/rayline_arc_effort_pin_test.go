//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// v5GLMHighNone is a GLM steering action at the high base (registry control
// ed143fc8, level none): the fixture's GLM actions are all at the default
// base, so binding it gives one worker two request-level efforts, as the
// prod package gives Opus.
const (
	v5GLMHighNone        = "4b8f0d6e2a1c3e5f7a9b0c2d4e6f8a1b3c5d7e9f0a2b4c6d8e0f1a3b5c7d9e1f"
	v5GLMHighNoneControl = "ed143fc8288b25589866829f3a7f4caa1382220286b0100383f49f87a3722e79"
)

// withGLMHighAction adds v5GLMHighNone: v5GLMNone's action at the high base.
func withGLMHighAction(manifest []byte) ([]byte, []string) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(manifest, &doc); err != nil {
		panic(err)
	}
	var actions []map[string]any
	if err := json.Unmarshal(doc["actions"], &actions); err != nil {
		panic(err)
	}
	for _, action := range actions {
		if action["action_id"] != v5GLMNone {
			continue
		}
		encoded, _ := json.Marshal(action)
		var high map[string]any
		_ = json.Unmarshal(encoded, &high)
		high["action_id"] = v5GLMHighNone
		high["control_id"] = v5GLMHighNoneControl
		high["control"].(map[string]any)["base"] = map[string]any{"native": "high"}
		actions = append(actions, high)
		break
	}
	encoded, err := json.Marshal(actions)
	if err != nil {
		panic(err)
	}
	doc["actions"] = encoded
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		panic(err)
	}
	return out, []string{v5GLMHighNone}
}

func bindGLMHigh(config string) string {
	up := "              - action_id: " + v5GLMUp + "\n                worker: glm\n"
	return strings.Replace(config, up, up+"              - action_id: "+v5GLMHighNone+"\n                worker: glm\n", 1)
}

// Once a worker's transcript has started, its offer keeps only the actions
// at that transcript's base: a change of request-level effort would restart
// the provider's cached transcript (pathfinder ADR 0085, decision 1). The
// instruction level still varies.
func TestV5OfferKeepsAWorkerAtItsTranscriptsBase(t *testing.T) {
	router, fake := v5RouterWithManifest(t, "anthropic", bindGLMHigh, withGLMHighAction)
	router.Config.KeepUnservedClaudeThinking = true
	for _, tc := range []struct {
		first, second string
		masked        []string
		offered       []string
	}{
		{first: v5GLMUp, second: v5GLMNone, masked: []string{v5GLMHighNone}, offered: []string{v5GLMUp, v5GLMNone}},
		{first: v5GLMHighNone, second: v5GLMHighNone, masked: []string{v5GLMUp, v5GLMNone}, offered: []string{v5GLMHighNone}},
	} {
		episode := "v5-effort-pin-" + tc.first[:8]
		first := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"step 0"}]}`
		v5Turn(t, router, fake, tc.first, episode, first)

		var mu sync.Mutex
		var offer []string
		fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			mu.Lock()
			defer mu.Unlock()
			offer = append([]string(nil), request.Selection.AvailableActionIDs...)
			return tc.second
		})
		second := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"step 0"},` +
			`{"role":"assistant","content":"reply 0"},{"role":"user","content":"step 1"}]}`
		v5TurnWithChooser(t, router, episode, second)
		for _, id := range tc.masked {
			if slices.Contains(offer, id) {
				t.Fatalf("after %s, %s was offered at another base: %v", tc.first[:8], id[:8], short(offer))
			}
		}
		for _, id := range tc.offered {
			if !slices.Contains(offer, id) {
				t.Fatalf("after %s, %s was not offered: %v", tc.first[:8], id[:8], short(offer))
			}
		}
	}
}

func short(ids []string) string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, id[:8])
	}
	return fmt.Sprint(out)
}
