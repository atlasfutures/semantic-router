//go:build !windows && cgo

package extproc

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const userImagePNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

// userImageTurns are one image in a user message, in each client format.
var userImageTurns = map[string]struct{ path, body string }{
	"chat image_url": {"/v1/chat/completions", `{"model":"auto","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + userImagePNG + `"}}]}]}`},
	"messages image": {"/v1/messages", `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"What is in this image?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + userImagePNG + `"}}]}]}`},
	"responses input_image": {"/v1/responses", `{"model":"auto","input":[{"role":"user","content":[` +
		`{"type":"input_text","text":"What is in this image?"},{"type":"input_image","image_url":"data:image/png;base64,` + userImagePNG + `"}]}]}`},
}

// A user-message image (not a tool result) reaches the vision gate on the
// policy path: the text-only arm leaves the offer and cannot be dispatched --
// but only when its model card says vision: false. An unmarked card counts as
// vision-capable (ModelParams.SupportsVision), so the image's safety rests on
// the basket's cards (semantic-router#215).
func TestPolicyOffersAUserImageOnlyToVisionArms(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	off := actions["off"].ActionID
	claude := actions["claude"].ActionID
	for _, marked := range []bool{false, true} {
		fake := newRelaxedPolicyFake(t)
		path := writeConsistentPolicyConfig(t, fake.URL(), "strict")
		if marked {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			const card = "    - name: \"off\"\n      modality: text\n"
			if !strings.Contains(string(raw), card) {
				t.Fatalf("the config no longer carries %q", card)
			}
			edited := strings.Replace(string(raw), card, card+"      vision: false\n", 1)
			if err := os.WriteFile(path, []byte(edited), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		router, err := NewOpenAIRouter(path)
		if err != nil {
			t.Fatalf("build router: %v", err)
		}
		awaitPolicySelectorArmed(t, router)
		for name, turn := range userImageTurns {
			label := "unmarked card/" + name
			if marked {
				label = "vision: false/" + name
			}
			t.Run(label, func(t *testing.T) {
				var offered []string
				// The service picks the text-only arm whenever it is offered.
				fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
					offered = request.Selection.AvailableActionIDs
					if slices.Contains(offered, off) {
						return off
					}
					return claude
				})
				body := dispatchPolicyClientRequest(t, router, "episode-user-image-"+label, turn.path, turn.body)
				model := string(body["model"])
				if marked {
					if slices.Contains(offered, off) || model == `"vendor/off"` {
						t.Fatalf("the text-only arm was offered (%v) or dispatched (%s) for a user image", offered, model)
					}
					return
				}
				if !slices.Contains(offered, off) || model != `"vendor/off"` {
					t.Fatalf("unmarked: offered %v, dispatched %s", offered, model)
				}
			})
		}
	}
}
