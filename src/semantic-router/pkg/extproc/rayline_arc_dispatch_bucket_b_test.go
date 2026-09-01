//go:build vsr_next_bucket_b

// Parked until Bucket B re-seats the ARC dispatch hooks on upstream's
// prepareProviderDispatch / applyDispatchDecision seam. Build with
// -tags vsr_next_bucket_b once those symbols exist again.

/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// TestToolMutationCannotEraseARCDispatch proves a later tool-selection body
// rewrite cannot revert the artifact-owned model, provider policy, or limits.
func TestToolMutationCannotEraseARCDispatch(t *testing.T) {
	worker := &raylinearc.WorkerManifest{
		Model:                       "artifact/provider-model",
		OpenRouterProviderOrder:     []string{"pinned-provider"},
		OpenRouterRequireParameters: true,
		ThinkingMode:                "off",
	}
	ctx := &RequestContext{Headers: map[string]string{}, RaylineARCDispatch: worker}
	// A tool mutation reserializes the client request, losing ARC shaping.
	clientBody := []byte(`{"model":"MoM","messages":[],"provider":{"order":["attacker"]}}`)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: clientBody},
					},
				},
			},
		},
	}

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.reapplyRaylineARCDispatch(response, ctx)

	final := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var object map[string]any
	if err := json.Unmarshal(final, &object); err != nil {
		t.Fatal(err)
	}
	if object["model"] != "artifact/provider-model" {
		t.Fatalf("artifact model was lost: %#v", object["model"])
	}
	provider, ok := object["provider"].(map[string]any)
	if !ok {
		t.Fatalf("artifact provider policy was lost: %#v", object["provider"])
	}
	order, ok := provider["order"].([]any)
	if !ok || len(order) != 1 || order[0] != "pinned-provider" {
		t.Fatalf("client provider order survived: %#v", provider["order"])
	}
	if provider["require_parameters"] != true {
		t.Fatalf("artifact parameter policy was lost: %#v", provider)
	}
}

// TestReapplyFailsClosedOnUnshapeableBody proves a body the artifact contract
// cannot shape is cleared rather than forwarded unshaped.
func TestReapplyFailsClosedOnUnshapeableBody(t *testing.T) {
	ctx := &RequestContext{
		Headers:            map[string]string{},
		RaylineARCDispatch: &raylinearc.WorkerManifest{Model: "artifact/model"},
	}
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{
							Body: []byte("not json"),
						},
					},
				},
			},
		},
	}

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.reapplyRaylineARCDispatch(response, ctx)

	if body := response.GetRequestBody().GetResponse().
		GetBodyMutation().GetBody(); len(body) != 0 {
		t.Fatalf("unshapeable body was forwarded: %q", body)
	}
}
