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
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestBuildRaylineARCSelectionContextParsesExactCloseSignal(t *testing.T) {
	tests := []struct {
		name        string
		header      string
		wantClose   bool
		wantFailure string
	}{
		{name: "absent"},
		{name: "false", header: "false"},
		{name: "true", header: "true", wantClose: true},
		{name: "uppercase rejected", header: "TRUE", wantFailure: "invalid_close_signal"},
		{name: "other rejected", header: "1", wantFailure: "invalid_close_signal"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store, err := raylinearc.NewMemoryEpisodeStore(
				raylinearc.MemoryEpisodeStoreConfig{
					MaxEpisodes: 4,
					IdleTTL:     time.Minute,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			router := &OpenAIRouter{RaylineARCEpisodeStore: store}
			requestContext := &RequestContext{
				Headers: map[string]string{
					"x-rayline-episode-id":    t.Name(),
					"x-rayline-episode-close": test.header,
				},
				SourceFormat: llmprotocol.OpenAIChatV1,
				// The selector reads the decoded request, so the episode is
				// prepared from SemanticRequest and not from the wire body.
				SemanticRequest: &llmprotocol.Request{
					Generation: 1,
					Messages: []llmprotocol.Message{{
						Role: llmprotocol.RoleUser,
						Content: []llmprotocol.Content{{
							Kind: llmprotocol.ContentText,
							Text: "public test turn",
						}},
					}},
				},
				TraceContext: context.Background(),
			}
			algorithm := &config.AlgorithmConfig{
				Type: config.RaylineARCAlgorithmType,
				RaylineARC: &config.RaylineARCAlgorithmConfig{
					Episode: config.RaylineARCEpisodeConfig{
						IDHeader:              "x-rayline-episode-id",
						CloseHeader:           "x-rayline-episode-close",
						AcquireTimeoutSeconds: 1,
						LeaseTTLSeconds:       30,
					},
				},
			}
			selectionContext := router.buildRaylineARCSelectionContext(
				algorithm,
				requestContext,
				[]config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}},
			)
			if selectionContext.PreparationFailure != test.wantFailure ||
				requestContext.RaylineARCCloseRequested != test.wantClose {
				t.Fatalf(
					"close/failure = %t/%q, want %t/%q",
					requestContext.RaylineARCCloseRequested,
					selectionContext.PreparationFailure,
					test.wantClose,
					test.wantFailure,
				)
			}
			if requestContext.RaylineARCTransaction != nil {
				router.finalizeRaylineARCAbort(requestContext, "test_cleanup")
			}
		})
	}
}
