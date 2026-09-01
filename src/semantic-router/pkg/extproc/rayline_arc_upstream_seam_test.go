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
	"errors"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// TestARCSelectionAddsNothingToProviderDispatch is the Bucket B contract. ARC
// answers one question -- which of the decision's models -- and the router
// builds the provider request from that model and its card. An armed ARC
// dispatch whose worker manifest carries provider pinning, its own credential
// environment and its own limits must therefore produce byte-identical
// dispatch headers to no ARC at all, and must add no body mutation.
func TestARCSelectionAddsNothingToProviderDispatch(t *testing.T) {
	router, model := seamTestRouter()
	dispatch, err := router.resolveProviderDispatch(model, "arc-decision", false)
	if err != nil {
		t.Fatal(err)
	}

	withoutARC := router.buildProviderDispatchResponse(
		dispatch,
		seamTestContext(nil),
	)
	withARC := router.buildProviderDispatchResponse(
		dispatch,
		seamTestContext(&raylinearc.WorkerManifest{
			ID:                          model,
			Model:                       "provider/model",
			APIKeyEnv:                   "ARC_TEST_PROVIDER_KEY",
			OpenRouterProviderOrder:     []string{"pinned-provider"},
			OpenRouterRequireParameters: true,
			ThinkingMode:                "on",
		}),
	)

	if got := dispatchHeaderNames(withARC); !equalStrings(got, dispatchHeaderNames(withoutARC)) {
		t.Fatalf("ARC changed the dispatch headers: %v", got)
	}
	if mutation := withARC.GetRequestBody().GetResponse().GetBodyMutation(); mutation != nil {
		t.Fatalf("ARC added a body mutation: %#v", mutation)
	}
}

// TestShedAnswers429AndTransportErrorAnswers503 pins the admission contract.
// A shed is capacity the caller can wait out, so it must be retryable and say
// how long. An encoder transport failure is breakage, so it must not invite a
// retry. Neither message may name a private component.
func TestShedAnswers429AndTransportErrorAnswers503(t *testing.T) {
	router := &OpenAIRouter{}
	tests := []struct {
		name           string
		class          string
		wantStatus     int
		wantRetryAfter string
	}{
		{
			name:           "admission shed",
			class:          arcEncoderFailureClassAdmission,
			wantStatus:     429,
			wantRetryAfter: "1",
		},
		{
			name:       "encoder transport error",
			class:      arcEncoderFailureClass(raylinearc.EncoderFailureTransport),
			wantStatus: 503,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := router.authoritativeSelectionFailureResponse(
				&modelSelectionFailure{algorithm: configRaylineARC, class: test.class},
				&RequestContext{},
			)
			immediate := response.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("response = %#v", response)
			}
			if got := int(immediate.GetStatus().GetCode()); got != test.wantStatus {
				t.Fatalf("status = %d, want %d", got, test.wantStatus)
			}
			if got := immediateHeaderValue(response, "retry-after"); got != test.wantRetryAfter {
				t.Fatalf("retry-after = %q, want %q", got, test.wantRetryAfter)
			}
			if body := string(immediate.GetBody()); strings.Contains(body, test.class) ||
				strings.Contains(strings.ToLower(body), "rayline") ||
				strings.Contains(strings.ToLower(body), "remote") {
				t.Fatalf("failure body leaks a private component: %s", body)
			}
		})
	}
}

// TestSelectionCommitFailureBecomesTyped503 proves an episode the router could
// not commit is never reported to the client as a provider success.
func TestSelectionCommitFailureBecomesTyped503(t *testing.T) {
	transaction := &recordingSelectionTransaction{
		commitErr: errors.New("private coordinator response"),
	}
	ctx := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	response, err := (&OpenAIRouter{}).handleResponseHeaders(
		arcResponseHeaders("200"),
		ctx,
	)
	if err != nil {
		t.Fatal(err)
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || int(immediate.GetStatus().GetCode()) != 503 {
		t.Fatalf("response = %#v", response)
	}
	if body := strings.ToLower(string(immediate.GetBody())); strings.Contains(body, "remote") ||
		strings.Contains(body, "rayline") ||
		strings.Contains(body, "coordinator") {
		t.Fatalf("failure body leaks a private component: %s", immediate.GetBody())
	}
	finalizeSelectionProcessTerminal(ctx)
	if transaction.commits != 1 || transaction.aborts != 1 {
		t.Fatalf("terminal calls = %#v", transaction)
	}
}

func seamTestRouter() (*OpenAIRouter, string) {
	const model = "arc-worker"
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: model,
			ModelConfig: map[string]config.ModelParams{
				model: {
					PreferredEndpoints: []string{"backend"},
					APIFormat:          config.APIFormatOpenAI,
					ExternalModelIDs:   map[string]string{"backend": "provider/model"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				Type: "openai", APIKey: "artifact-owned-key",
				APIKeyEnvName:       "ARC_TEST_PROVIDER_KEY",
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}, model
}

func seamTestContext(worker *raylinearc.WorkerManifest) *RequestContext {
	return &RequestContext{
		Headers:            map[string]string{"x-user-openai-key": "caller-key"},
		TargetFormat:       llmprotocol.OpenAIChatV1,
		RaylineARCDispatch: worker,
	}
}

func dispatchHeaderNames(response *ext_proc.ProcessingResponse) []string {
	mutation := response.GetRequestBody().GetResponse().GetHeaderMutation()
	names := make([]string, 0, len(mutation.GetSetHeaders()))
	for _, option := range mutation.GetSetHeaders() {
		names = append(names, strings.ToLower(option.GetHeader().GetKey()))
	}
	return names
}

func equalStrings(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}
