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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// errNotPortedBucketB marks a fork dispatch hook whose upstream seam was
// removed by PR #3076. It fails closed until the hook is re-seated on
// prepareProviderDispatch / applyDispatchDecision.
var errNotPortedBucketB = errors.New("rayline_arc dispatch hook not ported: re-seat on prepareProviderDispatch")

// modifyRequestBodyForRaylineARC applied the artifact-exact body contract on
// top of the pre-#3076 auto-routing body pipeline.
//
// TODO(vsr-next Bucket B): re-seat on prepareProviderDispatch. Upstream
// removed serializeOpenAIRequestWithStream, injectMemoryMessages and
// buildRequestParamsMutations; the neutral request is now rendered from
// ctx.SemanticRequest by prepareProviderRequest, so the ARC rewrite must move
// behind applyDispatchDecision or be deleted in favour of VSR-owned dispatch.
func (r *OpenAIRouter) modifyRequestBodyForRaylineARC(
	_ *openai.ChatCompletionNewParams,
	_ string,
	_ *config.ProviderProfile,
	ctx *RequestContext,
) ([]byte, error) {
	if ctx == nil || ctx.RaylineARCDispatch == nil {
		return nil, errors.New("ARC dispatch contract is unavailable")
	}
	return nil, errNotPortedBucketB
}

func applyRaylineARCDispatch(
	body []byte,
	worker *raylinearc.WorkerManifest,
) ([]byte, error) {
	if worker == nil {
		return nil, errors.New("ARC worker is unavailable")
	}
	object, err := decodeARCRequestObject(body)
	if err != nil {
		return nil, err
	}
	if err := normalizeARCCompletionAlias(object); err != nil {
		return nil, err
	}
	removeClientOwnedARCFields(object)
	object["model"] = worker.Model
	if worker.EffectiveDispatchBackend() == raylinearc.DispatchOpenRouter {
		object["provider"] = map[string]any{
			"order": append(
				[]string(nil),
				worker.OpenRouterProviderOrder...,
			),
			"allow_fallbacks":    worker.OpenRouterAllowFallbacks,
			"require_parameters": worker.OpenRouterRequireParameters,
		}
	}
	if mergeErr := mergeARCExtraBody(object, worker.ExtraBody); mergeErr != nil {
		return nil, mergeErr
	}
	return applyARCExecutionLimits(object, worker)
}

func normalizeARCCompletionAlias(object map[string]any) error {
	completionFromAlias, err := arcCompletionTokenField(
		object,
		"max_completion_tokens",
	)
	if err != nil {
		return err
	}
	if completionFromAlias > 0 {
		existing, existingErr := arcCompletionTokenField(object, "max_tokens")
		if existingErr != nil {
			return existingErr
		}
		if completionFromAlias > existing {
			object["max_tokens"] = completionFromAlias
		}
	}
	return nil
}

func removeClientOwnedARCFields(object map[string]any) {
	for _, key := range []string{
		"provider",
		"reasoning",
		"reasoning_effort",
		"chat_template_kwargs",
		"thinking",
		"context_management",
		"diagnostics",
		"output_config",
		"max_completion_tokens",
	} {
		delete(object, key)
	}
}

func applyARCExecutionLimits(
	object map[string]any,
	worker *raylinearc.WorkerManifest,
) ([]byte, error) {
	requested, err := arcCompletionTokenField(object, "max_tokens")
	if err != nil {
		return nil, err
	}
	completion := max(requested, worker.MinimumCompletionTokens)
	if worker.MaxCompletionTokens != nil &&
		completion > *worker.MaxCompletionTokens {
		completion = *worker.MaxCompletionTokens
	}
	if completion > 0 {
		object["max_tokens"] = completion
	} else {
		delete(object, "max_tokens")
	}
	if worker.Temperature == nil {
		delete(object, "temperature")
	} else {
		object["temperature"] = *worker.Temperature
	}
	materializeARCTools(object)
	return json.Marshal(object)
}

func decodeARCRequestObject(body []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var object map[string]any
	if err := decoder.Decode(&object); err != nil || object == nil {
		return nil, errors.New("ARC request must be a JSON object")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("ARC request must contain one JSON object")
	}
	return object, nil
}

func arcCompletionTokenField(
	object map[string]any,
	key string,
) (uint64, error) {
	value, exists := object[key]
	if !exists {
		return 0, nil
	}
	if parsed, ok := value.(uint64); ok {
		return parsed, nil
	}
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("ARC %s must be an integer", key)
	}
	parsed, err := number.Int64()
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("ARC %s must be a nonnegative integer", key)
	}
	return uint64(parsed), nil
}

func mergeARCExtraBody(
	object map[string]any,
	raw json.RawMessage,
) error {
	if len(raw) == 0 {
		// An arm that declares no extra body has nothing to merge; treating
		// that as malformed would fail an otherwise valid dispatch closed.
		return nil
	}
	extra, err := decodeARCRequestObject(raw)
	if err != nil {
		return errors.New("ARC extra body is invalid")
	}
	for key, value := range extra {
		object[key] = value
	}
	return nil
}

func materializeARCTools(object map[string]any) {
	tools, ok := object["tools"].([]any)
	if !ok {
		return
	}
	for _, value := range tools {
		tool, ok := value.(map[string]any)
		if !ok {
			continue
		}
		delete(tool, "defer_loading")
		delete(tool, "allowed_callers")
		delete(tool, "eager_input_streaming")
	}
}
