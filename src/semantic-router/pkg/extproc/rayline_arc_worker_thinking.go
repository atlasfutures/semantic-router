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
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// applyRaylineARCWorkerThinking puts the worker's registry base level on the
// wire in place of whatever the router derived.
//
// The derived reasoning for a thinking turn is a reasoning.max_tokens bound
// from the client's output allowance, with any effort removed, because
// OpenRouter refuses the two together. Agent clients state an output
// allowance on every turn, so a worker configured to think at a named effort
// never did. A worker with a base gets exactly its wire -- an effort, a token
// budget, or nothing -- and no derived control beside it.
//
// It runs after the reasoning mutation and before the body is read back for
// the routing record, so the record names what travelled.
func applyRaylineARCWorkerThinking(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	base, ok := raylineARCWorkerThinkingFor(dispatch, ctx)
	if !ok {
		return body, nil
	}
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	delete(requestMap, "reasoning_effort")
	switch base.Wire {
	case config.RaylineARCWorkerThinkingEffort:
		requestMap["reasoning"], _ = json.Marshal(map[string]string{"effort": base.Effort})
	case config.RaylineARCWorkerThinkingBudget:
		requestMap["reasoning"], _ = json.Marshal(map[string]int64{"max_tokens": base.MaxTokens})
	default:
		delete(requestMap, "reasoning")
	}
	mutated, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	ctx.RaylineARCWorkerThinking = &base
	return mutated, nil
}

// raylineARCWorkerThinkingFor answers the gates in one place: an ARC worker
// with a base, dispatched as Chat to a provider that reads OpenRouter's
// reasoning object, on a turn the decision routed to a thinking arm.
func raylineARCWorkerThinkingFor(
	dispatch *providerDispatch,
	ctx *RequestContext,
) (config.RaylineARCWorkerThinkingConfig, bool) {
	if dispatch == nil || ctx == nil || ctx.RaylineARCDispatch == nil || !dispatch.useReasoning ||
		dispatch.targetFormat != llmprotocol.OpenAIChatV1 ||
		!usesReasoningObjectTransport(resolveProviderReasoningTransport(dispatch.profile)) {
		return config.RaylineARCWorkerThinkingConfig{}, false
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return config.RaylineARCWorkerThinkingConfig{}, false
	}
	base, ok := decision.Algorithm.RaylineARC.WorkerThinking[ctx.RaylineARCDispatch.ID]
	return base, ok
}
