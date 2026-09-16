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
	"encoding/json"
	"errors"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// raylineRoutesAPIPath answers one question -- where should this request go --
// and the body it asks it with is the request the caller was about to send.
//
// It is served from here rather than from the management listener because the
// decision is a property of the data plane: the selector, the episode store
// and the turn normalizer all live behind this filter, and the ingress
// listener already delivers this path to it. Exposing the management listener
// would mean a new Envoy cluster for a surface that is not management.
const raylineRoutesAPIPath = "/v1/routes"

const (
	// raylineRoutesSessionHeader names the conversation this decision belongs
	// to. Absent means stateless, which is what a playground wants:
	// experimenting must not advance a real conversation's trajectory.
	raylineRoutesSessionHeader = "x-rayline-session"
	// raylineRoutesBranchHeader separates concurrent subagent lanes inside one
	// conversation, so they do not collapse onto a single trajectory.
	raylineRoutesBranchHeader = "x-rayline-branch"
	// raylineRoutesCheckpointHeader pins a checkpoint. This cell serves one
	// artifact, so a pin that names anything is reported rather than honoured.
	raylineRoutesCheckpointHeader = "x-rayline-checkpoint"
	// raylineRoutesRouteIDHeader carries a caller-minted id. Adopting it means
	// one id spans the caller's own record and this router's.
	raylineRoutesRouteIDHeader = "x-rayline-route-id"
)

// raylineRouteIDPrefix marks an id as naming a route rather than a request or
// a session. It is spelled out rather than reused from the legacy consult's
// rt_ because the two are minted by different parties and joining them by
// prefix would be a coincidence, not a contract.
const raylineRouteIDPrefix = "rte_"

// raylineRoutesResponse is the decision as a caller reads it.
//
// Every field here is a bounded fact about the choice. Provider order,
// fallback flags and dialect rewriting are deliberately absent: those are
// how this router would have executed the call, and executing it is the
// caller's job now.
type raylineRoutesResponse struct {
	RouteID      string                 `json:"route_id"`
	Object       string                 `json:"object"`
	Model        string                 `json:"model"`
	Thinking     raylineRoutesThinking  `json:"thinking"`
	Confidence   float64                `json:"confidence"`
	Reason       string                 `json:"reason,omitempty"`
	Checkpoint   string                 `json:"checkpoint,omitempty"`
	Alternatives []raylineRoutesScore   `json:"alternatives"`
	Baseline     *raylineRoutesBaseline `json:"baseline,omitempty"`
	Pricing      raylineRoutesPricing   `json:"selected_pricing"`
	Warnings     []string               `json:"warnings"`
	Episode      *raylineRoutesEpisode  `json:"episode,omitempty"`
	Usage        raylineRoutesUsage     `json:"usage"`
	LatencyMS    int64                  `json:"latency_ms"`
}

type raylineRoutesThinking struct {
	Mode string `json:"mode"`
	// BudgetTokens is null rather than zero when thinking is off, so a caller
	// reading it as a number cannot mistake "no budget applies" for "a budget
	// of zero applies".
	BudgetTokens *uint64 `json:"budget_tokens"`
}

type raylineRoutesScore struct {
	Model string  `json:"model"`
	Score float64 `json:"score"`
}

type raylineRoutesPricing struct {
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

type raylineRoutesBaseline struct {
	Model         string  `json:"model"`
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

type raylineRoutesEpisode struct {
	TurnIndex int  `json:"turn_index"`
	Stayed    bool `json:"stayed"`
}

type raylineRoutesUsage struct {
	EncodedInputTokens int `json:"encoded_input_tokens"`
	CacheReadTokens    int `json:"cache_read_tokens"`
}

// raylineRoutesAPIEnabled reports whether this deployment serves the endpoint.
//
// It is off until an operator turns it on, even where an ARC decision is
// configured. The endpoint drives the encoder with no paying turn behind it,
// so a cell should not acquire that traffic by upgrading.
func (r *OpenAIRouter) raylineRoutesAPIEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	for index := range r.Config.Decisions {
		algorithm := r.Config.Decisions[index].Algorithm
		if !raylineARCSelection(algorithm) || algorithm.RaylineARC == nil {
			continue
		}
		if algorithm.RaylineARC.RoutesAPI.Enabled {
			return true
		}
	}
	return false
}

// isRaylineRoutesRequest matches the request the body phase must answer itself
// instead of routing. Method is not re-checked: the header phase already
// refused everything but POST on this path.
func isRaylineRoutesRequest(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	return normalizeRequestPath(requestPathFromContext(ctx)) == raylineRoutesAPIPath
}

func requestPathFromContext(ctx *RequestContext) string {
	if ctx == nil || ctx.Headers == nil {
		return ""
	}
	return ctx.Headers[":path"]
}

// handleRaylineRoutesAPI answers one route lookup and returns an immediate
// response. It returns nil when the request is not a route lookup, so the
// body phase falls through to routing.
func (r *OpenAIRouter) handleRaylineRoutesAPI(
	body []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if !isRaylineRoutesRequest(ctx) {
		return nil
	}
	if !r.raylineRoutesAPIEnabled() {
		return r.createRaylineRoutesError(404, "not_found_error", "endpoint not found")
	}

	wireFormat, warnings, detail := readRaylineRoutesBody(body)
	if detail != "" {
		return r.createRaylineRoutesError(400, "invalid_request_error", detail)
	}
	warnings = append(warnings, raylineRoutesHeaderWarnings(ctx)...)

	runtime := r.routeDecisionRuntimeState()
	if runtime == nil {
		return r.createRaylineRoutesError(
			503,
			"api_error",
			"route lookup is not available on this router",
		)
	}

	routeID := raylineRoutesRouteID(ctx)
	started := time.Now()
	decision, err := runtime.RouteDecision(
		r.raylineRoutesContext(ctx),
		routerruntime.RouteDecisionRequest{
			Body:       body,
			DecisionID: routeID,
			SessionID:  raylineRoutesEpisodeIdentity(ctx),
			WireFormat: wireFormat,
		},
	)
	if err != nil {
		return r.raylineRoutesFailure(routeID, err)
	}

	logging.ComponentEvent("extproc", "routing_decision", map[string]interface{}{
		"routing_decision_id": routeID,
		"selected_worker":     decision.SelectedWorker,
		"worker_model":        decision.WorkerModel,
		"interface":           "routes",
	})
	return r.createJSONResponse(200, raylineRoutesPayload(
		routeID,
		decision,
		warnings,
		time.Since(started),
	))
}

// raylineRoutesContext keeps the decision on the request's own trace, and
// falls back to the processing context only when the header phase recorded
// none.
func (r *OpenAIRouter) raylineRoutesContext(ctx *RequestContext) context.Context {
	if ctx != nil && ctx.TraceContext != nil {
		return ctx.TraceContext
	}
	return context.Background()
}

func (r *OpenAIRouter) raylineRoutesFailure(
	routeID string,
	err error,
) *ext_proc.ProcessingResponse {
	contended := errors.Is(err, routerruntime.ErrRouteDecisionContended)
	logging.ComponentErrorEvent("extproc", "routing_decision_failed", map[string]interface{}{
		"routing_decision_id": routeID,
		"error":               err.Error(),
		"contended":           contended,
	})
	// Contention is not an outage. A 503 sends a caller into fallback and
	// reads as this router being down; a contended lookup is a healthy router
	// that is briefly busy with this very session, so it says so and says
	// when to come back.
	if contended {
		return r.createRaylineRoutesError(
			429,
			"rate_limit_error",
			"route lookup contended: the session or the encoder is briefly at capacity",
		)
	}
	// Fail closed. The selector owns the choice and has no fallback arm, so
	// answering with a default here would replace a policy decision with this
	// handler's guess.
	return r.createRaylineRoutesError(503, "api_error", "route lookup failed")
}

func raylineRoutesPayload(
	routeID string,
	decision routerruntime.RouteDecision,
	warnings []string,
	elapsed time.Duration,
) raylineRoutesResponse {
	payload := raylineRoutesResponse{
		RouteID:      routeID,
		Object:       "route",
		Model:        decision.WorkerModel,
		Thinking:     raylineRoutesThinkingOf(decision.Thinking),
		Confidence:   decision.Confidence,
		Reason:       decision.Reason,
		Checkpoint:   decision.Checkpoint,
		Alternatives: raylineRoutesAlternatives(decision.Alternatives),
		Pricing: raylineRoutesPricing{
			InputPerMTok:  decision.SelectedPricing.InputPerMTok,
			OutputPerMTok: decision.SelectedPricing.OutputPerMTok,
		},
		Warnings: append(warnings, decision.Warnings...),
		Usage: raylineRoutesUsage{
			EncodedInputTokens: decision.Usage.EncodedInputTokens,
			CacheReadTokens:    decision.Usage.CacheReadTokens,
		},
		LatencyMS: elapsed.Milliseconds(),
	}
	if payload.Warnings == nil {
		payload.Warnings = []string{}
	}
	if decision.Baseline.Model != "" {
		payload.Baseline = &raylineRoutesBaseline{
			Model:         decision.Baseline.Model,
			InputPerMTok:  decision.Baseline.Pricing.InputPerMTok,
			OutputPerMTok: decision.Baseline.Pricing.OutputPerMTok,
		}
	}
	if decision.Episode != nil {
		payload.Episode = &raylineRoutesEpisode{
			TurnIndex: decision.Episode.TurnIndex,
			Stayed:    decision.Episode.Stayed,
		}
	}
	return payload
}

func raylineRoutesThinkingOf(thinking routerruntime.RouteThinking) raylineRoutesThinking {
	rendered := raylineRoutesThinking{Mode: thinking.Mode}
	if thinking.BudgetTokens > 0 {
		budget := thinking.BudgetTokens
		rendered.BudgetTokens = &budget
	}
	return rendered
}

func raylineRoutesAlternatives(
	alternatives []routerruntime.RouteAlternative,
) []raylineRoutesScore {
	rendered := make([]raylineRoutesScore, 0, len(alternatives))
	for _, alternative := range alternatives {
		rendered = append(rendered, raylineRoutesScore{
			Model: alternative.Model,
			Score: alternative.Score,
		})
	}
	return rendered
}

// readRaylineRoutesBody discriminates the wire format from the body's own
// shape, so the same bytes serve this endpoint and the executing one.
//
// This is the discrimination the platform's own client already performs, kept
// here rather than pushed onto the caller as a protocol field: a field would
// be one more thing to get wrong, and getting it wrong would be silent.
func readRaylineRoutesBody(body []byte) (llmprotocol.WireFormat, []string, string) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil || envelope == nil {
		return "", nil, "request body must be a JSON object"
	}
	warnings := raylineRoutesBodyWarnings(envelope)
	if _, responses := envelope["input"]; responses {
		return llmprotocol.OpenAIResponsesV1, warnings, ""
	}
	raw, present := envelope["messages"]
	if !present {
		return "", nil, "request body must carry messages or input"
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil || len(messages) == 0 {
		return "", nil, "messages must be a non-empty list of message objects"
	}
	if _, chat := envelope["max_completion_tokens"]; chat {
		return llmprotocol.OpenAIChatV1, warnings, ""
	}
	return llmprotocol.AnthropicMessagesV1, warnings, ""
}

// raylineRoutesBodyWarnings names what this router did with the body other
// than what the body asked for.
//
// A developer who sends tools and gets a well-formed decision back has no way
// to learn that the tools were never encoded, because the answer looks right
// either way. That is the class of mistake this array exists to retire.
func raylineRoutesBodyWarnings(envelope map[string]json.RawMessage) []string {
	warnings := []string{}
	if _, present := envelope["tools"]; present {
		warnings = append(warnings, "tools_not_encoded: tools were dropped before encoding and did not influence this route")
	}
	return warnings
}

func raylineRoutesHeaderWarnings(ctx *RequestContext) []string {
	warnings := []string{}
	if raylineRoutesHeader(ctx, raylineRoutesCheckpointHeader) != "" {
		warnings = append(warnings, "checkpoint_not_pinned: this cell serves one checkpoint and the requested pin was ignored")
	}
	return warnings
}

// raylineRoutesEpisodeIdentity composes the episode this decision belongs to.
//
// The branch is folded into the identity rather than carried beside it,
// matching how the gateway already composes userId:conversationId[:branch]:
// two concurrent subagents on one conversation are two trajectories, and
// scoring them as one would make each read as the other's previous turn.
// An absent session yields an empty identity, and the runtime gives that call
// its own single-turn episode.
func raylineRoutesEpisodeIdentity(ctx *RequestContext) string {
	session := raylineRoutesHeader(ctx, raylineRoutesSessionHeader)
	if session == "" {
		return ""
	}
	branch := raylineRoutesHeader(ctx, raylineRoutesBranchHeader)
	if branch == "" {
		return session
	}
	return session + ":" + branch
}

// raylineRoutesRouteID adopts the caller's id when it sent one, so a single id
// spans their record and ours, and mints one otherwise.
func raylineRoutesRouteID(ctx *RequestContext) string {
	if adopted := raylineRoutesHeader(ctx, raylineRoutesRouteIDHeader); adopted != "" {
		return adopted
	}
	return raylineRouteIDPrefix + strings.ReplaceAll(uuid.NewString(), "-", "")[:8]
}

func raylineRoutesHeader(ctx *RequestContext, name string) string {
	if ctx == nil || ctx.Headers == nil {
		return ""
	}
	return strings.TrimSpace(ctx.Headers[name])
}

// createRaylineRoutesError answers in the Anthropic error envelope rather than
// this router's own, because a caller of this endpoint is already parsing that
// envelope from the endpoint it would otherwise have called. One error path
// for both is the point of taking the same body.
func (r *OpenAIRouter) createRaylineRoutesError(
	statusCode int,
	errorType string,
	message string,
) *ext_proc.ProcessingResponse {
	return r.createJSONResponse(statusCode, map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    errorType,
			"message": message,
		},
	})
}
