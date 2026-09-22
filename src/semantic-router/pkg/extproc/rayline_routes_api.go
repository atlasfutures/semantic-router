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
	"net/url"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	// raylineRoutesFormatHeader lets a caller state the dialect instead of
	// having it inferred. The body stays exactly what they were going to
	// send, which is the point of the endpoint, and the one thing the body
	// cannot always say about itself is which of two overlapping dialects it
	// is written in.
	raylineRoutesFormatHeader = "x-rayline-format"
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
	RouteID string `json:"route_id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	// Worker and Provider identify WHICH arm was selected. Two arms may serve
	// the same model through different providers at different prices -- the
	// manifest requires worker ids to be unique, not model names -- so the
	// model alone cannot be executed faithfully.
	Worker       string                 `json:"worker"`
	Provider     string                 `json:"provider,omitempty"`
	Thinking     raylineRoutesThinking  `json:"thinking"`
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
	Model string `json:"model"`
	// Which arm, not just which model. Two arms can serve one model through
	// different providers at different prices, so a list keyed on the model
	// alone repeats itself and explains nothing.
	Worker   string  `json:"worker"`
	Provider string  `json:"provider,omitempty"`
	Score    float64 `json:"score"`
}

type raylineRoutesPricing struct {
	InputPerMTok      float64 `json:"input_per_mtok"`
	OutputPerMTok     float64 `json:"output_per_mtok"`
	CacheReadPerMTok  float64 `json:"cache_read_per_mtok"`
	CacheWritePerMTok float64 `json:"cache_write_per_mtok"`
}

type raylineRoutesBaseline struct {
	Model             string  `json:"model"`
	InputPerMTok      float64 `json:"input_per_mtok"`
	OutputPerMTok     float64 `json:"output_per_mtok"`
	CacheReadPerMTok  float64 `json:"cache_read_per_mtok"`
	CacheWritePerMTok float64 `json:"cache_write_per_mtok"`
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
	return r.raylineRoutesConfig() != nil
}

// raylineRoutesEncodesToolNames reports whether this cell shows the selector
// the turn's tool names, which decides what the tools warning can honestly
// claim.
func (r *OpenAIRouter) raylineRoutesEncodesToolNames() bool {
	if r == nil || r.Config == nil {
		return false
	}
	for index := range r.Config.Decisions {
		algorithm := r.Config.Decisions[index].Algorithm
		if !raylineARCSelection(algorithm) || algorithm.RaylineARC == nil {
			continue
		}
		if algorithm.RaylineARC.RoutesAPI.Enabled {
			return algorithm.RaylineARC.IncludeToolNames
		}
	}
	return false
}

// raylineRoutesConfig returns the endpoint's configuration, or nil where this
// deployment does not serve it.
func (r *OpenAIRouter) raylineRoutesConfig() *config.RaylineARCRoutesAPIConfig {
	if r == nil || r.Config == nil {
		return nil
	}
	for index := range r.Config.Decisions {
		algorithm := r.Config.Decisions[index].Algorithm
		if !raylineARCSelection(algorithm) || algorithm.RaylineARC == nil {
			continue
		}
		if algorithm.RaylineARC.RoutesAPI.Enabled {
			return &algorithm.RaylineARC.RoutesAPI
		}
	}
	return nil
}

// raylineRoutesCheckpoint composes the public checkpoint identity.
//
// The hash alone identifies the artifact but tells a reader nothing; the
// label alone is not unique across deployments that reuse a release name. The
// pair is readable and still pins exactly one artifact.
func raylineRoutesCheckpoint(label string, revisionHash string) string {
	if label == "" {
		return revisionHash
	}
	if revisionHash == "" {
		return label
	}
	return label + "." + revisionHash
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
		return r.createRaylineRoutesError(ctx, 404, "not_found_error", "endpoint not found")
	}
	// The header phase already refused every other method. Checking again is
	// cheap, and the alternative is that one path reaching here without that
	// validation answers a GET with a route.
	if method := raylineRoutesHeader(ctx, ":method"); method != "" && method != "POST" {
		return r.createRaylineRoutesError(ctx, 405, "invalid_request_error", "method not allowed")
	}

	settings := r.raylineRoutesConfig()
	wireFormat, warnings, detail := readRaylineRoutesBody(body, r.raylineRoutesEncodesToolNames(), raylineRoutesHeader(ctx, raylineRoutesFormatHeader))
	if detail != "" {
		return r.createRaylineRoutesError(ctx, 400, "invalid_request_error", detail)
	}
	warnings = append(warnings, raylineRoutesHeaderWarnings(ctx, settings)...)
	// The format the body was actually read under, recorded before selection
	// runs. wireFormatOf reads this when a decode fails, and without the
	// update it still held the path-derived default -- so a caller who sent
	// x-rayline-format: anthropic, and whose body the Anthropic codec then
	// refused, was told it "could not be read as openai.chat.v1 messages".
	if ctx != nil {
		ctx.SourceFormat = wireFormat
	}

	runtime := r.routeDecisionRuntimeState()
	if runtime == nil {
		return r.createRaylineRoutesError(
			ctx,
			503,
			"api_error",
			"route lookup is not available on this router",
		)
	}

	// The deadline is this endpoint's own, not the encoder's. The encoder is
	// configured for a dispatched turn that streams for minutes and wraps
	// whatever context it is handed; the caller here is waiting on an answer,
	// so the bound has to be imposed from outside.
	lookupContext, cancel := context.WithTimeout(
		r.raylineRoutesContext(ctx),
		settings.EffectiveDeadline(),
	)
	defer cancel()

	routeID := raylineRoutesRouteID(ctx)
	episode := raylineRoutesEpisodeIdentity(ctx)
	// Ephemeral unless this cell keeps episodes and the caller named a
	// conversation. Both halves are required: without the identity there is
	// no trajectory to join, and without the setting this cell does not keep
	// one to join it to.
	ephemeral := episode == "" || !settings.EpisodeWrites
	started := time.Now()
	decision, err := runtime.RouteDecision(
		lookupContext,
		raylineRoutesConsult(body, routeID, episode, ephemeral, wireFormat),
	)
	if err != nil {
		return r.raylineRoutesFailure(ctx, lookupContext, routeID, err)
	}

	logging.ComponentEvent("extproc", "routing_decision", map[string]interface{}{
		"routing_decision_id": routeID,
		"selected_worker":     decision.SelectedWorker,
		"worker_model":        decision.WorkerModel,
		"interface":           "routes",
	})
	// Same claim as the error path: this body is the route contract, not an
	// inference response to be re-encoded into a client protocol.
	ctx.ImmediateResponseEncoded = true
	return r.createJSONResponse(200, raylineRoutesPayload(
		routeID,
		raylineRoutesCheckpoint(settings.CheckpointLabel, decision.Checkpoint),
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

// raylineRoutesFailure classifies a failed lookup.
//
// Expiry is read from the lookup's own context rather than from the error.
// The selector converts every failure into a bounded class string on purpose,
// so a cancelled encode arrives here as "encoder" and a cancelled lease as
// "episode_timeout", neither of which unwraps to context.DeadlineExceeded.
// The context is the only thing that still knows, and it is sufficient: if
// this deadline fired, the lookup ran out of time whatever else also went
// wrong on the way.
func (r *OpenAIRouter) raylineRoutesFailure(
	ctx *RequestContext,
	lookupContext context.Context,
	routeID string,
	err error,
) *ext_proc.ProcessingResponse {
	contended := errors.Is(err, routerruntime.ErrRouteDecisionContended)
	invalid := errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody)
	expired := errors.Is(lookupContext.Err(), context.DeadlineExceeded)
	logging.ComponentErrorEvent("extproc", "routing_decision_failed", map[string]interface{}{
		"routing_decision_id": routeID,
		"error":               err.Error(),
		"contended":           contended,
		"deadline_exceeded":   expired,
	})
	// A body the codec refused is the caller's to fix. It passed the shallow
	// envelope check and failed the real decode -- a bad role, a malformed
	// content block, a missing required field -- and no amount of retrying
	// changes that, so answering 503 would blame a healthy router for a
	// request that can never succeed.
	if invalid {
		return r.createRaylineRoutesError(
			ctx,
			400,
			"invalid_request_error",
			"request body could not be read as "+string(wireFormatOf(ctx))+" messages",
		)
	}
	// A lookup that ran out of time is reported as exactly that. Folding it
	// into the 503 would tell the caller this router is unhealthy when the
	// honest answer is that it did not answer in the budget they are entitled
	// to, and the two have different fixes.
	if expired {
		return r.createRaylineRoutesError(
			ctx,
			504,
			"timeout_error",
			"route lookup exceeded its deadline",
		)
	}
	// Contention is not an outage. A 503 sends a caller into fallback and
	// reads as this router being down; a contended lookup is a healthy router
	// that is briefly busy with this very session, so it says so and says
	// when to come back.
	if contended {
		return raylineRoutesRetryAfter(r.createRaylineRoutesError(
			ctx,
			429,
			"rate_limit_error",
			"route lookup contended: the session or the encoder is briefly at capacity",
		), raylineRoutesContendedRetrySeconds)
	}
	// Fail closed. The selector owns the choice and has no fallback arm, so
	// answering with a default here would replace a policy decision with this
	// handler's guess.
	return r.createRaylineRoutesError(ctx, 503, "api_error", "route lookup failed")
}

func raylineRoutesPayload(
	routeID string,
	checkpoint string,
	decision routerruntime.RouteDecision,
	warnings []string,
	elapsed time.Duration,
) raylineRoutesResponse {
	payload := raylineRoutesResponse{
		RouteID:      routeID,
		Object:       "route",
		Model:        decision.WorkerModel,
		Worker:       decision.SelectedWorker,
		Provider:     decision.Provider,
		Thinking:     raylineRoutesThinkingOf(decision.Thinking),
		Checkpoint:   checkpoint,
		Alternatives: raylineRoutesAlternatives(decision.Alternatives),
		Pricing:      raylineRoutesPricingOf(decision.SelectedPricing),
		Warnings:     append(warnings, decision.Warnings...),
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
		baseline := raylineRoutesPricingOf(decision.Baseline.Pricing)
		payload.Baseline = &raylineRoutesBaseline{
			Model:             decision.Baseline.Model,
			InputPerMTok:      baseline.InputPerMTok,
			OutputPerMTok:     baseline.OutputPerMTok,
			CacheReadPerMTok:  baseline.CacheReadPerMTok,
			CacheWritePerMTok: baseline.CacheWritePerMTok,
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

func raylineRoutesPricingOf(pricing routerruntime.RoutePricing) raylineRoutesPricing {
	return raylineRoutesPricing{
		InputPerMTok:      pricing.InputPerMTok,
		OutputPerMTok:     pricing.OutputPerMTok,
		CacheReadPerMTok:  pricing.CacheReadPerMTok,
		CacheWritePerMTok: pricing.CacheWritePerMTok,
	}
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
			Model:    alternative.Model,
			Worker:   alternative.Worker,
			Provider: alternative.Provider,
			Score:    alternative.Score,
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
func readRaylineRoutesBody(
	body []byte,
	encodesToolNames bool,
	formatHint string,
) (llmprotocol.WireFormat, []string, string) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil || envelope == nil {
		return "", nil, "request body must be a JSON object"
	}
	warnings := raylineRoutesBodyWarnings(envelope, encodesToolNames)
	_, hasInput := envelope["input"]
	raw, hasMessages := envelope["messages"]
	if !hasInput && !hasMessages {
		return "", nil, "request body must carry messages or input"
	}
	if hasMessages {
		var messages []json.RawMessage
		if err := json.Unmarshal(raw, &messages); err != nil || len(messages) == 0 {
			return "", nil, "messages must be a non-empty list of message objects"
		}
	}
	// The header is consulted before any inference, including the Responses
	// one. A caller who states the dialect has said the one thing the body
	// cannot always say about itself, and an override that some body shapes
	// silently outrank is not an override -- it is a second inference rule
	// the caller has no way to predict.
	if declared, ok := declaredWireFormat(formatHint); ok {
		return declared, warnings, ""
	}
	if hasInput {
		return llmprotocol.OpenAIResponsesV1, warnings, ""
	}
	format, certain := messagesDialect(envelope)
	if !certain {
		// Say so rather than route on a coin flip. The two dialects decode a
		// bare message array almost identically, so this is rarely visible --
		// which is exactly why it needs announcing when it happens.
		warnings = append(warnings, "format_inferred: the body carries no field unique to Anthropic Messages or Chat Completions, and was read as "+string(format))
	}
	return format, warnings, ""
}

// anthropicOnlyFields and chatOnlyFields are members that exist in one
// messages dialect and not the other. Neither list has to be exhaustive: it
// only has to be right, because a field in the wrong list would misread a
// well-formed body.
var (
	anthropicOnlyFields = []string{"system", "stop_sequences", "anthropic_version", "thinking"}
	// Roles Chat carries inside the message array. Anthropic hoists system
	// text to a top-level field, so a system or developer turn in messages[]
	// is Chat by construction.
	chatOnlyRoles  = map[string]bool{"system": true, "developer": true, "tool": true}
	chatOnlyFields = []string{
		"max_completion_tokens", "frequency_penalty", "presence_penalty",
		"logprobs", "top_logprobs", "n", "seed", "logit_bias",
	}
)

// messagesDialect separates Anthropic Messages from Chat Completions, and
// reports whether the body actually said which it was.
//
// An earlier version keyed solely on max_completion_tokens, so a Chat request
// that omitted it -- the field is optional -- was read as Anthropic. Reading
// several markers, in a defined order, removes that single point of failure:
// a body can only be misread now if it carries markers from both dialects,
// which is already malformed.
func messagesDialect(envelope map[string]json.RawMessage) (llmprotocol.WireFormat, bool) {
	if format, present := messagesRoleDialect(envelope["messages"]); present {
		return format, true
	}
	if format, present := messagesToolsDialect(envelope["tools"]); present {
		return format, true
	}
	for _, field := range anthropicOnlyFields {
		if _, present := envelope[field]; present {
			return llmprotocol.AnthropicMessagesV1, true
		}
	}
	for _, field := range chatOnlyFields {
		if _, present := envelope[field]; present {
			return llmprotocol.OpenAIChatV1, true
		}
	}
	// max_tokens deliberately decides nothing. It is required by Anthropic
	// Messages and accepted by Chat as a legacy field, so a Chat body using
	// it was being read as Anthropic and then interpreted under the wrong
	// codec. A field both dialects accept cannot discriminate between them.
	// A body with no marker at all is Chat. It cannot be valid Anthropic --
	// max_tokens is required there and would have been caught above -- so
	// defaulting the other way made every ordinary Chat request fail the
	// Anthropic codec and answer 400.
	return llmprotocol.OpenAIChatV1, false
}

// raylineRoutesBodyWarnings names what this router did with the body other
// than what the body asked for.
//
// A developer who sends tools and gets a well-formed decision back has no way
// to learn that the tools were never encoded, because the answer looks right
// either way. That is the class of mistake this array exists to retire.
// messagesRoleDialect reads the dialect off the message roles.
//
// A system, developer or tool role inside messages[] is Chat by
// construction, because Anthropic carries system text in a top-level field
// and tool results as content blocks on a user turn. A message carrying
// tool_calls is Chat for the same reason.
// declaredWireFormat reads an explicit dialect from the caller. An
// unrecognised value is ignored rather than refused: the body is still
// readable by inference, and failing a well-formed request over a header
// typo would be a worse answer than the one inference gives.
func declaredWireFormat(hint string) (llmprotocol.WireFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(hint)) {
	case "anthropic", string(llmprotocol.AnthropicMessagesV1):
		return llmprotocol.AnthropicMessagesV1, true
	case "chat", string(llmprotocol.OpenAIChatV1):
		return llmprotocol.OpenAIChatV1, true
	case "responses", string(llmprotocol.OpenAIResponsesV1):
		return llmprotocol.OpenAIResponsesV1, true
	default:
		return "", false
	}
}

func messagesRoleDialect(raw json.RawMessage) (llmprotocol.WireFormat, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		return "", false
	}
	for _, message := range messages {
		if _, present := message["tool_calls"]; present {
			return llmprotocol.OpenAIChatV1, true
		}
		if _, present := message["tool_call_id"]; present {
			return llmprotocol.OpenAIChatV1, true
		}
		// Anthropic carries a tool call and its result as content blocks on
		// ordinary user and assistant turns. Such a body names no dialect any
		// other way -- its roles are ordinary and max_tokens says nothing --
		// so without this the commonest agentic Anthropic request in
		// existence falls through to Chat and is refused by a codec that has
		// no such blocks.
		if format, present := messageContentDialect(message["content"]); present {
			return format, true
		}
		var role string
		if err := json.Unmarshal(message["role"], &role); err != nil {
			continue
		}
		if chatOnlyRoles[role] {
			return llmprotocol.OpenAIChatV1, true
		}
	}
	return "", false
}

// anthropicOnlyContentBlocks are content block types Anthropic Messages
// defines and Chat Completions does not. Chat's own array form carries text
// and image_url blocks, and expresses tool traffic with the tool_calls member
// and the tool role instead, so these types cannot appear in a Chat body.
var anthropicOnlyContentBlocks = map[string]bool{
	"tool_use":          true,
	"tool_result":       true,
	"thinking":          true,
	"redacted_thinking": true,
}

// messageContentDialect reads the dialect off one message's content blocks.
// A string content, which both dialects accept, says nothing.
func messageContentDialect(raw json.RawMessage) (llmprotocol.WireFormat, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", false
	}
	for _, block := range blocks {
		var blockType string
		if err := json.Unmarshal(block["type"], &blockType); err != nil {
			continue
		}
		if anthropicOnlyContentBlocks[blockType] {
			return llmprotocol.AnthropicMessagesV1, true
		}
	}
	return "", false
}

// messagesToolsDialect reads the dialect off the tool declarations.
//
// The two dialects declare the same tool differently and unmistakably:
// Anthropic puts the JSON Schema on input_schema at the top of the tool
// object, and Chat nests the whole callable under a function member. A body
// whose only distinguishing feature is its tools -- ordinary roles, plain
// string content, max_tokens -- is otherwise read as Chat and then refused.
func messagesToolsDialect(raw json.RawMessage) (llmprotocol.WireFormat, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil {
		return "", false
	}
	for _, tool := range tools {
		if _, present := tool["function"]; present {
			return llmprotocol.OpenAIChatV1, true
		}
		if _, present := tool["input_schema"]; present {
			return llmprotocol.AnthropicMessagesV1, true
		}
		// A schema-less server tool -- web search, the advisor -- declares only
		// a type, which is the one thing it has. Chat expresses every tool as
		// type "function" with the callable nested under `function`, so a type
		// that is neither of the callable spellings, on an object carrying no
		// `function`, can only be Anthropic's. Without this the commonest
		// server-tool request has no marker at all, reads as Chat, and is
		// refused by a codec that has no such tool -- while the selector, which
		// now encodes these by their type, is told they were there.
		var toolType string
		if err := json.Unmarshal(tool["type"], &toolType); err != nil {
			continue
		}
		if toolType != "" && toolType != "function" && toolType != "custom" {
			return llmprotocol.AnthropicMessagesV1, true
		}
	}
	return "", false
}

func raylineRoutesBodyWarnings(
	envelope map[string]json.RawMessage,
	encodesToolNames bool,
) []string {
	warnings := []string{}
	if !declaresTools(envelope["tools"]) {
		// Presence is not the question. Clients commonly serialise an absent
		// tool collection as `"tools": []`, and a warning on that claims the
		// route was shaped by tools the projection never saw -- in the
		// encoding case, that names influenced a decision they took no part
		// in. Both spellings of "no tools" have nothing to explain.
		return warnings
	}
	// What the cell did with the tools, not what it used to do with them. On
	// a cell that encodes names, the old warning was simply false: the names
	// reached the selector and can change the arm it picks, so telling a
	// caller their tools did not influence the route would misdescribe the
	// decision they are being handed.
	if encodesToolNames {
		warnings = append(warnings, "tool_schemas_not_encoded: tool names influenced this route; their descriptions and schemas were not encoded")
		return warnings
	}
	warnings = append(warnings, "tools_not_encoded: tools were dropped before encoding and did not influence this route")
	return warnings
}

// declaresTools reports whether the body actually offers the model a tool, as
// opposed to carrying the member with nothing in it.
func declaresTools(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil {
		// Not a list. Malformed rather than empty, and the codec will say so
		// in a moment -- warning about it here would explain a route that is
		// not going to exist.
		return false
	}
	return len(tools) > 0
}

// wireFormatOf reports the format this request was read as, for an error
// message that names it. Empty before the body phase has run.
func wireFormatOf(ctx *RequestContext) llmprotocol.WireFormat {
	if ctx == nil || ctx.SourceFormat == "" {
		return llmprotocol.AnthropicMessagesV1
	}
	return ctx.SourceFormat
}

func raylineRoutesHeaderWarnings(
	ctx *RequestContext,
	settings *config.RaylineARCRoutesAPIConfig,
) []string {
	warnings := []string{}
	if raylineRoutesHeader(ctx, raylineRoutesCheckpointHeader) != "" {
		warnings = append(warnings, "checkpoint_not_pinned: this cell serves one checkpoint and the requested pin was ignored")
	}
	// A caller who sends a conversation id and gets a plausible route back has
	// no way to see that continuity was never applied. Every turn would look
	// like a first turn, and the only symptom is routing that is quietly worse
	// than it should be.
	if settings != nil && !settings.EpisodeWrites &&
		raylineRoutesHeader(ctx, raylineRoutesSessionHeader) != "" {
		warnings = append(warnings, "episode_not_tracked: this cell keeps no episodes for route lookups, so the conversation did not influence this route")
	}
	// A branch names a lane inside a conversation, so it means nothing
	// without the conversation. Sent alone it is silently dropped and the
	// lookup becomes ephemeral, which looks identical to a correct stateless
	// call: the caller gets a plausible route and loses every branch's
	// continuity and its retained encoder prefix with no symptom to read.
	if raylineRoutesHeader(ctx, raylineRoutesBranchHeader) != "" &&
		raylineRoutesHeader(ctx, raylineRoutesSessionHeader) == "" {
		warnings = append(warnings, "branch_ignored: "+raylineRoutesBranchHeader+" names a lane inside a conversation and was dropped because "+raylineRoutesSessionHeader+" was absent")
	}
	return warnings
}

// raylineRoutesEpisodeIdentity composes the episode this decision belongs to.
//
// The branch is folded into the identity rather than carried beside it,
// matching how the gateway already composes userId:conversationId[:branch]:
// two concurrent subagents on one conversation are two trajectories, and
// scoring them as one would make each read as the other's previous turn.
// An absent session yields an empty identity, which makes the lookup
// ephemeral. So does a cell with episode_writes off, whatever the caller
// sent.
func raylineRoutesEpisodeIdentity(ctx *RequestContext) string {
	session := raylineRoutesHeader(ctx, raylineRoutesSessionHeader)
	if session == "" {
		return ""
	}
	// The session is escaped on BOTH paths, not just the one that appends a
	// branch. Escaping only the joined form leaves the collision it was meant
	// to close: session "a:b" alone still encodes to "a:b", which is exactly
	// what session "a" with branch "b" produces. The value is hashed
	// downstream, so this has to be injective, not readable.
	encoded := url.QueryEscape(session)
	branch := raylineRoutesHeader(ctx, raylineRoutesBranchHeader)
	if branch == "" {
		return encoded
	}
	return encoded + ":" + url.QueryEscape(branch)
}

// raylineRoutesRouteID adopts the caller's id when it sent one, so a single id
// spans their record and ours, and mints one otherwise.
func raylineRoutesRouteID(ctx *RequestContext) string {
	if adopted := raylineRoutesHeader(ctx, raylineRoutesRouteIDHeader); adopted != "" {
		return adopted
	}
	// Half a UUID, not a quarter. Eight hex characters is 32 bits, where a
	// birthday collision becomes likely in the tens of thousands of lookups --
	// well inside one busy day, and this id is the join key a support lookup
	// and a later settle call both rely on being unique.
	return raylineRouteIDPrefix + strings.ReplaceAll(uuid.NewString(), "-", "")[:16]
}

func raylineRoutesHeader(ctx *RequestContext, name string) string {
	if ctx == nil || ctx.Headers == nil {
		return ""
	}
	return strings.TrimSpace(ctx.Headers[name])
}

// raylineRoutesConsult composes the consult this endpoint sends the selector.
//
// It is a function rather than a literal at the call site so the surface can
// be asserted without standing up the whole selection stack. That mattered:
// the field was once added to the request type and to this endpoint's
// documentation without ever being set, which is indistinguishable from the
// bug it exists to prevent.
func raylineRoutesConsult(
	body []byte,
	routeID string,
	episode string,
	ephemeral bool,
	wireFormat llmprotocol.WireFormat,
) routerruntime.RouteDecisionRequest {
	return routerruntime.RouteDecisionRequest{
		Body:       body,
		DecisionID: routeID,
		SessionID:  episode,
		Ephemeral:  ephemeral,
		WireFormat: wireFormat,
		// Named, not defaulted. The two consult surfaces share one
		// RouteDecision method and disambiguate a multi-decision deployment
		// differently; an unset surface reads as the legacy consult and
		// resolves a different decision than this endpoint's own routes_api
		// claim.
		Surface: routerruntime.RouteDecisionSurfaceRoutes,
	}
}

// raylineRoutesGuardBreach answers a streamed body that broke an accumulation
// guard, in this endpoint's own envelope.
//
// It exists because the generic handling -- return the error, close the
// ExtProc stream, let Envoy decide -- resolves to "forward upstream" under
// failure_mode_allow, and forwarding is the one outcome this endpoint is
// defined by never producing.
func (r *OpenAIRouter) raylineRoutesGuardBreach(
	ctx *RequestContext,
	err error,
) *ext_proc.ProcessingResponse {
	if errors.Is(err, ErrStreamedBodyTimeout) {
		return r.createRaylineRoutesError(
			ctx,
			504,
			"timeout_error",
			"route lookup body did not arrive within the accumulation deadline",
		)
	}
	return r.createRaylineRoutesError(
		ctx,
		413,
		"invalid_request_error",
		"route lookup body exceeds the accumulation limit",
	)
}

// raylineRoutesContendedRetrySeconds is how long a contended caller is told to
// wait. One second, because the thing it collided with is one turn's episode
// lease or one encoder slot, both of which clear in about that time. A longer
// delay would make a healthy router look unavailable.
const raylineRoutesContendedRetrySeconds = 1

// raylineRoutesRetryAfter says when to come back.
//
// A 429 that carries no Retry-After leaves an automated client with nothing
// to honour, and the default behaviour is then an immediate retry -- onto the
// same lease or the same encoder queue that produced the contention, which
// amplifies exactly what the refusal was trying to shed.
func raylineRoutesRetryAfter(
	response *ext_proc.ProcessingResponse,
	seconds int,
) *ext_proc.ProcessingResponse {
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return response
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	immediate.Headers.SetHeaders = append(
		immediate.Headers.SetHeaders,
		&core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "retry-after",
				RawValue: []byte(strconv.Itoa(seconds)),
			},
		},
	)
	return response
}

// createRaylineRoutesError answers in the Anthropic error envelope rather than
// this router's own, because a caller of this endpoint is already parsing that
// envelope from the endpoint it would otherwise have called. One error path
// for both is the point of taking the same body.
func (r *OpenAIRouter) createRaylineRoutesError(
	ctx *RequestContext,
	statusCode int,
	errorType string,
	message string,
) *ext_proc.ProcessingResponse {
	response := r.createJSONResponse(statusCode, map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    errorType,
			"message": message,
		},
	})
	// Claim the body before the protocol contract re-encodes it. Every
	// immediate response at 400 or above is otherwise rewritten into the
	// source format's error shape, which for this path resolves to Chat --
	// so the envelope this endpoint documents, and its specific message,
	// would never reach a caller. Marking it encoded is how a producer says
	// the body is already in its final wire form.
	if ctx != nil {
		ctx.ImmediateResponseEncoded = true
	}
	return response
}
