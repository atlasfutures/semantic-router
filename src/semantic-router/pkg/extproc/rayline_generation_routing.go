package extproc

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A generation turn's routing block (pathfinder#5664, GenerationRouting):
// routing.{value_usd, temperature} on /v1/messages, /v1/chat/completions and
// /v1/responses, read on a policy-service cell whose policy_service sets
// generation_routing. See config.RaylineARCGenerationRoutingConfig.

// raylineGenerationRouting is a validated generation routing block. A member
// the caller did not send is nil.
type raylineGenerationRouting struct {
	valueUSD *float64
	// costWeight is the turn's own cost weight on the configurable cell:
	// 1 / valueUSD. Nil on a tier cell, which keeps its own.
	costWeight  *float64
	temperature *float64
}

// raylineGenerationRoutingPolicy is the one policy-service ARC decision
// whose generation_routing is set, or nil. A cell serves one ARC decision;
// more than one with the block reads none, since the turn's decision is not
// yet known here and guessing would apply another cell's V.
func (r *OpenAIRouter) raylineGenerationRoutingPolicy() *config.RaylineARCPolicyServiceConfig {
	if r == nil || r.Config == nil {
		return nil
	}
	var found *config.RaylineARCPolicyServiceConfig
	for index := range r.Config.Decisions {
		algorithm := r.Config.Decisions[index].Algorithm
		if !raylineARCSelection(algorithm) || algorithm.RaylineARC == nil ||
			algorithm.RaylineARC.PolicyService == nil || algorithm.RaylineARC.PolicyService.GenerationRouting == nil {
			continue
		}
		if found != nil {
			return nil
		}
		found = algorithm.RaylineARC.PolicyService
	}
	return found
}

// applyRaylineGenerationRouting reads and removes a generation turn's
// routing block, before the codec, the encoder or the provider sees the body.
// It returns the body to serve, or a refusal in the contract's codes.
func (r *OpenAIRouter) applyRaylineGenerationRouting(
	body []byte,
	ctx *RequestContext,
) ([]byte, *ext_proc.ProcessingResponse) {
	policy := r.raylineGenerationRoutingPolicy()
	if policy == nil {
		return body, nil
	}
	routing, stripped, refusal := readRaylineGenerationRouting(body, policy)
	if refusal != nil {
		return nil, r.raylineGenerationRoutingRefusal(ctx, refusal)
	}
	if routing == nil {
		return body, nil
	}
	ctx.RaylineGenerationRouting = routing
	// Turns that differ in their V or T are different turns.
	ctx.RaylineRoutesControlsKey = routing.key()
	return stripped, nil
}

func (routing *raylineGenerationRouting) key() string {
	format := func(value *float64) string {
		if value == nil {
			return "-"
		}
		return strconv.FormatFloat(*value, 'g', -1, 64)
	}
	return "generation:v=" + format(routing.valueUSD) + ":t=" + format(routing.temperature)
}

// readRaylineGenerationRouting validates the block in the contract's order:
// the block's shape, prices or exclusive (a Decisions member), unknown
// members, then value_usd and temperature. The body member is the one
// transport (memex-desktop#7539); no header carries V or T. A body without a
// block returns a nil routing and the body unchanged.
func readRaylineGenerationRouting(
	body []byte,
	policy *config.RaylineARCPolicyServiceConfig,
) (*raylineGenerationRouting, []byte, *routesContractError) {
	var envelope map[string]json.RawMessage
	if json.Unmarshal(body, &envelope) != nil {
		// Not a JSON object: the codec refuses it in its own words.
		return nil, body, nil
	}
	raw, sent := envelope["routing"]
	if !sent {
		return nil, body, nil
	}
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil || members == nil {
		return nil, nil, routesInvalid("routing_invalid", "routing must be an object.")
	}
	// routing.execution is a Decisions member (pathfinder #5874): a turn
	// this router executes keeps its own execution state.
	if _, execution := members["execution"]; execution {
		return nil, nil, routesInvalid("execution_not_supported",
			"routing.execution is accepted on /v1/routes only.")
	}
	_, prices := members["prices"]
	_, exclusive := members["exclusive"]
	if prices || exclusive {
		return nil, nil, routesInvalid("prices_not_supported",
			"routing.prices and routing.exclusive are accepted on /v1/routes only.")
	}
	var unknown []string
	for name := range members {
		if name != "value_usd" && name != "temperature" {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, nil, routesInvalid("routing_unknown_member", "routing has unsupported member(s): "+strings.Join(unknown, ", ")+".")
	}
	routing := &raylineGenerationRouting{}
	if raw, ok := members["value_usd"]; ok {
		value, finite := routesFiniteNumber(raw)
		if !finite {
			return nil, nil, routesInvalid("value_invalid", "routing.value_usd must be a finite number of USD.")
		}
		if value < routesValueUSDMin || value > routesValueUSDMax {
			return nil, nil, routesInvalid("value_out_of_bounds", fmt.Sprintf(
				"routing.value_usd must lie in [%g, %g]; it is never clamped.", routesValueUSDMin, routesValueUSDMax))
		}
		if !policy.GenerationRouting.CallerValue {
			// A tier: its V is ours. The caller may restate it, never move it.
			fixed, ok := raylineGenerationFixedValue(policy)
			if !ok || math.Abs(value-fixed) > 1e-9*fixed {
				return nil, nil, routesInvalid("tier_value_immutable", raylineGenerationTierMessage(fixed, ok))
			}
		} else {
			weight := 1 / value
			routing.costWeight = &weight
		}
		routing.valueUSD = &value
	}
	if raw, ok := members["temperature"]; ok {
		value, finite := routesFiniteNumber(raw)
		if !finite || value < 0 {
			return nil, nil, routesInvalid("temperature_invalid", "routing.temperature must be a finite number >= 0 "+
				"(a softmax temperature over normalized scores; 0 is greedy).")
		}
		routing.temperature = &value
	}
	stripped, refusal := stripRoutingMember(body)
	if refusal != nil {
		return nil, nil, refusal
	}
	return routing, stripped, nil
}

// raylineGenerationFixedValue is a tier cell's own V, 1 / its pinned
// decisions_v2.cost_weight.
func raylineGenerationFixedValue(policy *config.RaylineARCPolicyServiceConfig) (float64, bool) {
	if policy.DecisionsV2 == nil || policy.DecisionsV2.CostWeight == nil || *policy.DecisionsV2.CostWeight <= 0 {
		return 0, false
	}
	return 1 / *policy.DecisionsV2.CostWeight, true
}

func raylineGenerationTierMessage(fixed float64, known bool) string {
	if !known {
		return "This ARC model decides at its own fixed value_usd; use arc-1.0 for a custom value."
	}
	return fmt.Sprintf("This ARC model decides at a fixed value_usd of %g; use arc-1.0 for a custom value.", fixed)
}

// raylineGenerationRoutingRefusal answers a refused block as a 400 in the
// client's protocol, with the contract's code. Chat and Responses carry it as
// error.code through the protocol encoder. The Messages encoder writes no
// code, so a Messages refusal is written here, in the Anthropic envelope the
// contract's ErrorEnvelope extends with error.code, as /v1/routes does.
func (r *OpenAIRouter) raylineGenerationRoutingRefusal(
	ctx *RequestContext,
	refusal *routesContractError,
) *ext_proc.ProcessingResponse {
	if ctx != nil && ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 {
		return r.createRaylineRoutesCodedError(ctx, refusal)
	}
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, refusal.code, refusal.message, nil)
	}
	return r.createErrorResponse(http.StatusBadRequest, refusal.message)
}

// raylineGenerationSignalHeaders adds the block's temperature to the trusted
// turn signals as the caller T, on a cell that trusts its gateway: the
// hosted exploration rules (eligibility, caller_temperature, the service's
// clamp) then decide whether it is sent.
func raylineGenerationSignalHeaders(
	headers map[string]string,
	routing *raylineGenerationRouting,
	policy *config.RaylineARCPolicyServiceConfig,
) map[string]string {
	if routing == nil || routing.temperature == nil || policy == nil || !policy.TrustTurnSignalHeaders {
		return headers
	}
	with := make(map[string]string, len(headers)+1)
	for name, value := range headers {
		with[name] = value
	}
	with[raylineARCCallerTemperatureSignal] = strconv.FormatFloat(*routing.temperature, 'g', -1, 64)
	return with
}

// raylineGenerationRoutingStatus is which routing members the turn applied
// and which it did not: value_usd applies only when the decision was taken
// and echoed at the turn's cost weight (a capacity or media hold decides
// nothing, so it is unapplied there; on a tier cell, which restates its own
// V, only a decision applies it too); temperature applies only when the
// decision carries a caller-sourced exploration record.
func raylineGenerationRoutingStatus(ctx *RequestContext) (applied []string, unapplied []string) {
	if ctx == nil || ctx.RaylineGenerationRouting == nil || ctx.VSRRaylineARC == nil {
		return nil, nil
	}
	routing := ctx.RaylineGenerationRouting
	if routing.valueUSD != nil {
		trace := ctx.VSRRaylineARC
		decided := trace.TurnCostWeightApplied ||
			(routing.costWeight == nil && trace.PolicyActionID != "" && !raylineARCHoldOfferKind(trace.OfferKind))
		if decided {
			applied = append(applied, "value_usd")
		} else {
			unapplied = append(unapplied, "value_usd")
		}
	}
	if routing.temperature != nil {
		if raylineGenerationTemperatureApplied(ctx.VSRRaylineARC) {
			applied = append(applied, "temperature")
		} else {
			unapplied = append(unapplied, "temperature")
		}
	}
	return applied, unapplied
}

func raylineGenerationTemperatureApplied(trace *selection.RaylineARCTrace) bool {
	record := trace.PolicyExploration
	return trace.PolicyExplorationSent && record != nil && record.Source != nil && *record.Source == "caller"
}

// addRaylineGenerationRoutingHeaders publishes the routing status:
// x-vsr-routing-applied and x-vsr-routing-unapplied.
func (builder *responseHeaderMutationBuilder) addRaylineGenerationRoutingHeaders(ctx *RequestContext) {
	applied, unapplied := raylineGenerationRoutingStatus(ctx)
	builder.addString(headers.VSRRoutingApplied, strings.Join(applied, ","))
	builder.addString(headers.VSRRoutingUnapplied, strings.Join(unapplied, ","))
}

// raylineARCHoldOfferKind reports a turn served by a hold (capacity or media):
// the policy decided nothing on it.
func raylineARCHoldOfferKind(kind string) bool {
	for _, hold := range policyHoldOfferKinds {
		if kind == hold {
			return true
		}
	}
	return false
}
