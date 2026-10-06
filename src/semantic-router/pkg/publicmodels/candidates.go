package publicmodels

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// RoutingCandidate is one model a virtual id may resolve to, with its
// card's facts: a decision's model refs, a route action's destination, and
// the models an algorithm executes beside its refs.
//
// A virtual id is the only thing a caller may send, and what stands behind
// it is otherwise invisible: a gateway that fronts the router as one model
// had to hard-code that model's context window, output limit, modalities,
// tool support and price. Those are per-candidate facts the router already
// holds on its model cards, so the listing states them here, under the alias
// they belong to, rather than listing the candidates as ids of their own.
//
// RoutingCandidate is one model the alias's decisions may resolve to and its
// card's facts.
//
// The limits are pointers: a card that declares neither reports null, not
// zero, because zero reads as a measured limit of nothing. Vision is the
// card's effective verdict, capable when the card is silent, and tools is a
// positive claim an unmarked card does not make.
type RoutingCandidate struct {
	// Model is the name selection resolves to: the model ref's lora_name
	// when it sets one, since that is the adapter the decision selects and
	// dispatches, and the ref's model otherwise. BaseModel names the card a
	// LoRA candidate is served under; absent on every other candidate.
	// ProviderModel is the id the candidate dispatches as, when a backend
	// renames it.
	Model         string            `json:"model"`
	BaseModel     string            `json:"base_model,omitempty"`
	ProviderModel string            `json:"provider_model,omitempty"`
	Provider      string            `json:"provider,omitempty"`
	Thinking      CandidateThinking `json:"thinking"`
	Vision        bool              `json:"vision"`
	Tools         bool              `json:"tools"`
	// ContextWindow is the card's context_window_size, which selection
	// enforces: a plain decision drops a candidate whose window the request
	// exceeds, and a rayline_arc decision masks the arm where it stands.
	ContextWindow *int `json:"context_window"`
	// MaxOutputTokens is the limit the router dispatches when a request
	// states none of its own: the card's max_output_tokens, capped by the
	// declaring decision's request_params.max_tokens_limit, and across
	// decisions that declare the same candidate the smallest, since the
	// alias cannot choose the decision.
	MaxOutputTokens *int              `json:"max_output_tokens"`
	Pricing         *CandidatePricing `json:"pricing"`
	// Disabled is the card's out-of-service flag as selection enforces it:
	// a rayline_arc decision masks a disabled arm, so its candidate says so;
	// no other decision type reads the flag, its model still serves, and
	// the candidate says false rather than advertise a verdict nothing
	// honours.
	Disabled bool `json:"disabled"`
}

// CandidateThinking spells the decision's use_reasoning for the candidate
// the way a route lookup spells the arm it chose, so the two compare.
type CandidateThinking struct {
	Mode string `json:"mode"`
}

// CandidatePricing is the card's rate card in the route lookup's spelling.
// The cache-write rate is a pointer because the card's is: a card that prices
// no cache write reports null rather than a free one.
type CandidatePricing struct {
	Currency          string   `json:"currency,omitempty"`
	InputPerMTok      float64  `json:"input_per_mtok"`
	OutputPerMTok     float64  `json:"output_per_mtok"`
	CacheReadPerMTok  float64  `json:"cache_read_per_mtok"`
	CacheWritePerMTok *float64 `json:"cache_write_per_mtok"`
}

// routingCandidatesOf lists a recipe's candidates, decision by decision in
// declared order, or nil when the recipe declares none, so an alias with
// nothing behind it carries no empty array. The router's default_model,
// which an unmatched request falls back to, is deliberately not listed: on
// an ARC cell the one decision matches every turn and the fallback is never
// reached, and a gateway deriving a document from this list would otherwise
// fold in a model no turn is routed to.
func routingCandidatesOf(cfg *config.RouterConfig, recipe *config.RoutingRecipe) []RoutingCandidate {
	if cfg == nil || recipe == nil {
		return nil
	}
	return routingCandidatesOfDecisions(cfg, recipe.Profile.Decisions)
}

// routingCandidatesForAlgorithm lists the candidates of every default-profile
// decision that runs one algorithm. A direct Looper alias (ReMoM, Fusion,
// Flow) is served by exactly those decisions, filtered by type rather than
// by recipe, the way decisionCandidatesForRequestModel scopes a request to
// such an alias.
func routingCandidatesForAlgorithm(cfg *config.RouterConfig, algorithm string) []RoutingCandidate {
	if cfg == nil {
		return nil
	}
	var decisions []config.Decision
	for _, decision := range cfg.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == algorithm {
			decisions = append(decisions, decision)
		}
	}
	return routingCandidatesOfDecisions(cfg, decisions)
}

// routingCandidatesOfDecisions lists each distinct candidate once, in the
// order the decisions first declare it. A model two decisions declare is one
// candidate; a model one decision runs with reasoning on and another with it
// off is two, because the thinking mode is a property of the ref and the
// listing reports it. Which decision picks which candidate is a routing
// detail the route lookup reports per turn, not a fact about the model.
func routingCandidatesOfDecisions(cfg *config.RouterConfig, decisions []config.Decision) []RoutingCandidate {
	var candidates []RoutingCandidate
	for index := range decisions {
		decision := &decisions[index]
		// Only a rayline_arc decision enforces a card's disabled flag; see
		// RoutingCandidate.Disabled.
		enforcesDisabled := decision.Algorithm != nil && decision.Algorithm.Type == config.RaylineARCAlgorithmType
		// The decision's request_params.max_tokens_limit caps the output
		// limit the router dispatches; see RoutingCandidate.MaxOutputTokens.
		outputCap := decisionOutputCap(decision)
		// A route action resolves straight to its destination, ahead of
		// the refs, which it falls back to only when the destination cannot
		// hold the request -- and then dispatches the ref's model, not an
		// adapter the ref names, so the refs are described as the model.
		// The algorithm never runs behind a route action, so the models it
		// alone would execute are not listed.
		routeAction := decision.Action != nil && decision.Action.Type == config.DecisionActionRoute
		if routeAction {
			if destination := strings.TrimSpace(decision.Action.Destination); destination != "" {
				candidates = appendDistinctCandidate(cfg, candidates, config.ModelRef{Model: destination}, enforcesDisabled, outputCap)
			}
		}
		// A Fusion decision that names its analysis models runs those, not
		// its refs (resolveFusionExecutionConfig), so the refs are listed
		// only when Fusion falls back to them.
		fusionOverridesRefs := decision.Algorithm != nil && decision.Algorithm.Fusion != nil && len(decision.Algorithm.Fusion.AnalysisModels) > 0
		if !fusionOverridesRefs {
			for _, modelRef := range decision.ModelRefs {
				if routeAction {
					modelRef.LoRAName = ""
				}
				candidates = appendDistinctCandidate(cfg, candidates, modelRef, enforcesDisabled, outputCap)
			}
		}
		if routeAction {
			continue
		}
		// The models an algorithm executes beside its refs: a Fusion judge,
		// a ReMoM synthesis model, a workflow's planner. They answer the
		// request, so a document derived from this list has to hold them.
		// Each runs under the reasoning the Looper resolves for it: its
		// ref's when a ref names it, else its card's reasoning family.
		for _, model := range decision.Algorithm.ExplicitModels() {
			if model = strings.TrimSpace(model); model != "" {
				candidates = appendDistinctCandidate(cfg, candidates, decisionRefFor(cfg, decision, model), enforcesDisabled, outputCap)
			}
		}
	}
	return candidates
}

// decisionOutputCap is the decision's request_params.max_tokens_limit, or
// zero when it sets none.
func decisionOutputCap(decision *config.Decision) int {
	params := decision.GetRequestParamsConfig()
	if params == nil || params.MaxTokensLimit == nil || *params.MaxTokensLimit <= 0 {
		return 0
	}
	return *params.MaxTokensLimit
}

// decisionRefFor is the ref an algorithm-owned model runs under, resolved
// the way the Looper runtime resolves the model's reasoning
// (getReasoningInfoFromDecision): the first ref naming that model, adapter or
// not, and failing one, the model's card, which turns reasoning on when it
// declares a reasoning family. The model is executed as itself, so an
// adapter name is not carried.
func decisionRefFor(cfg *config.RouterConfig, decision *config.Decision, model string) config.ModelRef {
	for _, modelRef := range decision.ModelRefs {
		if strings.TrimSpace(modelRef.Model) == model {
			return config.ModelRef{Model: model, ModelReasoningControl: modelRef.ModelReasoningControl}
		}
	}
	modelRef := config.ModelRef{Model: model}
	if params, known := cfg.ModelConfig[model]; known && params.ReasoningFamily != "" {
		on := true
		modelRef.UseReasoning = &on
	}
	return modelRef
}

// appendDistinctCandidate lists a ref unless a candidate of the same model,
// adapter and thinking mode is listed already. A candidate stays disabled
// only while every decision that declares it enforces the flag: one that
// still dispatches the model makes it reachable, whatever the declaration
// order.
func appendDistinctCandidate(cfg *config.RouterConfig, candidates []RoutingCandidate, modelRef config.ModelRef, enforcesDisabled bool, outputCap int) []RoutingCandidate {
	candidate := routingCandidateOf(cfg, modelRef, enforcesDisabled, outputCap)
	for index := range candidates {
		listed := &candidates[index]
		if listed.Model == candidate.Model && listed.BaseModel == candidate.BaseModel && listed.Thinking == candidate.Thinking {
			listed.Disabled = listed.Disabled && candidate.Disabled
			// The smallest output limit any declaring decision dispatches
			// is the one every turn is guaranteed; the alias cannot choose
			// the decision.
			if candidate.MaxOutputTokens != nil && (listed.MaxOutputTokens == nil || *candidate.MaxOutputTokens < *listed.MaxOutputTokens) {
				listed.MaxOutputTokens = candidate.MaxOutputTokens
			}
			return candidates
		}
	}
	return append(candidates, candidate)
}

func routingCandidateOf(cfg *config.RouterConfig, modelRef config.ModelRef, enforcesDisabled bool, outputCap int) RoutingCandidate {
	model := strings.TrimSpace(modelRef.Model)
	candidate := RoutingCandidate{
		Model:    model,
		Thinking: CandidateThinking{Mode: thinkingModeOf(modelRef)},
		// An unknown card claims nothing and refuses nothing, exactly as
		// selection reads it: SupportsVision on a zero card is true, and
		// SupportsCapability is false.
		Vision: true,
	}
	// A LoRA ref selects the adapter, not the base: the adapter name is
	// what the decision picks, what dispatch sends and what the backend
	// answers with. Its facts resolve field by field the way the runtime
	// resolves them, not card by card: ARC gating reads vision, tools and
	// disabled off the ref's model, the base card, whatever the adapter
	// declares, and so does the context eligibility filter
	// (modelNameExceedsContextWindow); the output limit is the adapter's
	// own when it states one and the base's otherwise
	// (GetModelMaxOutputTokens); pricing likewise.
	if lora := strings.TrimSpace(modelRef.LoRAName); lora != "" {
		candidate.Model, candidate.BaseModel = lora, model
	}
	if params, known := cfg.ModelConfig[model]; known {
		candidate.Vision = params.SupportsVision()
		candidate.Tools = params.SupportsCapability(llmprotocol.RoutingCapabilityTools)
		candidate.ContextWindow = positiveIntPointer(params.ContextWindowSize)
		candidate.Disabled = enforcesDisabled && params.IsDisabled()
	}
	candidate.MaxOutputTokens = positiveIntPointer(cfg.GetModelMaxOutputTokens(candidate.Model))
	// The decision's max_tokens_limit caps what the router dispatches for a
	// caller that states no limit (planDispatchOutputBound); a card that
	// declares no limit stays unbounded on Chat, cap or not, so it stays
	// null here.
	if candidate.MaxOutputTokens != nil && outputCap > 0 && outputCap < *candidate.MaxOutputTokens {
		candidate.MaxOutputTokens = positiveIntPointer(outputCap)
	}
	// The primary backend names the provider and the id the candidate
	// dispatches as: the highest-weight endpoint, which is the one dispatch
	// takes, not the first listed. A renamed model is reported only when
	// the backend actually renames it, so a passthrough name is not
	// repeated. The backend resolves a LoRA name through its base card; the
	// id is looked up under the name dispatch sends, as dispatch does.
	if _, endpointName, ok, err := cfg.ResolvePrimaryBackendForModel(candidate.Model); ok && err == nil {
		if endpoint, found := cfg.GetEndpointByName(endpointName); found {
			candidate.Provider = endpoint.Type
			// An endpoint that declares no type is a vLLM endpoint: that is
			// the type dispatch resolves its model id under
			// (ResolveExternalModelID), so the listing says the same.
			if candidate.Provider == "" {
				candidate.Provider = "vllm"
			}
		}
		if providerModel := cfg.ResolveExternalModelID(candidate.Model, endpointName); providerModel != candidate.Model {
			candidate.ProviderModel = providerModel
		}
	}
	if pricing, ok := cfg.GetFullModelPricing(candidate.Model); ok {
		candidate.Pricing = &CandidatePricing{
			Currency:          pricing.Currency,
			InputPerMTok:      pricing.PromptPer1M,
			OutputPerMTok:     pricing.CompletionPer1M,
			CacheReadPerMTok:  pricing.CachedInputPer1M,
			CacheWritePerMTok: pricing.CacheWritePer1M,
		}
	}
	return candidate
}

func thinkingModeOf(modelRef config.ModelRef) string {
	if modelRef.UseReasoning != nil && *modelRef.UseReasoning {
		return "on"
	}
	return "off"
}

func positiveIntPointer(value int) *int {
	if value <= 0 {
		return nil
	}
	return &value
}
