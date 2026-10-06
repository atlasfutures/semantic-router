package publicmodels

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// RoutingCandidate is one model a virtual id may resolve to, with its
// card's facts: a decision's model refs, a route action's destination, and
// the router's default_model, which an auto or entrypoint request resolves
// to when no decision matches.
//
// A virtual id is the only thing a caller may send, and what stands behind
// it is otherwise invisible: a gateway that fronts the router as one model
// had to hard-code that model's context window, output limit, modalities,
// tool support and price. Those are per-candidate facts the router already
// holds on its model cards, so the listing states them here, under the alias
// they belong to, rather than listing the candidates as ids of their own.
//
// RoutingCandidate is one model ref of a decision and its card's facts.
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
	// MaxOutputTokens is also the limit the router dispatches when a request
	// states none of its own.
	MaxOutputTokens *int              `json:"max_output_tokens"`
	Pricing         *CandidatePricing `json:"pricing"`
	Disabled        bool              `json:"disabled"`
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
// declared order, then the model an auto or entrypoint request resolves to
// when no decision matches or the matched one has no usable ref, which is
// the router's default_model. That fallback is reached whenever the router
// routes at all -- a recipe with no decisions of its own still resolves an
// unmatched request to it when another recipe declares some -- so it is
// listed whenever any recipe has a decision. Nil on a router with none, so
// an alias with nothing behind it carries no empty array.
func routingCandidatesOf(cfg *config.RouterConfig, recipe *config.RoutingRecipe) []RoutingCandidate {
	if cfg == nil || recipe == nil || !cfg.HasRoutingDecisions() {
		return nil
	}
	candidates := routingCandidatesOfDecisions(cfg, recipe.Profile.Decisions)
	if fallback := strings.TrimSpace(cfg.DefaultModel); fallback != "" {
		candidates = appendDistinctCandidate(cfg, candidates, config.ModelRef{Model: fallback})
	}
	return candidates
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
		// A route action resolves straight to its destination, ahead of
		// the refs, which it falls back to only when the destination cannot
		// hold the request. The destination is what the alias selects, so
		// it is listed first and the refs after it.
		if decision.Action != nil && decision.Action.Type == config.DecisionActionRoute {
			if destination := strings.TrimSpace(decision.Action.Destination); destination != "" {
				candidates = appendDistinctCandidate(cfg, candidates, config.ModelRef{Model: destination})
			}
		}
		for _, modelRef := range decision.ModelRefs {
			candidates = appendDistinctCandidate(cfg, candidates, modelRef)
		}
		// The models an algorithm executes beside its refs: a Fusion judge,
		// a ReMoM synthesis model, a workflow's planner. They answer the
		// request, so a document derived from this list has to hold them.
		for _, model := range decision.Algorithm.ExplicitModels() {
			if model = strings.TrimSpace(model); model != "" {
				candidates = appendDistinctCandidate(cfg, candidates, config.ModelRef{Model: model})
			}
		}
	}
	return candidates
}

// appendDistinctCandidate lists a ref unless a candidate of the same model,
// adapter and thinking mode is listed already.
func appendDistinctCandidate(cfg *config.RouterConfig, candidates []RoutingCandidate, modelRef config.ModelRef) []RoutingCandidate {
	candidate := routingCandidateOf(cfg, modelRef)
	for _, listed := range candidates {
		if listed.Model == candidate.Model && listed.BaseModel == candidate.BaseModel && listed.Thinking == candidate.Thinking {
			return candidates
		}
	}
	return append(candidates, candidate)
}

func routingCandidateOf(cfg *config.RouterConfig, modelRef config.ModelRef) RoutingCandidate {
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
		candidate.Disabled = params.IsDisabled()
	}
	candidate.MaxOutputTokens = positiveIntPointer(cfg.GetModelMaxOutputTokens(candidate.Model))
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
