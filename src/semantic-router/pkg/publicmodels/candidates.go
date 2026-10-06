package publicmodels

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// RoutingCandidate is one arm of a Rayline ARC decision a virtual id resolves
// through, with its card's facts.
//
// A virtual id is the only thing a caller may send, and what stands behind
// it is otherwise invisible: a gateway that fronts an ARC cell as one model
// had to hard-code that model's context window, output limit, modalities,
// tool support and price. Those are per-arm facts the router already holds
// on its model cards, so the listing states them here, under the alias they
// belong to, rather than listing the arms as ids of their own.
//
// Only a rayline_arc decision is described. Its arm set is fixed and
// positional, every arm is a plain model ref, an arm a turn cannot use is
// masked rather than removed, and the decision takes no route action, no
// minimum and no algorithm-owned models, so the list is exactly what the
// artifact chooses among. The stock router's other algorithms resolve a
// request through paths the refs alone do not describe -- a route action's
// destination, a Fusion judge, a static workflow's plan, a fast response
// that calls no model -- and are not described here.
//
// The limits are pointers: a card that declares neither reports null, not
// zero, because zero reads as a measured limit of nothing. Vision is the
// card's effective verdict, which is capable when the card is silent, and
// tools is a positive claim an unmarked card does not make.
type RoutingCandidate struct {
	// Model is the arm's name, the model ref the decision declares and the
	// worker id the route lookup reports. ProviderModel is the id the arm
	// dispatches as, when its backend renames it.
	Model         string            `json:"model"`
	ProviderModel string            `json:"provider_model,omitempty"`
	Provider      string            `json:"provider,omitempty"`
	Thinking      CandidateThinking `json:"thinking"`
	Vision        bool              `json:"vision"`
	Tools         bool              `json:"tools"`
	// ContextWindow is the card's context_window_size, which the ARC
	// selector enforces as a mask: an arm the turn does not fit is excluded
	// where it stands.
	ContextWindow *int `json:"context_window"`
	// MaxOutputTokens is the limit the router dispatches when a request
	// states none of its own: the card's max_output_tokens, capped by the
	// decision's request_params.max_tokens_limit.
	MaxOutputTokens *int              `json:"max_output_tokens"`
	Pricing         *CandidatePricing `json:"pricing"`
	// Disabled is the card's out-of-service flag, which the ARC selector
	// enforces as a mask. A disabled arm is listed, so the arms keep their
	// artifact ordinals, and marked.
	Disabled bool `json:"disabled"`
}

// CandidateThinking spells the decision's use_reasoning for the arm the way
// a route lookup spells the arm it chose, so the two compare.
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

// routingCandidatesOf lists the arms of a recipe's rayline_arc decisions in
// artifact order, each arm once, or nil when the recipe has no such
// decision, so an alias with nothing to describe carries no empty array.
func routingCandidatesOf(cfg *config.RouterConfig, recipe *config.RoutingRecipe) []RoutingCandidate {
	if cfg == nil || recipe == nil {
		return nil
	}
	var candidates []RoutingCandidate
	for index := range recipe.Profile.Decisions {
		decision := &recipe.Profile.Decisions[index]
		if decision.Algorithm == nil || decision.Algorithm.Type != config.RaylineARCAlgorithmType {
			continue
		}
		outputCap := decisionOutputCap(decision)
		for _, modelRef := range decision.ModelRefs {
			candidate := routingCandidateOf(cfg, modelRef, outputCap)
			if listed(candidates, candidate) {
				continue
			}
			candidates = append(candidates, candidate)
		}
	}
	return candidates
}

// listed reports whether an arm of the same model and thinking mode is in
// the list already: two ARC decisions in one recipe declare the same arm set
// (readiness holds them to one artifact), so the second adds nothing.
func listed(candidates []RoutingCandidate, candidate RoutingCandidate) bool {
	for _, existing := range candidates {
		if existing.Model == candidate.Model && existing.Thinking == candidate.Thinking {
			return true
		}
	}
	return false
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

func routingCandidateOf(cfg *config.RouterConfig, modelRef config.ModelRef, outputCap int) RoutingCandidate {
	model := strings.TrimSpace(modelRef.Model)
	candidate := RoutingCandidate{
		Model:    model,
		Thinking: CandidateThinking{Mode: thinkingModeOf(modelRef)},
		// An unknown card claims nothing and refuses nothing, exactly as
		// selection reads it: SupportsVision on a zero card is true, and
		// SupportsCapability is false.
		Vision: true,
	}
	if params, known := cfg.ModelConfig[model]; known {
		candidate.Vision = params.SupportsVision()
		candidate.Tools = params.SupportsCapability(llmprotocol.RoutingCapabilityTools)
		candidate.ContextWindow = positiveIntPointer(params.ContextWindowSize)
		candidate.Disabled = params.IsDisabled()
	}
	candidate.MaxOutputTokens = positiveIntPointer(cfg.GetModelMaxOutputTokens(model))
	// The decision's max_tokens_limit caps what the router dispatches for a
	// caller that states no limit (planDispatchOutputBound); a card that
	// declares no limit stays unbounded on Chat, cap or not, so it stays
	// null here.
	if candidate.MaxOutputTokens != nil && outputCap > 0 && outputCap < *candidate.MaxOutputTokens {
		candidate.MaxOutputTokens = positiveIntPointer(outputCap)
	}
	// The primary backend names the provider and the id the arm dispatches
	// as: the highest-weight endpoint, which is the one dispatch takes, not
	// the first listed. A renamed model is reported only when the backend
	// actually renames it, so a passthrough name is not repeated. An
	// endpoint that declares no type is a vLLM endpoint, the type dispatch
	// resolves its model id under (ResolveExternalModelID).
	if _, endpointName, ok, err := cfg.ResolvePrimaryBackendForModel(model); ok && err == nil {
		if endpoint, found := cfg.GetEndpointByName(endpointName); found {
			candidate.Provider = endpoint.Type
			if candidate.Provider == "" {
				candidate.Provider = "vllm"
			}
		}
		if providerModel := cfg.ResolveExternalModelID(model, endpointName); providerModel != model {
			candidate.ProviderModel = providerModel
		}
	}
	if pricing, ok := cfg.GetFullModelPricing(model); ok {
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
