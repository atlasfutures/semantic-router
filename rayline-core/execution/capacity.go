package execution

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"sort"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

const (
	CapacityContext            = "context_capacity"
	CapacityRuntimeUnavailable = "runtime_unavailable"

	// Allocation is stated as configured until a runtime adapter observes it.
	// llama.cpp's per-slot n_ctx is not yet observed, so nothing here is ever
	// labelled runtime_observed.
	capacitySourceConfigured = "configured"

	// A fact counted by the target's own tokenizer, versus one estimated from
	// the prepared request (capacity_estimate.go). The policy contract
	// (pathfinder arc_execution_v2.decide_inputs_v2) admits an estimated
	// fits:false as exclusion and reads an estimated fits:true as unknown.
	capacityProvenanceMeasured  = "local_measured"
	capacityProvenanceEstimated = "caller_estimated"

	// CodeContextExceedsAllDeployments refuses a turn that every admitted
	// deployment reports it cannot hold, before any runtime is contacted.
	CodeContextExceedsAllDeployments = "context_exceeds_all_deployments"
)

// CapacityFact is this exact request's verdict on one target action. Nil means
// unknown: never zero and never "fits".
type CapacityFact struct {
	Fits            bool    `json:"fits"`
	Reason          *string `json:"reason"`
	Required        *int64  `json:"required"`
	Available       *int64  `json:"available"`
	AvailableSource *string `json:"available_source"`
	Units           *string `json:"units"`
	Provenance      string  `json:"provenance"`
	// prompt is the prompt's own size within Required, for refusal messages.
	prompt int64
}

// capacityFact compares the exact prepared prompt plus the actual output limit
// and configured margin with the deployment's configured context, in the
// target's own tokenizer units. Windows are never compared across targets.
func capacityFact(deployment Deployment, prompt *cachecontext.MeasuredPrompt, outputLimit int64) *CapacityFact {
	if prompt == nil || deployment.ContextWindow <= 0 || prompt.Units() == "" {
		return nil
	}
	required := int64(prompt.PromptTokens()) + outputLimit + deployment.CapacityMarginTokens
	available := deployment.ContextWindow
	source, units := capacitySourceConfigured, prompt.Units()
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Units: &units, Provenance: capacityProvenanceMeasured, prompt: int64(prompt.PromptTokens())}
	if !fact.Fits {
		reason := CapacityContext
		fact.Reason = &reason
	}
	return fact
}

// runtimeContextRejection is the runtime's own refusal before any output. It
// is admitted as fits:false whatever the allocation turns out to be.
func runtimeContextRejection(deployment Deployment, previous *CapacityFact) *CapacityFact {
	reason := CapacityContext
	fact := &CapacityFact{Fits: false, Reason: &reason, Provenance: "provider_reported"}
	if previous != nil {
		fact.Required, fact.Units, fact.prompt = previous.Required, previous.Units, previous.prompt
	} else if deployment.TokenUnits != "" {
		units := deployment.TokenUnits
		fact.Units = &units
	}
	return fact
}

// measureCapacity prepares each target exactly as dispatch would. A target
// with a qualified counter is counted with its own runtime tokenizer
// (local_measured); a local target without one is sized by
// localPromptSize; a cloud target without one stays unknown. Nothing is
// committed.
func (server *Server) measureCapacity(ctx context.Context, body []byte, format llmprotocol.WireFormat, state *branchState) (map[string]*CapacityFact, map[string]*cachecontext.MeasuredPrompt, error) {
	facts := make(map[string]*CapacityFact)
	measured := make(map[string]*cachecontext.MeasuredPrompt)
	ids := make([]string, 0, len(server.actions))
	for id := range server.actions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	type promptSize struct {
		prompt            int64
		provenance, units string
	}
	sizes := map[string]promptSize{}
	for _, id := range ids {
		selected := server.actions[id]
		counted := server.counters[selected.deployment.ID] != nil
		// A cloud target without a counter stays unknown: its provider
		// refuses an oversized prompt itself, and an estimate would only
		// exclude it early.
		if selected.deployment.ID == "" || !counted && selected.deployment.Kind != "local" {
			continue
		}
		request, envelope, _, err := server.engine.DecodeRequestForMutation(format, body)
		if err != nil {
			return nil, nil, err
		}
		prepared, err := server.prepare(request, envelope, selected, state)
		if err != nil {
			// This action cannot carry the request at all (for example an
			// output allowance above its limit); that is not a context fact.
			slog.Warn("capacity preparation failed", "action_id", id, "deployment", selected.deployment.ID, "error", err.Error())
			continue
		}
		if !counted {
			if selected.deployment.ContextWindow <= 0 {
				continue
			}
			// Actions differing only in a control often prepare the same
			// bytes: size each distinct body once per deployment.
			key := selected.deployment.ID + "\x00" + fingerprint(prepared.body)
			size, ok := sizes[key]
			if !ok {
				size.prompt, size.provenance, size.units = localPromptSize(selected.deployment, prepared.body)
				sizes[key] = size
			}
			facts[id] = sizedCapacityFact(selected.deployment, size.prompt, size.provenance, size.units, prepared.outputLimit)
			continue
		}
		prompt, err := server.measurePreparedPrompt(ctx, selected, prepared.body, state, nil)
		if err != nil {
			// A runtime that cannot count is a fact about that target only:
			// it is unavailable, never a reason to refuse the whole turn.
			slog.Warn("capacity measurement failed", "action_id", id, "deployment", selected.deployment.ID, "error", err.Error())
			reason := CapacityRuntimeUnavailable
			facts[id] = &CapacityFact{Fits: false, Reason: &reason, Provenance: capacityProvenanceMeasured}
			continue
		}
		measured[id] = prompt
		facts[id] = capacityFact(selected.deployment, prompt, prepared.outputLimit)
	}
	server.relaxEstimatedShortfalls(facts)
	return facts, measured, nil
}

// relaxEstimatedShortfalls applies shortfallStands to the facts sent to
// Decisions: when every executable target falls short, counting every
// shortfall, an estimated one on a context_overflow "refuses" deployment is
// sent as unknown for all that deployment's actions, so the turn dispatches
// there and the runtime's own context refusal decides. While some target is
// not short, every estimate still excludes its target, so the turn hands off
// rather than probing a runtime.
func (server *Server) relaxEstimatedShortfalls(facts map[string]*CapacityFact) {
	if !server.everyTarget(facts, isContextShortfall) {
		return
	}
	for id, fact := range facts {
		if server.yieldsToRuntime(id, fact) {
			delete(facts, id)
		}
	}
}

func isContextShortfall(fact *CapacityFact) bool {
	return fact != nil && !fact.Fits && fact.Reason != nil && *fact.Reason == CapacityContext
}

// yieldsToRuntime reports an estimated shortfall on a deployment whose
// runtime refuses an over-window prompt itself (context_overflow "refuses"):
// it never settles that every target falls short. A truncating runtime's
// estimate (the default) stands.
func (server *Server) yieldsToRuntime(id string, fact *CapacityFact) bool {
	return isContextShortfall(fact) && fact.Provenance == capacityProvenanceEstimated && server.actions[id].deployment.ContextOverflow == ContextOverflowRefuses
}

// everyTarget reports whether every executable action's fact satisfies
// holds; false when there is none.
func (server *Server) everyTarget(facts map[string]*CapacityFact, holds func(*CapacityFact) bool) bool {
	executable := 0
	for id, selected := range server.actions {
		if selected.deployment.ID == "" {
			continue
		}
		executable++
		if !holds(facts[id]) {
			return false
		}
	}
	return executable > 0
}

// isRuntimeContextRejection recognizes a local runtime's refusal of a prompt
// over its context before any output: llama.cpp's documented error, or an
// OpenAI-compatible context-length refusal (vLLM, MLX, a custom endpoint:
// context_length_exceeded or "maximum context length" in the error's code,
// type or message).
func isRuntimeContextRejection(deployment Deployment, status int, body []byte) bool {
	if deployment.Kind != "local" {
		return false
	}
	if status == 400 {
		for _, marker := range [][]byte{[]byte("exceed_context_size_error"), []byte("exceeds the available context size")} {
			if bytes.Contains(body, marker) {
				return true
			}
		}
	}
	return contextRefusalNames(status, body, "context_length_exceeded", "maximum context length")
}

func int64OrNil(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

// contextExceededMessage explains a turn no deployment can hold: the
// request's size from its facts and the largest configured context window
// among the executable targets. It starts with Anthropic's "prompt is too
// long" (Claude Code compacts on it) and carries OpenAI's "maximum context
// length" wording. Callers check exceedsAllDeployments first.
func (server *Server) contextExceededMessage(facts map[string]*CapacityFact) string {
	ids := make([]string, 0, len(server.actions))
	for id, selected := range server.actions {
		if selected.deployment.ID != "" {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	var window int64
	model, windowID := "", ""
	for _, id := range ids {
		if deployment := server.actions[id].deployment; deployment.ContextWindow > window {
			window, model, windowID = deployment.ContextWindow, server.actions[id].action.Model, id
		}
	}
	// The size is the largest target's own fact when it has one, else the
	// largest stated requirement.
	sized := facts[windowID]
	if sized == nil || sized.Required == nil {
		sized = nil
		for _, id := range ids {
			if fact := facts[id]; fact != nil && fact.Required != nil && (sized == nil || *fact.Required > *sized.Required) {
				sized = fact
			}
		}
	}
	// A stated size is never smaller than the window it is compared with.
	size := "every available deployment reports that it exceeds its context window, so it exceeds"
	if sized != nil && sized.Required != nil && *sized.Required > window {
		basis := map[string]string{capacityProvenanceEstimated: "estimated", capacityProvenanceMeasured: "measured"}[sized.Provenance]
		if basis == "" {
			basis = "reported"
		}
		size = fmt.Sprintf("this request needs %d tokens (%s), more than", *sized.Required, basis)
		if sized.prompt > 0 {
			size = fmt.Sprintf("about %d prompt tokens (%s) plus the output allowance need %d tokens, more than", sized.prompt, basis, *sized.Required)
		}
	}
	return fmt.Sprintf("prompt is too long: %s every available deployment; the largest maximum context length is %d tokens (%s). Compact or shorten the conversation, then retry.", size, window, model)
}

// exceedsAllDeployments reports whether every executable action has a
// context shortfall that stands for this request: any provenance, except an
// estimate on a runtime that refuses for itself (yieldsToRuntime), which
// counts as unknown. A target with no fact (unknown, such as a cloud target
// without a counter) is never assumed not to fit. Every refusal path (turn
// start, re-consult, re-consult limit, a service refusal) decides with it.
func (server *Server) exceedsAllDeployments(facts map[string]*CapacityFact) bool {
	return server.everyTarget(facts, isContextShortfall) && !server.anyYields(facts)
}

func (server *Server) anyYields(facts map[string]*CapacityFact) bool {
	for id, fact := range facts {
		if server.yieldsToRuntime(id, fact) {
			return true
		}
	}
	return false
}
