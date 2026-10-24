package execution

import (
	"bytes"
	"context"
	"fmt"
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
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Units: &units, Provenance: "local_measured"}
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
		fact.Required, fact.Units = previous.Required, previous.Units
	} else if deployment.TokenUnits != "" {
		units := deployment.TokenUnits
		fact.Units = &units
	}
	return fact
}

// measureCapacity prepares every counter-qualified action exactly as dispatch
// would and counts it with the target's own runtime tokenizer. Targets without
// a qualified counter stay unknown. Nothing is committed.
func (server *Server) measureCapacity(ctx context.Context, body []byte, format llmprotocol.WireFormat, state *branchState) (map[string]*CapacityFact, map[string]*cachecontext.MeasuredPrompt, error) {
	facts := make(map[string]*CapacityFact)
	measured := make(map[string]*cachecontext.MeasuredPrompt)
	ids := make([]string, 0, len(server.actions))
	for id := range server.actions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		selected := server.actions[id]
		if selected.deployment.ID == "" || server.counters[selected.deployment.ID] == nil {
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
			continue
		}
		prompt, err := server.measurePreparedPrompt(ctx, selected, prepared.body, state, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("measure target capacity: %w", err)
		}
		measured[id] = prompt
		facts[id] = capacityFact(selected.deployment, prompt, prepared.outputLimit)
	}
	return facts, measured, nil
}

// isRuntimeContextRejection recognizes llama.cpp's documented refusal of a
// prompt that exceeds its slot context. It applies only to local targets and
// only before any output was delivered.
func isRuntimeContextRejection(deployment Deployment, status int, body []byte) bool {
	if deployment.Kind != "local" || status != 400 {
		return false
	}
	for _, marker := range [][]byte{[]byte("exceed_context_size_error"), []byte("exceeds the available context size")} {
		if bytes.Contains(body, marker) {
			return true
		}
	}
	return false
}

func int64OrNil(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}
