package execution

import (
	"context"
	"fmt"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// RegisterPromptCounter installs a serving-model/template counter before the
// server begins handling requests. Registration is optional: absent counters
// leave current-prefix evidence unknown. A registered counter that fails is
// an execution error, never replaced with an ARC count or text estimate.
//
// A provider adapter must count the exact rendered system/tools/messages and identify the carried
// prefix in the actual serving template before it can return a known count.
func (server *Server) RegisterPromptCounter(deploymentID string, counter cachecontext.PromptCounter) error {
	var deployment *Deployment
	for index := range server.config.Deployments {
		if server.config.Deployments[index].ID == deploymentID {
			deployment = &server.config.Deployments[index]
			break
		}
	}
	if deployment == nil || counter == nil {
		return fmt.Errorf("prompt counter requires a known deployment")
	}
	if deployment.TokenUnits == "" || counter.Identity() != promptIdentity(*deployment) {
		return fmt.Errorf("prompt counter does not match the admitted deployment model, template and token units")
	}
	if server.counters == nil {
		server.counters = make(map[string]cachecontext.PromptCounter)
	}
	if _, exists := server.counters[deploymentID]; exists {
		return fmt.Errorf("deployment already has a prompt counter")
	}
	server.counters[deploymentID] = counter
	return nil
}

func promptIdentity(deployment Deployment) cachecontext.PromptIdentity {
	return cachecontext.PromptIdentity{RuntimeIdentity: deployment.RuntimeIdentity, Model: deployment.WireModel, Protocol: string(deployment.Protocol), Units: deployment.TokenUnits}
}

// measureExecutionContext prepares each previously used worker through the
// same codec/renderer as execution, on fresh decoded requests and resumed
// placers. Nothing here commits a placer, refreshes a completion clock or
// asserts local KV residency. Measurements are reusable within this attempt.
func (server *Server) measureExecutionContext(ctx context.Context, body []byte, format llmprotocol.WireFormat, roles []string, state *branchState) (ExecutionContext, map[string]*cachecontext.MeasuredPrompt, error) {
	evidence := state.evidenceForRequest(server.instance, format, roles)
	measured := make(map[string]*cachecontext.MeasuredPrompt)
	for index := range evidence.Workers {
		worker := &evidence.Workers[index]
		// Prefix evidence always describes this request, never the previous one.
		worker.PrefixTokens, worker.OverlapTokens = nil, nil
		worker.TokenUnits, worker.PrefixProvenance = "", ""
		if server.counters[worker.DeploymentID] == nil {
			continue
		}
		selected, exists := server.actions[worker.Call.ActionID]
		if !exists || selected.deployment.ID != worker.DeploymentID {
			return ExecutionContext{}, nil, fmt.Errorf("cached worker no longer matches its admitted action")
		}
		request, envelope, _, err := server.engine.DecodeRequestForMutation(format, body)
		if err != nil {
			return ExecutionContext{}, nil, err
		}
		prepared, err := server.prepare(request, envelope, selected, state)
		if err != nil {
			return ExecutionContext{}, nil, fmt.Errorf("prepare worker prefix: %w", err)
		}
		current, err := server.measurePreparedPrompt(ctx, selected, prepared.body, state, nil)
		if err != nil {
			return ExecutionContext{}, nil, err
		}
		measured[worker.DeploymentID] = current
		if prefix := current.Prefix(worker.prompt); prefix != nil {
			worker.PrefixTokens, worker.OverlapTokens = &prefix.Tokens, prefix.OverlapTokens
			worker.TokenUnits, worker.PrefixProvenance = prefix.Units, prefix.Provenance
		}
	}
	return evidence, measured, nil
}

// measurePreparedPrompt stages the exact dispatch prompt. Only the successful
// completion branch may store it in WorkerEvidence.prompt. Same-body reuse
// avoids calling a provider count endpoint twice for the selected candidate.
func (server *Server) measurePreparedPrompt(ctx context.Context, selected admittedAction, body []byte, state *branchState, measured map[string]*cachecontext.MeasuredPrompt) (*cachecontext.MeasuredPrompt, error) {
	counter := server.counters[selected.deployment.ID]
	if counter == nil {
		return nil, nil
	}
	prompt := cachecontext.PreparedPrompt{Identity: promptIdentity(selected.deployment), ContextEpoch: state.evidence(server.instance).ContextEpoch, Body: body}
	if previous := measured[selected.deployment.ID]; previous.Matches(prompt) {
		return previous, nil
	}
	result, err := cachecontext.MeasurePrompt(ctx, counter, prompt)
	if err != nil {
		return nil, fmt.Errorf("measure worker prefix: %w", err)
	}
	return result, nil
}
