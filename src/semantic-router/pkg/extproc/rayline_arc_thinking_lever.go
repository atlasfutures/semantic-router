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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

// Skip reasons that belong to the router rather than the planner.
const (
	thinkingSkipWorkerUnbound = "worker_unbound"
	thinkingSkipNoEpisode     = "no_episode"
)

// raylineARCThinkingTrace is what one turn's lever did. It carries level
// names and counts only, never lever bytes or transcript content.
type raylineARCThinkingTrace struct {
	Lever          string
	Source         string
	Admission      string
	ExportSHA256   string
	ControlInForce string
	LevelRequested string
	LevelInForce   string
	Rank           int
	Propensity     float64
	Emitted        bool
	Retry          bool
	Placement      string
	Replayed       int
	Epoch          uint32
	ResetReason    string
	Skipped        string
}

// applyRaylineARCThinkingLever writes this turn's lever items into the
// provider-bound request. It runs once per request, after selection, so the
// selector never sees an item: the trained encoder reads the client's
// transcript, and that is what it keeps reading.
//
// A worker the decision does not bind is never steered, and its turn leaves
// the ledger untouched. Another worker's items stay out of its transcript,
// because a worker the registry has not admitted may refuse the item
// outright. Returning to a bound worker replays the ledger, and its cached
// prefix still matches.
func (r *OpenAIRouter) applyRaylineARCThinkingLever(
	request *llmprotocol.Request,
	ctx *RequestContext,
) (bool, error) {
	if ctx == nil || ctx.RaylineARCThinking != nil || ctx.RaylineARCDispatch == nil {
		return false, nil
	}
	lever := raylineARCThinkingLeverConfig(ctx.VSRSelectedDecision)
	if lever == nil || !lever.Enabled {
		return false, nil
	}
	trace := &raylineARCThinkingTrace{Source: lever.Source, LevelRequested: lever.Level, Propensity: 1}
	ctx.RaylineARCThinking = trace
	defer recordThinkingLeverTurn(trace)
	bindingConfig, bound := lever.Workers[ctx.RaylineARCDispatch.ID]
	if !bound {
		trace.Skipped = thinkingSkipWorkerUnbound
		return false, nil
	}
	ledger, turnIndex, hasEpisode := ctx.RaylineARCTransaction.committedThinking()
	if !hasEpisode {
		trace.Skipped = thinkingSkipNoEpisode
		return false, nil
	}
	binding := bindingConfig.Binding()
	trace.Lever, trace.Admission = string(binding.Lever), bindingConfig.Admission
	trace.ExportSHA256 = bindingConfig.ExportSHA256
	plan, err := thinkinglever.PlanTurn(thinkinglever.Turn{
		Binding:                binding,
		Ledger:                 ledger,
		Messages:               thinkinglever.Messages(request.Messages),
		TurnIndex:              turnIndex,
		Requested:              lever.Level,
		MinTurnsBetweenChanges: lever.MinSpacingTurns,
		MaxEntries:             lever.MaxLedgerEntries,
	})
	if err != nil {
		return false, err
	}
	messages, err := thinkinglever.ApplyLedger(request.Messages, binding.Lever, plan.Next)
	if err != nil {
		return false, err
	}
	if ledger != nil || len(plan.Next.Entries) > 0 {
		ctx.RaylineARCTransaction.stageThinkingLedger(plan.Next)
	}
	fillThinkingTrace(trace, binding, plan)
	if len(messages) == len(request.Messages) && plan.Replayed == 0 && !plan.Emitted {
		return false, nil
	}
	request.Messages = messages
	return true, nil
}

func fillThinkingTrace(trace *raylineARCThinkingTrace, binding thinkinglever.Binding, plan thinkinglever.Plan) {
	trace.LevelInForce, trace.ControlInForce = plan.LevelInForce, plan.ControlInForce
	if level, ok := binding.Level(plan.LevelInForce); ok {
		trace.Rank = level.Rank
	}
	trace.Emitted, trace.Retry = plan.Emitted, plan.Retry
	trace.Placement = string(plan.Placement)
	trace.Replayed, trace.Epoch = plan.Replayed, plan.Next.Epoch
	trace.ResetReason, trace.Skipped = plan.ResetReason, plan.Skipped
}

// recordThinkingLeverTurn counts one governed turn. Every label value comes
// from a closed set in this package or the planner's.
func recordThinkingLeverTurn(trace *raylineARCThinkingTrace) {
	outcome, reason := "held", trace.ResetReason
	switch {
	case trace.Retry:
		outcome = "retry"
	case trace.Emitted:
		outcome = "emitted"
	case trace.Skipped != "":
		outcome, reason = "skipped", trace.Skipped
	}
	metrics.RecordRaylineARCThinkingLeverTurn(trace.Lever, outcome, reason)
}

func raylineARCThinkingLeverConfig(decision *config.Decision) *config.RaylineARCThinkingLeverConfig {
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return nil
	}
	return decision.Algorithm.RaylineARC.ThinkingLever
}

// appendRaylineARCThinkingFields names what the lever did on the routing
// record, so a turn's level can be joined to its usage without reading any
// transcript.
func appendRaylineARCThinkingFields(record map[string]interface{}, ctx *RequestContext) {
	if ctx == nil {
		return
	}
	if base := ctx.RaylineARCWorkerThinking; base != nil {
		record["thinking_base_level"] = base.Level
		record["thinking_base_wire"] = base.Wire
	}
	if ctx.RaylineARCThinking == nil {
		return
	}
	trace := ctx.RaylineARCThinking
	record["thinking_source"] = trace.Source
	record["thinking_level_requested"] = trace.LevelRequested
	if trace.Skipped != "" {
		record["thinking_skipped"] = trace.Skipped
	}
	if trace.Lever == "" {
		return
	}
	record["thinking_lever"] = trace.Lever
	record["thinking_admission"] = trace.Admission
	record["thinking_level_in_force"] = trace.LevelInForce
	if trace.ControlInForce != "" {
		record["thinking_control_sha256"] = trace.ControlInForce
	}
	if trace.ExportSHA256 != "" {
		record["thinking_export_sha256"] = trace.ExportSHA256
	}
	record["thinking_level_rank"] = trace.Rank
	record["thinking_propensity"] = trace.Propensity
	record["thinking_emitted"] = trace.Emitted
	record["thinking_retry"] = trace.Retry
	record["thinking_replayed"] = trace.Replayed
	record["thinking_epoch"] = trace.Epoch
	if trace.Placement != "" {
		record["thinking_placement"] = trace.Placement
	}
	if trace.ResetReason != "" {
		record["thinking_epoch_reset_reason"] = trace.ResetReason
	}
}
