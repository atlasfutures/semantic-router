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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineReportAPIPath tells the cell what ran after a route lookup: a caller
// that ran the model itself reports the model and its usage, so the
// conversation's episode records the turn as it ran rather than as decided.
// It is answered here, beside /v1/routes, for the same reason: the episode
// store lives behind this filter.
const raylineReportAPIPath = "/v1/report"

const (
	// raylineReportSourceHeader says who reports: "gateway" for the gateway
	// in front of the cell reporting a call it sent to a provider itself,
	// "client" (the default) for a report the gateway forwards. It decides
	// only which of two reports of one turn wins; the gateway sets it and
	// overwrites any value a client sent.
	raylineReportSourceHeader = "x-rayline-report-source"
	// raylineReportMaxBodyBytes bounds a report; a valid one is well under
	// a kilobyte.
	raylineReportMaxBodyBytes = 16 * 1024
	// raylineReportMaxModelBytes bounds the model name a report records.
	raylineReportMaxModelBytes = 256
	// raylineReportMaxTTLSeconds bounds a reported cache lifetime.
	raylineReportMaxTTLSeconds = 24 * 60 * 60
	// raylineReportNotTracked is the answer of a cell that keeps no episodes
	// for route lookups: episode writes off, or no policy service.
	raylineReportNotTracked = "not_tracked"
)

// isRaylineReportRequest matches a report, which the body phase answers.
func isRaylineReportRequest(ctx *RequestContext) bool {
	return ctx != nil && normalizeRequestPath(requestPathFromContext(ctx)) == raylineReportAPIPath
}

// isRaylineServedRequest matches the paths this router answers itself and
// must never forward: a route lookup and a report.
func isRaylineServedRequest(ctx *RequestContext) bool {
	return isRaylineRoutesRequest(ctx) || isRaylineReportRequest(ctx)
}

// raylineReportBody is a report as sent. Every member is read strictly: an
// unknown one is refused, as the routing block of a lookup is.
type raylineReportBody struct {
	RouteID  *string `json:"route_id"`
	ReportID *string `json:"report_id"`
	// Model is the ARC id the conversation is routed by; raylineRoutesTier
	// reads it to pick the tier's episode store.
	Model    json.RawMessage        `json:"model"`
	Executed *raylineReportExecuted `json:"executed"`
}

type raylineReportExecuted struct {
	Model         string              `json:"model"`
	ThinkingLevel *string             `json:"thinking_level"`
	ActionID      *string             `json:"action_id"`
	Usage         *raylineReportUsage `json:"usage"`
	FinishedAt    string              `json:"finished_at"`
}

// raylineReportUsage is what the generation flow records of a call: prompt
// tokens (the whole prompt, cache read and write included), the cache legs
// and the cache lifetime. Nothing else is accepted.
type raylineReportUsage struct {
	PromptTokens     *int `json:"prompt_tokens"`
	CacheReadTokens  *int `json:"cache_read_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
	CacheTTLSeconds  *int `json:"cache_ttl_seconds"`
}

// raylineReport is a validated report.
type raylineReport struct {
	routeID, reportID, source string
	model, thinkingLevel      string
	actionID                  string
	call                      cachecontext.CallCache
	finishedAt                time.Time
}

// raylineReportAnswer is the 200 answer to a report.
type raylineReportAnswer struct {
	Object     string  `json:"object"`
	Status     string  `json:"status"`
	Reason     string  `json:"reason"`
	RouteID    *string `json:"route_id"`
	TurnIndex  *uint64 `json:"turn_index"`
	Attributed *bool   `json:"attributed"`
}

func reportInvalid(message string) *routesContractError {
	return routesInvalid("invalid_report", message)
}

// readRaylineReport reads and validates a report body and its source.
func readRaylineReport(body []byte, source string, now time.Time) (raylineReport, *routesContractError) {
	var wire raylineReportBody
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	// The object must be the whole body: anything after it, even a stray
	// delimiter, is refused before the tier is read from the same bytes.
	if err := decoder.Decode(&wire); err != nil {
		return raylineReport{}, reportInvalid("report body is not a valid report object")
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return raylineReport{}, reportInvalid("report body is not a valid report object")
	}
	report := raylineReport{source: raylinearc.ReportSourceClient}
	switch source {
	case "", raylinearc.ReportSourceClient:
	case raylinearc.ReportSourceGateway:
		report.source = source
	default:
		return raylineReport{}, reportInvalid(raylineReportSourceHeader + " must be gateway or client")
	}
	for _, id := range []struct {
		name  string
		value *string
	}{{"route_id", wire.RouteID}, {"report_id", wire.ReportID}} {
		if id.value != nil && (*id.value == "" || len(*id.value) > raylinearc.MaxReportIDBytes) {
			return raylineReport{}, reportInvalid(fmt.Sprintf("%s must be 1 to %d bytes", id.name, raylinearc.MaxReportIDBytes))
		}
	}
	report.routeID, report.reportID = stringOf(wire.RouteID), stringOf(wire.ReportID)
	// Only an absent model means the default tier: one that is present must
	// name a model (null, empty or blank is refused).
	if len(wire.Model) > 0 {
		var model string
		if json.Unmarshal(wire.Model, &model) != nil || strings.TrimSpace(model) == "" {
			return raylineReport{}, reportInvalid("model, when present, must name an ARC model")
		}
	}
	executed := wire.Executed
	if executed == nil {
		return raylineReport{}, reportInvalid("executed is required")
	}
	// Names are compared and stored without surrounding whitespace, and a
	// blank one names nothing.
	report.model = strings.TrimSpace(executed.Model)
	if report.model == "" || len(report.model) > raylineReportMaxModelBytes {
		return raylineReport{}, reportInvalid(fmt.Sprintf("executed.model is required, at most %d bytes", raylineReportMaxModelBytes))
	}
	if executed.ThinkingLevel != nil && strings.TrimSpace(*executed.ThinkingLevel) == "" {
		return raylineReport{}, reportInvalid("executed.thinking_level must not be empty")
	}
	report.thinkingLevel = strings.TrimSpace(stringOf(executed.ThinkingLevel))
	if executed.ActionID != nil && !isLowerHexID(*executed.ActionID) {
		return raylineReport{}, reportInvalid("executed.action_id must be 64 lowercase hex characters")
	}
	report.actionID = stringOf(executed.ActionID)
	finished, err := time.Parse(time.RFC3339Nano, executed.FinishedAt)
	if err != nil {
		return raylineReport{}, reportInvalid("executed.finished_at must be an RFC 3339 time")
	}
	if finished.After(now.Add(raylinearc.ReportClockSkew)) {
		return raylineReport{}, reportInvalid("executed.finished_at is in the future")
	}
	// Kept as sent, even a little ahead of this clock. How late is too late
	// is decided against the decision the report applies to, not here.
	report.finishedAt = finished.UTC()
	if usage := executed.Usage; usage != nil {
		for _, tokens := range []struct {
			name  string
			value *int
		}{{"prompt_tokens", usage.PromptTokens}, {"cache_read_tokens", usage.CacheReadTokens}, {"cache_write_tokens", usage.CacheWriteTokens}} {
			if tokens.value != nil && *tokens.value < 0 {
				return raylineReport{}, reportInvalid("executed.usage." + tokens.name + " must not be negative")
			}
		}
		if ttl := usage.CacheTTLSeconds; ttl != nil && (*ttl < 1 || *ttl > raylineReportMaxTTLSeconds) {
			return raylineReport{}, reportInvalid(fmt.Sprintf("executed.usage.cache_ttl_seconds must be 1 to %d", raylineReportMaxTTLSeconds))
		}
		report.call = cachecontext.CallCache{
			PromptTokens: usage.PromptTokens, CacheReadTokens: usage.CacheReadTokens, CacheWriteTokens: usage.CacheWriteTokens,
		}
		if usage.CacheTTLSeconds != nil {
			report.call.TTLSeconds, report.call.TTLSource = *usage.CacheTTLSeconds, raylinearc.ReportedTTLSource
		}
	}
	return report, nil
}

func stringOf(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func isLowerHexID(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

// reportCandidates are the actions a report's model may have run, in the
// package's order: the action it names, or the actions dispatched on the
// model, at its thinking level when it names one. None means no arm of the
// package serves the model. An action or level the model does not have is
// refused.
func (scorer *policyServiceScorer) reportCandidates(report raylineReport) ([]string, *routesContractError) {
	if report.actionID != "" {
		binding, bound := scorer.bindings[report.actionID]
		if !bound || !scorer.actionServes(report.actionID, report.model) {
			return nil, reportInvalid("executed.action_id is not an action of executed.model")
		}
		if report.thinkingLevel != "" && binding.level != report.thinkingLevel {
			return nil, reportInvalid("executed.thinking_level is not the level of executed.action_id")
		}
		return []string{report.actionID}, nil
	}
	var served, candidates []string
	for _, actionID := range scorer.actionOrder {
		if !scorer.actionServes(actionID, report.model) {
			continue
		}
		served = append(served, actionID)
		if report.thinkingLevel == "" || scorer.bindings[actionID].level == report.thinkingLevel {
			candidates = append(candidates, actionID)
		}
	}
	if len(served) > 0 && len(candidates) == 0 {
		return nil, reportInvalid("executed.thinking_level is not a level of executed.model")
	}
	return candidates, nil
}

// actionServes reports whether model names what actionID runs: the model
// a route lookup returns for it (its dispatch worker's provider model), that
// worker, or the action's own model.
func (scorer *policyServiceScorer) actionServes(actionID, model string) bool {
	dispatch := scorer.bindings[actionID].dispatchArm
	return model == scorer.workers[dispatch].Model || model == scorer.workerIDs[dispatch] ||
		model == scorer.actionModel(actionID)
}

// reportedRun maps a report onto the package, given the action already
// decided or recorded for the turn: the one candidate, or the decided action
// when it is a candidate (the caller ran the decided model at the decided
// level), else the candidates' arm with the action unknown.
func (scorer *policyServiceScorer) reportedRun(report raylineReport, candidates []string, decided string) raylinearc.ReportedRun {
	if len(candidates) == 0 {
		return unattributedReportRun(report)
	}
	action := ""
	switch {
	case len(candidates) == 1:
		action = candidates[0]
	case slices.Contains(candidates, decided):
		action = decided
	}
	run := raylinearc.ReportedRun{Call: report.call}
	if action == "" {
		// The model's actions may sit on different policy arms (two arms can
		// share one dispatch worker): with no action to tell them apart, the
		// arm is not known and the turn is recorded with none.
		arm := scorer.bindings[candidates[0]].arm
		for _, candidate := range candidates[1:] {
			if scorer.bindings[candidate].arm != arm {
				return unattributedReportRun(report)
			}
		}
		run.Arm = arm
		run.Model = scorer.actionModel(candidates[0])
		return run
	}
	run.Arm, run.ActionID, run.Model = scorer.bindings[action].arm, action, scorer.actionModel(action)
	return run
}

// unattributedReportRun is the rule for a report of a model no arm of the
// package serves (a local model, another provider's): the turn is recorded
// with no arm -- counted, the held arm cleared, no ledger or usage entry --
// rather than refused. Refused, the pending decision would later be inferred
// as the decided model, claiming a warm cache on a model that never ran.
func unattributedReportRun(report raylineReport) raylinearc.ReportedRun {
	return raylinearc.ReportedRun{Arm: -1, Model: report.model}
}

// handleRaylineReportAPI answers one report and returns an immediate
// response; nil when the request is not a report.
func (r *OpenAIRouter) handleRaylineReportAPI(body []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if !isRaylineReportRequest(ctx) {
		return nil
	}
	if !r.raylineRoutesAPIEnabled() {
		return r.createRaylineRoutesError(ctx, 404, "not_found_error", "endpoint not found")
	}
	if method := raylineRoutesHeader(ctx, ":method"); method != "" && method != "POST" {
		return r.createRaylineRoutesError(ctx, 405, "invalid_request_error", "method not allowed")
	}
	if len(body) > raylineReportMaxBodyBytes {
		return r.createRaylineRoutesError(ctx, 413, "invalid_request_error", "report body exceeds its limit")
	}
	report, refusal := readRaylineReport(body, raylineRoutesHeader(ctx, raylineReportSourceHeader), time.Now())
	if refusal != nil {
		return r.createRaylineRoutesCodedError(ctx, refusal)
	}
	recipe, target, refusal := r.raylineRoutesTier(body)
	if refusal != nil {
		return r.createRaylineRoutesCodedError(ctx, refusal)
	}
	// Only an absent model means the default tier. A named one must be a
	// name this cell's config routes to a recipe serving route lookups, or a
	// public ARC tier; a lookup reads any other name as the default tier, a
	// report refuses it.
	if model := strings.TrimSpace(raylineRoutesModelOf(body)); model != "" {
		named, configured := r.Config.RecipeForRoutingModel(model)
		if (configured && r.raylineRoutesDecision(named) == nil) || (!configured && raylineARCCanonicalTier(model) == "") {
			return r.createRaylineRoutesCodedError(ctx, routesInvalid("unknown_model", "Unknown ARC model: "+model))
		}
	}
	episode := raylineRoutesEpisodeIdentity(ctx)
	if episode == "" {
		return r.createRaylineRoutesCodedError(ctx, routesInvalid("missing_session", raylineRoutesSessionHeader+" is required"))
	}
	arc := target.Algorithm.RaylineARC
	// Only a policy-service cell keeps pending decisions; elsewhere a lookup
	// commits at decision time, and a report would count its turn twice.
	if arc.PolicyService == nil {
		return r.raylineReportAnswer(ctx, report, raylinearc.ReportOutcome{Reason: raylineReportNotTracked})
	}
	// The model is mapped before the switch is read, so a report is refused
	// or accepted the same way whether episode writes are on or off.
	scorer := newPolicyServiceScorer(r.Config, target)
	candidates, refusal := scorer.reportCandidates(report)
	if refusal != nil {
		return r.createRaylineRoutesCodedError(ctx, refusal)
	}
	if !arc.RoutesAPI.EpisodeWrites {
		return r.raylineReportAnswer(ctx, report, raylinearc.ReportOutcome{Reason: raylineReportNotTracked})
	}
	store := r.raylineARCEpisodeStoreOf(recipe.Name)
	if store == nil {
		return r.createRaylineRoutesCodedError(ctx, raylineReportUnavailable())
	}
	reportContext, cancel := context.WithTimeout(r.raylineRoutesContext(ctx), arc.RoutesAPI.EffectiveDeadline())
	defer cancel()
	episodeIDHash := raylinearc.HashEpisodeID(episode)
	outcome, err := applyRaylineReport(reportContext, store, arc.Episode.RelaxedConsistency(), episodeIDHash, len(target.ModelRefs),
		raylinearc.Report{
			RouteID: report.routeID, ReportID: report.reportID, Source: report.source, FinishedAt: report.finishedAt,
			Run: func(decided string) raylinearc.ReportedRun { return scorer.reportedRun(report, candidates, decided) },
		})
	fields := map[string]interface{}{
		"episode_id_hash": episodeIDHash, "route_id": report.routeID, "report_id": report.reportID,
		"source": report.source, "reason": outcome.Reason, "recorded": outcome.Recorded,
		"turn_index": outcome.TurnIndex, "attributed": outcome.Attributed,
	}
	if err != nil {
		failure := raylineReportFailure(err)
		fields["error"] = failure.code
		logging.ComponentEvent("extproc", "rayline_arc_route_report", fields)
		metrics.RecordRaylineARCRouteReport(report.source, "error", failure.code)
		response := r.createRaylineRoutesCodedError(ctx, failure)
		if failure.status == 429 {
			return raylineRoutesRetryAfter(response, raylineRoutesContendedRetrySeconds)
		}
		return response
	}
	logging.ComponentEvent("extproc", "rayline_arc_route_report", fields)
	return r.raylineReportAnswer(ctx, report, outcome)
}

// raylineReportAnswer is the rule for what a report's caller is told: 200
// whether the report changed the episode or was ignored, with the reason, so
// a caller that reports and moves on never retries a report that was read.
func (r *OpenAIRouter) raylineReportAnswer(
	ctx *RequestContext,
	report raylineReport,
	outcome raylinearc.ReportOutcome,
) *ext_proc.ProcessingResponse {
	answer := raylineReportAnswer{Object: "route.report", Status: "ignored", Reason: outcome.Reason}
	if outcome.Recorded {
		turn, attributed := outcome.TurnIndex, outcome.Attributed
		answer.Status, answer.TurnIndex, answer.Attributed = "recorded", &turn, &attributed
		if outcome.RouteID != "" {
			routeID := outcome.RouteID
			answer.RouteID = &routeID
		}
	}
	metrics.RecordRaylineARCRouteReport(report.source, answer.Status, answer.Reason)
	if ctx != nil {
		ctx.ImmediateResponseEncoded = true
	}
	return r.createJSONResponse(200, answer)
}

func raylineReportUnavailable() *routesContractError {
	return &routesContractError{status: 503, errType: "api_error", code: "unavailable", message: "the episode store is unavailable"}
}

// raylineReportFailure is the answer to a store round trip that failed: a
// conversation another request holds is busy for a moment, a deadline is a
// timeout, anything else the store's unavailability.
func raylineReportFailure(err error) *routesContractError {
	switch {
	case errors.Is(err, raylinearc.ErrEpisodeLeaseHeld) || errors.Is(err, raylinearc.ErrEpisodeConflict):
		return &routesContractError{
			status: 429, errType: "rate_limit_error", code: "episode_busy",
			message: "the conversation is being updated by another request; retry shortly",
		}
	case errors.Is(err, context.DeadlineExceeded):
		return &routesContractError{status: 504, errType: "timeout_error", code: "deadline_exceeded", message: "the report did not complete within its deadline"}
	}
	return raylineReportUnavailable()
}

// applyRaylineReport applies report to the conversation's episode. A
// lease-free read first answers a conversation with no history without
// taking its lease or leaving anything behind. A strict episode is then
// leased, read again and committed under the lease; a relaxed one is
// committed if unchanged since the read, and a collision answers busy.
func applyRaylineReport(
	ctx context.Context,
	store raylinearc.EpisodeStore,
	relaxed bool,
	episodeIDHash string,
	workerCount int,
	report raylinearc.Report,
) (raylinearc.ReportOutcome, error) {
	snapshots, ok := store.(raylinearc.EpisodeSnapshotStore)
	if !ok {
		return raylinearc.ReportOutcome{}, errors.New("the episode store serves no lease-free read")
	}
	state, read, err := snapshots.Snapshot(ctx, episodeIDHash, workerCount)
	if err != nil {
		return raylinearc.ReportOutcome{}, err
	}
	if !raylinearc.EpisodeHasHistory(state) {
		return raylinearc.ReportOutcome{Reason: raylinearc.ReportNoEpisode}, nil
	}
	if relaxed {
		next, outcome, applyErr := raylinearc.ApplyReport(state, report)
		if applyErr != nil || !outcome.Recorded {
			return outcome, applyErr
		}
		return outcome, snapshots.CommitIfUnchanged(ctx, episodeIDHash, read, next)
	}
	lease, leased, err := store.Prepare(ctx, episodeIDHash, workerCount)
	if err != nil {
		return raylinearc.ReportOutcome{}, err
	}
	next, outcome, err := raylinearc.ApplyReport(leased, report)
	if err == nil && outcome.Recorded {
		err = store.Commit(ctx, lease, lease.Version(), next)
	}
	if err != nil || !outcome.Recorded {
		// The lease is released even past the deadline, so the next request
		// does not wait out its TTL, within the bound a turn's release has.
		abortContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), episodeFinalizeTimeout)
		_ = store.Abort(abortContext, lease)
		cancel()
	}
	return outcome, err
}
