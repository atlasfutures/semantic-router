//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/textproto"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// routeDecisionPath is the literal path the decision-only caller already
// speaks. It is a deliberate exception to the /api/v1/* catalog convention:
// the wire contract is fixed by a shipped client, so the path is an input to
// this router, not a choice it gets to make.
const routeDecisionPath = "/v1/route"

// routeDecisionRetryAfterSeconds is the Retry-After a contended consult
// carries. One second is deliberately short: the holder of a session's episode
// is usually one consult ahead, not minutes away, and a caller that waits
// longer than its own request budget would drop the turn instead of retrying.
const routeDecisionRetryAfterSeconds = "1"

const (
	// routeIDHeader carries the caller's own route id. This adapter adopts it
	// as the decision id and echoes it, so one id spans the caller's durable
	// usage row and this router's decision record.
	routeIDHeader = "x-rayline-route-id"
	// routeSessionHeader carries the caller's session identity. The runtime
	// maps it onto the routing algorithm's configured episode identity.
	routeSessionHeader = "x-rayline-session"
	// executedModelHeader reports what the caller actually ran last turn. It
	// is record-only; see routerruntime.RouteDecisionRequest.
	executedModelHeader = "x-rayline-executed-model"
)

// routeDecisionBodyLimit keeps decision-only requests on the same bounded JSON
// contract as the rest of the management API. Real conversation envelopes can
// exceed 4 MiB before ARC normalization removes non-text content.
const routeDecisionBodyLimit int64 = defaultJSONRequestBodyLimit

var (
	// routeIDPattern matches the caller's minted route id, case-insensitively,
	// exactly as the reference decision server validates it.
	routeIDPattern = regexp.MustCompile(`\A(?i:rt_(ml_)?[a-f0-9-]{1,40})\z`)
	// executedModelPattern matches a bounded provider model identifier.
	executedModelPattern = regexp.MustCompile(`\A[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\z`)
)

func apiRouteDecisionRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{
				Path:        routeDecisionPath,
				Method:      http.MethodPost,
				Description: "Select a worker for one request without executing it",
			},
			routePolicy{
				Permission:  PermRouteDecision,
				Sensitivity: SensitivityOperational,
			},
			(*ClassificationAPIServer).handleRouteDecision,
			jsonBodyWithLimit(routeDecisionBodyLimit),
		),
	}
}

// handleRouteDecision answers one decision-only consult: it runs the routing
// algorithm and stops before dispatch, so the caller executes the chosen
// worker itself.
//
// Validation ordering is part of the contract. Every malformed request is
// rejected before the runtime is consulted, because consulting it reserves an
// episode turn — malformed traffic must not be able to shift a caller's route
// indices.
func (s *ClassificationAPIServer) handleRouteDecision(w http.ResponseWriter, r *http.Request) {
	decisionStarted := time.Now()

	body, err := readJSONRequestBody(r, routeDecisionBodyLimit)
	if err != nil {
		// This endpoint's failures are read against the reference decision
		// server's shape, so an oversized body reports like every other
		// rejection here rather than in the management envelope.
		if errors.Is(err, errRequestBodyTooLarge) {
			s.writeRouteDecisionError(w, http.StatusRequestEntityTooLarge, "request body is too large")
			return
		}
		s.writeRouteDecisionError(w, http.StatusBadRequest, "request body could not be read")
		return
	}
	if detail := validateRouteDecisionBody(body); detail != "" {
		s.writeRouteDecisionError(w, http.StatusBadRequest, detail)
		return
	}
	decisionID, detail := routeDecisionID(r.Header)
	if detail != "" {
		s.writeRouteDecisionError(w, http.StatusBadRequest, detail)
		return
	}
	executedModel, detail := routeDecisionExecutedModel(r.Header)
	if detail != "" {
		s.writeRouteDecisionError(w, http.StatusBadRequest, detail)
		return
	}

	runtime := s.routeDecisionRuntime()
	if runtime == nil {
		s.writeRouteDecisionError(
			w,
			http.StatusServiceUnavailable,
			"decision-only routing is not available on this router",
		)
		return
	}

	decision, err := runtime.RouteDecision(r.Context(), routerruntime.RouteDecisionRequest{
		Body:          body,
		DecisionID:    decisionID,
		SessionID:     strings.TrimSpace(r.Header.Get(routeSessionHeader)),
		ExecutedModel: executedModel,
	})
	if err != nil {
		// Fail closed. The algorithm owns the choice and has no fallback
		// worker, so answering with a default here would silently replace a
		// policy decision with this adapter's guess.
		//
		// Contention is reported separately. A 503 says this router is
		// unavailable, which sends a caller into fallback and reads as an
		// outage in its dashboards; a contended consult is neither. It is
		// back-pressure from a healthy router -- already busy with this
		// session, or at its encoder admission cap -- so it answers 429 and
		// names when to come back.
		contended := errors.Is(err, routerruntime.ErrRouteDecisionContended)
		logging.ComponentErrorEvent("apiserver", "route_decision_failed", map[string]interface{}{
			"decision_id": decisionID,
			"error":       err.Error(),
			"contended":   contended,
		})
		if contended {
			w.Header().Set("Retry-After", routeDecisionRetryAfterSeconds)
			s.writeRouteDecisionError(
				w,
				http.StatusTooManyRequests,
				"route decision contended: routing capacity is briefly exhausted",
			)
			return
		}
		s.writeRouteDecisionError(w, http.StatusServiceUnavailable, "route decision failed")
		return
	}
	if decision.SelectedWorker == "" || decision.WorkerModel == "" {
		logging.ComponentErrorEvent("apiserver", "route_decision_incomplete", map[string]interface{}{
			"decision_id": decisionID,
		})
		s.writeRouteDecisionError(w, http.StatusServiceUnavailable, "route decision was incomplete")
		return
	}

	s.logRouteDecision(decisionID, executedModel, decision)
	s.writeJSONResponse(w, http.StatusOK, routeDecisionResponse(
		decisionID,
		decision,
		time.Since(decisionStarted),
	))
}

// routeDecisionResponse emits only fields this router can source. Absent keys
// read as "unknown" to the caller; invented ones would read as measured and
// poison an offline join, so anything without a real source is left out.
func routeDecisionResponse(
	decisionID string,
	decision routerruntime.RouteDecision,
	elapsed time.Duration,
) map[string]interface{} {
	response := map[string]interface{}{
		"decision_id":         decisionID,
		"selected_worker":     decision.SelectedWorker,
		"worker_model":        decision.WorkerModel,
		"decision_latency_ms": roundMillis(elapsed),
	}
	if decision.Provider != "" {
		response["provider"] = decision.Provider
	}
	return response
}

func roundMillis(elapsed time.Duration) float64 {
	return math.Round(float64(elapsed.Nanoseconds())/1e3) / 1e3
}

func (s *ClassificationAPIServer) routeDecisionRuntime() routerruntime.RouteDecisionRuntime {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	return s.runtimeRegistry.RouteDecisionRuntime()
}

// validateRouteDecisionBody checks only what the contract promises to reject.
// The body is otherwise forwarded unmutated, so the routing algorithm sees
// exactly what the caller sent.
func validateRouteDecisionBody(body []byte) string {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil || envelope == nil {
		return "request body must be a JSON object"
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(envelope["messages"], &messages); err != nil || len(messages) == 0 {
		return "messages must be a non-empty list of message objects"
	}
	for _, message := range messages {
		var object map[string]json.RawMessage
		if json.Unmarshal(message, &object) != nil || object == nil {
			return "messages must be a non-empty list of message objects"
		}
	}
	if raw, present := envelope["model"]; present && string(raw) != "null" {
		var model string
		if json.Unmarshal(raw, &model) != nil {
			return "model must be a string"
		}
	}
	return ""
}

// routeDecisionID adopts the caller's route id, or mints one when the caller
// sent none. A present-but-unusable value is refused rather than replaced: a
// silent fallback would re-split the very ids this header exists to join.
func routeDecisionID(header http.Header) (string, string) {
	value, detail := singleHeaderValue(header, routeIDHeader)
	if detail != "" {
		return "", detail
	}
	if value == "" {
		return uuid.NewString(), ""
	}
	if !routeIDPattern.MatchString(value) {
		return "", "invalid " + routeIDHeader + " header: expected the caller's route id"
	}
	return value, ""
}

func routeDecisionExecutedModel(header http.Header) (string, string) {
	value, detail := singleHeaderValue(header, executedModelHeader)
	if detail != "" {
		return "", detail
	}
	if value != "" && !executedModelPattern.MatchString(value) {
		return "", "invalid " + executedModelHeader + " header: expected a provider model identifier"
	}
	return value, ""
}

// singleHeaderValue refuses duplicates instead of picking one. A trusted proxy
// stamps exactly one copy, so a second copy means something upstream is
// shadowing it and which value wins would be proxy-dependent.
func singleHeaderValue(header http.Header, name string) (string, string) {
	values := header.Values(textproto.CanonicalMIMEHeaderKey(name))
	if len(values) == 0 {
		return "", ""
	}
	if len(values) > 1 {
		return "", "duplicate " + name + " headers: expected exactly one"
	}

	return strings.TrimSpace(values[0]), ""
}

// writeRouteDecisionError mirrors the reference decision server's error shape
// rather than this router's management envelope, because the caller reads this
// endpoint's failures against that contract.
func (s *ClassificationAPIServer) writeRouteDecisionError(
	w http.ResponseWriter,
	statusCode int,
	detail string,
) {
	s.writeJSONResponse(w, statusCode, map[string]interface{}{
		"detail": scrubSecretsInErrorMessage(detail),
	})
}

// logRouteDecision grounds divergence analysis: the executed model reaches the
// record here and nowhere else, which is the whole of its record-only role.
func (s *ClassificationAPIServer) logRouteDecision(
	decisionID string,
	executedModel string,
	decision routerruntime.RouteDecision,
) {
	logging.ComponentEvent("apiserver", "route_decision", map[string]interface{}{
		"decision_id":     decisionID,
		"selected_worker": decision.SelectedWorker,
		"worker_model":    decision.WorkerModel,
		"executed_model":  executedModel,
	})
}
