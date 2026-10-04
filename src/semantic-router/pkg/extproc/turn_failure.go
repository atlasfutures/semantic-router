package extproc

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Turn failure classes: why a turn did not deliver a served reply, in one
// bounded vocabulary shared with routerProxy's router_usage and Pathfinder's
// serving contract (fallback Phase 0, router-infra#96). They ride llm_usage's
// failure_class, the turn_failed line and a counter. The client header
// x-vsr-failure-class keeps its cell-level public set; the cell classes here
// refine "unavailable" in the logs only.
const (
	// The arm failed: the provider refused or could not serve the call.
	turnFailureRefusal         = "refusal"
	turnFailureRateLimited     = "rate_limited"
	turnFailureUpstream5xx     = "upstream_5xx"
	turnFailureTimeout         = "timeout"
	turnFailureNoEndpoint      = "no_endpoint"
	turnFailureContextOverflow = "context_overflow"
	turnFailureUpstreamError   = "upstream_error"
	turnFailureStreamCut       = "stream_cut"
	// The cell's own response check (jailbreak or hallucination) refused to
	// deliver a reply the arm produced; failure_detail names the check.
	turnFailureResponseBlocked = "response_blocked"
	// The cell failed before any call: the header's classes, and two
	// refinements of "unavailable".
	turnFailurePackageNotLoaded  = "package_not_loaded"
	turnFailureNoAvailableAction = "no_available_action"
)

// recordTurnFailure classes this turn's failure on its llm_usage line, logs
// the turn_failed line and counts it. contentSent says whether the client had
// already received model output when the failure arrived, which decides
// whether a later fallback could have re-served the turn unseen.
func recordTurnFailure(ctx *RequestContext, class string, contentSent bool) {
	recordTurnFailureDetail(ctx, class, "", contentSent)
}

// recordTurnFailureDetail is recordTurnFailure with the specific cause behind
// the bounded class, such as the protocol code of a reply the Router could not
// use. The class stays in the bounded vocabulary; the detail is diagnostic.
func recordTurnFailureDetail(ctx *RequestContext, class, detail string, contentSent bool) {
	if ctx == nil || class == "" {
		return
	}
	if class == turnFailureRefusal && ctx.SelectionSettlement.OutcomeClass != selectionOutcomeRefusal {
		// A refusal the provider reported as an error, at the status or in
		// the stream, is declined like a refusal it answered with: the turn
		// commits nothing and the fallback excludes its model.
		declineRefusedTurn(ctx)
	}
	ctx.ResponseFailureClass = class
	ctx.ResponseFailureDetail = detail
	ctx.ContentSentBeforeFailure = &contentSent
	fields := map[string]interface{}{
		"request_id":                  ctx.RequestID,
		"failure_class":               class,
		"content_sent_before_failure": contentSent,
		"failure_detail":              detail,
		"model":                       ctx.RequestModel,
		"upstream_status":             ctx.UpstreamStatusCode,
		"streaming":                   ctx.IsStreamingResponse,
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "turn_failed", fields)
	metrics.RecordTurnFailure(class)
	noteCellExclusion(ctx, class)
}

// upstreamFailureClass classes a provider's error by its status and what it
// said. A moderation rejection reported as an error (Alibaba's
// data_inspection_failed) is a refusal like any other.
func upstreamFailureClass(status int, protocolError *llmprotocol.ProtocolError) string {
	said := ""
	if protocolError != nil {
		said = strings.ToLower(protocolError.Code + " " + protocolError.Message)
	}
	switch {
	case strings.Contains(said, "data_inspection_failed"):
		return turnFailureRefusal
	case status == http.StatusTooManyRequests:
		return turnFailureRateLimited
	case status == http.StatusRequestTimeout || status == http.StatusGatewayTimeout:
		return turnFailureTimeout
	case strings.Contains(said, "no endpoints found"):
		return turnFailureNoEndpoint
	case (status == http.StatusBadRequest || status == http.StatusRequestEntityTooLarge) && saysContextOverflow(said):
		return turnFailureContextOverflow
	case status >= 500:
		return turnFailureUpstream5xx
	default:
		return turnFailureUpstreamError
	}
}

// streamFailureClass classes an error a provider raised mid-stream, which
// carries a category instead of a status.
func streamFailureClass(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return turnFailureUpstreamError
	}
	status := http.StatusBadGateway
	switch protocolError.Category {
	case llmprotocol.ErrorRateLimited:
		status = http.StatusTooManyRequests
	case llmprotocol.ErrorUpstreamTimeout:
		status = http.StatusGatewayTimeout
	case llmprotocol.ErrorInvalidRequest:
		status = http.StatusBadRequest
	case llmprotocol.ErrorNotFound:
		status = http.StatusNotFound
	}
	return upstreamFailureClass(status, protocolError)
}

func saysContextOverflow(said string) bool {
	for _, phrase := range []string{"context length", "context_length", "maximum context", "context window", "prompt is too long", "too many tokens"} {
		if strings.Contains(said, phrase) {
			return true
		}
	}
	return false
}

// contentBeforeRefusal reports model output beside a refusal: non-empty
// text, or any other kind of output the client received with it.
func contentBeforeRefusal(response *llmprotocol.Response) bool {
	if response == nil {
		return false
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			switch content.Kind {
			case llmprotocol.ContentRefusal:
			case llmprotocol.ContentText:
				if content.Text != "" {
					return true
				}
			default:
				// Any other output (a tool call, reasoning, an image, a file,
				// a carried block) reached the client whether or not it has
				// text.
				return true
			}
		}
	}
	return false
}

// selectionTurnFailureClass maps a selection failure's internal class onto
// the turn vocabulary: the public header class, with "unavailable" refined
// where the cause is the package or the catalog.
func selectionTurnFailureClass(internal string) string {
	switch internal {
	case "policy_service_package_not_loaded", "policy_package_mismatch":
		return turnFailurePackageNotLoaded
	case "policy_no_available_action", "policy_unbound_action", "policy_catalog_unbound", "policy_action_not_offered":
		return turnFailureNoAvailableAction
	default:
		return publicSelectionFailureClass(internal)
	}
}

// turnFailureIsBackendError reports a class that says the backend failed the
// call, which is what the load-balancing window counts as an error. A
// refusal, a blocked reply and the cell's own failures are not.
func turnFailureIsBackendError(class string) bool {
	switch class {
	case turnFailureRateLimited, turnFailureUpstream5xx, turnFailureTimeout,
		turnFailureNoEndpoint, turnFailureUpstreamError, turnFailureStreamCut:
		return true
	default:
		return false
	}
}

// maxStreamFailureDetail bounds the failure detail a stream error leaves on
// the usage line and the turn_failed line.
const maxStreamFailureDetail = 160

// codecStreamFailureDetail is the code and message of an error the Router's
// own codec raised when it rejected a provider stream, so a cut is
// diagnosable from the logs instead of reading as a bare class. The message
// is the codec's own text (the only provider-chosen part is a bounded event
// name), and it is still bounded to printable ASCII.
func codecStreamFailureDetail(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return ""
	}
	detail := protocolError.Code
	if protocolError.Message != "" {
		detail += ": " + protocolError.Message
	}
	return boundedPrintable(detail, maxStreamFailureDetail)
}

// providerStreamFailureDetail is only the code of an error a provider raised
// in its stream. Its message is the provider's free text, which may echo
// request content, and is never logged (pkg/observability/logging/content.go).
func providerStreamFailureDetail(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return ""
	}
	return boundedPrintable(protocolError.Code, maxStreamFailureDetail)
}

func boundedPrintable(text string, limit int) string {
	out := make([]byte, 0, len(text))
	for index := 0; index < len(text) && len(out) < limit; index++ {
		if character := text[index]; character >= 0x20 && character < 0x7f {
			out = append(out, character)
		}
	}
	return string(out)
}
