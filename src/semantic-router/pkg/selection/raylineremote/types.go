// Package raylineremote implements the bounded, versioned client contract for
// Pathfinder-owned Rayline route selection.
package raylineremote

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"
)

const (
	TransactionSchemaVersion  = "rayline-router.selection-transaction.v1"
	CapabilitiesSchemaVersion = "rayline-router.selection-capabilities.v1"
	OpenAIChatProtocol        = "openai.chat.completions"

	maxWireBytes       = 1_000_000
	maxDecisionIDBytes = 200
	maxCandidates      = 128
	maxMessages        = 2_048
	maxTools           = 256
)

var (
	decisionIDPattern = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,200}$`)
	episodeKeyPattern = regexp.MustCompile(`^hmac-sha256:[0-9a-f]{64}$`)
	receiptPattern    = regexp.MustCompile(`^[A-Za-z0-9_-]{1,512}$`)
	errorCodePattern  = regexp.MustCompile(`^[a-z][a-z0-9_]{0,79}$`)
)

// FailureClass is a bounded failure category safe for logs and metrics.
type FailureClass string

const (
	FailureRequest   FailureClass = "request"
	FailureTimeout   FailureClass = "timeout"
	FailureTransport FailureClass = "transport"
	FailureStatus    FailureClass = "status"
	FailureDecode    FailureClass = "decode"
	FailureContract  FailureClass = "contract"
	FailureLease     FailureClass = "lease"
	FailureState     FailureClass = "state"
)

// Failure intentionally excludes response bodies, prompts, credentials,
// receipts, episode identifiers, and worker identifiers.
type Failure struct {
	Class      FailureClass
	Operation  string
	Code       string
	StatusCode int
}

func (failure *Failure) Error() string {
	if failure == nil {
		return "Rayline remote operation failed"
	}
	return fmt.Sprintf(
		"Rayline remote operation failed (operation=%s,class=%s,code=%s,status=%d)",
		failure.Operation,
		failure.Class,
		failure.Code,
		failure.StatusCode,
	)
}

func newFailure(
	class FailureClass,
	operation string,
	code string,
	statusCode int,
) *Failure {
	if !errorCodePattern.MatchString(code) {
		code = "unspecified"
	}
	return &Failure{
		Class:      class,
		Operation:  operation,
		Code:       code,
		StatusCode: statusCode,
	}
}

// DeriveEpisodeKey converts an authenticated ingress episode identifier into
// the only episode identity allowed to cross the Pathfinder boundary.
func DeriveEpisodeKey(rawEpisodeID string, key []byte) (string, error) {
	if strings.TrimSpace(rawEpisodeID) == "" ||
		len(rawEpisodeID) > 4_096 {
		return "", newFailure(
			FailureRequest,
			"prepare",
			"invalid_episode_id",
			0,
		)
	}
	if len(key) < 32 {
		return "", newFailure(
			FailureRequest,
			"prepare",
			"invalid_hmac_key",
			0,
		)
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(rawEpisodeID))
	return "hmac-sha256:" + hex.EncodeToString(mac.Sum(nil)), nil
}

// MintDecisionID creates an unguessable request idempotency key. Client input
// is never used as the transaction key unless a separately trusted ingress
// contract has validated and stamped it.
func MintDecisionID() (string, error) {
	random := make([]byte, 24)
	if _, err := rand.Read(random); err != nil {
		return "", newFailure(
			FailureRequest,
			"prepare",
			"decision_id_entropy",
			0,
		)
	}
	return "rt_" + base64.RawURLEncoding.EncodeToString(random), nil
}

// ChatRequest is the allowlisted policy input sent to Pathfinder. Provider
// headers, credentials, endpoint data, and executable request mutations have
// no representation in this type.
type ChatRequest struct {
	Protocol string            `json:"protocol"`
	Messages []json.RawMessage `json:"messages"`
	Tools    []json.RawMessage `json:"tools,omitempty"`
}

// ChatRequestFromBody extracts only messages and tools from a validated OpenAI
// Chat Completions body. In particular, the client-controlled model field is
// omitted so it cannot activate Pathfinder's direct worker-pinning behavior.
func ChatRequestFromBody(body []byte) (ChatRequest, error) {
	if len(body) == 0 || len(body) > maxWireBytes {
		return ChatRequest{}, newFailure(
			FailureRequest,
			"prepare",
			"invalid_chat_body",
			0,
		)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil ||
		envelope == nil {
		return ChatRequest{}, newFailure(
			FailureRequest,
			"prepare",
			"invalid_chat_json",
			0,
		)
	}
	messages, err := decodeObjectArray(
		envelope["messages"],
		1,
		maxMessages,
	)
	if err != nil {
		return ChatRequest{}, newFailure(
			FailureRequest,
			"prepare",
			"invalid_messages",
			0,
		)
	}
	var tools []json.RawMessage
	if rawTools, ok := envelope["tools"]; ok &&
		string(rawTools) != "null" {
		tools, err = decodeObjectArray(rawTools, 0, maxTools)
		if err != nil {
			return ChatRequest{}, newFailure(
				FailureRequest,
				"prepare",
				"invalid_tools",
				0,
			)
		}
	}
	request := ChatRequest{
		Protocol: OpenAIChatProtocol,
		Messages: messages,
		Tools:    tools,
	}
	if encoded, err := json.Marshal(request); err != nil ||
		len(encoded) > maxWireBytes {
		return ChatRequest{}, newFailure(
			FailureRequest,
			"prepare",
			"chat_request_too_large",
			0,
		)
	}
	return request, nil
}

func validateChatRequest(request ChatRequest) error {
	if request.Protocol != OpenAIChatProtocol ||
		len(request.Messages) == 0 ||
		len(request.Messages) > maxMessages ||
		len(request.Tools) > maxTools {
		return errors.New("invalid chat request")
	}
	for _, collection := range [][]json.RawMessage{
		request.Messages,
		request.Tools,
	} {
		for _, value := range collection {
			var object map[string]json.RawMessage
			if json.Unmarshal(value, &object) != nil || object == nil {
				return errors.New("chat member is not an object")
			}
		}
	}
	encoded, err := json.Marshal(request)
	if err != nil || len(encoded) > maxWireBytes {
		return errors.New("chat request exceeds wire contract")
	}
	return nil
}

func decodeObjectArray(
	raw json.RawMessage,
	minimum int,
	maximum int,
) ([]json.RawMessage, error) {
	var values []json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &values) != nil ||
		len(values) < minimum || len(values) > maximum {
		return nil, errors.New("invalid object array")
	}
	for _, value := range values {
		var object map[string]json.RawMessage
		if json.Unmarshal(value, &object) != nil || object == nil {
			return nil, errors.New("array member is not an object")
		}
	}
	return values, nil
}

// PrepareInput is a complete, idempotent selection request.
type PrepareInput struct {
	DecisionID string
	EpisodeKey string
	Candidates []string
	Request    ChatRequest
}

// PrepareResult exposes selection facts while keeping the receipt owned by the
// returned Transaction.
type PrepareResult struct {
	SelectedWorker string
	RouteCallIndex int
	Policy         string
}

type AbortReason string

const (
	AbortClientCancelled AbortReason = "client_cancelled"
	AbortDispatchFailure AbortReason = "dispatch_failure"
	AbortProviderNon2xx  AbortReason = "provider_non_2xx"
	AbortProviderNetwork AbortReason = "provider_transport"
	AbortRequestFailure  AbortReason = "request_failure"
	AbortSelectionFailed AbortReason = "selection_failure"
)

func validAbortReason(reason AbortReason) bool {
	switch reason {
	case AbortClientCancelled,
		AbortDispatchFailure,
		AbortProviderNon2xx,
		AbortProviderNetwork,
		AbortRequestFailure,
		AbortSelectionFailed:
		return true
	default:
		return false
	}
}

type OutcomeClass string

const (
	OutcomeSuccess         OutcomeClass = "success"
	OutcomeUpstreamError   OutcomeClass = "upstream_error"
	OutcomeClientCancelled OutcomeClass = "client_cancelled"
	OutcomeStreamError     OutcomeClass = "stream_error"
	OutcomeUnknown         OutcomeClass = "unknown"
)

// ActualOutcome carries only bounded execution facts. Pointer fields preserve
// "unknown" instead of fabricating measured zero values.
type ActualOutcome struct {
	OutcomeClass OutcomeClass `json:"outcome_class"`
	StatusCode   int          `json:"status_code"`
	InputTokens  *int         `json:"input_tokens,omitempty"`
	OutputTokens *int         `json:"output_tokens,omitempty"`
	LatencyMS    *float64     `json:"latency_ms,omitempty"`
	CostUSD      *float64     `json:"cost_usd,omitempty"`
	ErrorType    *string      `json:"error_type,omitempty"`
}

func validateOutcome(outcome ActualOutcome) error {
	if !validOutcomeClass(outcome.OutcomeClass) {
		return settleValidationFailure("invalid_outcome_class")
	}
	if outcome.StatusCode != 0 &&
		(outcome.StatusCode < 100 || outcome.StatusCode > 599) {
		return settleValidationFailure("invalid_status_code")
	}
	if !validTokenCount(outcome.InputTokens) ||
		!validTokenCount(outcome.OutputTokens) {
		return settleValidationFailure("invalid_token_count")
	}
	if !validOptionalMetric(outcome.LatencyMS, 86_400_000) {
		return settleValidationFailure("invalid_latency")
	}
	if !validOptionalMetric(outcome.CostUSD, 1_000_000) {
		return settleValidationFailure("invalid_cost")
	}
	if outcome.ErrorType != nil &&
		len(*outcome.ErrorType) > 160 {
		return settleValidationFailure("invalid_error_type")
	}
	return nil
}

func validOutcomeClass(outcomeClass OutcomeClass) bool {
	switch outcomeClass {
	case OutcomeSuccess,
		OutcomeUpstreamError,
		OutcomeClientCancelled,
		OutcomeStreamError,
		OutcomeUnknown:
		return true
	default:
		return false
	}
}

func validTokenCount(value *int) bool {
	return value == nil || (*value >= 0 && *value <= 100_000_000)
}

func validOptionalMetric(value *float64, maximum float64) bool {
	return value == nil ||
		(!math.IsNaN(*value) &&
			!math.IsInf(*value, 0) &&
			*value >= 0 &&
			*value <= maximum)
}

func settleValidationFailure(reason string) error {
	return newFailure(FailureRequest, "settle", reason, 0)
}
