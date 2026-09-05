package llmprotocol

import (
	"errors"
	"fmt"
)

type ErrorCategory string

const (
	ErrorInvalidRequest      ErrorCategory = "invalid_request"
	ErrorAuthentication      ErrorCategory = "authentication"
	ErrorPermission          ErrorCategory = "permission"
	ErrorNotFound            ErrorCategory = "not_found"
	ErrorConflict            ErrorCategory = "conflict"
	ErrorUnsupportedFeature  ErrorCategory = "unsupported_feature"
	ErrorRateLimited         ErrorCategory = "rate_limited"
	ErrorUpstreamUnavailable ErrorCategory = "upstream_unavailable"
	ErrorUpstreamTimeout     ErrorCategory = "upstream_timeout"
	ErrorInternal            ErrorCategory = "internal"
)

type ProtocolError struct {
	Category   ErrorCategory
	Code       string
	Message    string
	Parameter  string
	RetryAfter int64
	Cause      error
}

// TransportError is the protocol-neutral representation of an HTTP error
// response. It is deliberately separate from Response.Error: the latter is a
// model-generation result (for example, a Responses resource whose status is
// "failed"), while TransportError represents a non-2xx API response.
type TransportError struct {
	Error             *ProtocolError
	ProviderRequestID string
}

func (err *ProtocolError) Error() string {
	if err == nil {
		return ""
	}
	message := err.Message
	if err.Code != "" {
		message = fmt.Sprintf("%s: %s", err.Code, err.Message)
	}
	// The cause names which member or value failed. Without it a decode
	// failure reads only as "some field was non-canonical", which is not
	// enough to find the field from a log line. Causes carry member names,
	// Go types and JSON offsets; they never carry a decoded value.
	if err.Cause != nil {
		message += ": " + err.Cause.Error()
	}
	return message
}

func (err *ProtocolError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Cause
}

func NewError(category ErrorCategory, code, message string, cause error) *ProtocolError {
	return &ProtocolError{Category: category, Code: code, Message: message, Cause: cause}
}

// NewFieldError builds a refusal that names where it happened in both places
// anyone looks. The cause reaches the ingress_request_refused line; the
// message reaches the client, because a client error body carries the message
// and nothing else.
//
// Both are needed. An operator cannot find a block in a conversation of
// hundreds from a feature name, and the refused body is never stored. And a
// client repairs its own request only from what the body says: the 2.1.260
// Claude Code retries without a beta field when, and only when, the 400 body
// names that field.
//
// The detail is contract text -- a block type, an index, a field path -- and
// never a value the client wrote.
func NewFieldError(category ErrorCategory, code, message, location, field string) *ProtocolError {
	detail := fmt.Sprintf("field %q", field)
	if location != "" {
		detail = fmt.Sprintf("%s: field %q", location, field)
	}
	return &ProtocolError{
		Category: category, Code: code,
		Message:   message + ": " + detail,
		Parameter: field,
		Cause:     errors.New(detail),
	}
}
