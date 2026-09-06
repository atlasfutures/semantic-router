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

// WithCount states what the request carried and what the limit is, in both
// halves of the refusal. The two numbers are what tell an operator whether the
// limit is wrong or the request is: the CP9r refusals of 2026-09-04 recorded
// only that something was too large, and explaining them took a code read and
// a client capture. Neither number is a value the client wrote.
func (err *ProtocolError) WithCount(unit string, observed, limit int) *ProtocolError {
	overflow := fmt.Sprintf("%d, limit %d", observed, limit)
	if unit != "" {
		overflow = fmt.Sprintf("%d %s, limit %d", observed, unit, limit)
	}
	err.Message += " (" + overflow + ")"
	if err.Cause == nil {
		err.Cause = errors.New(overflow)
		return err
	}
	err.Cause = fmt.Errorf("%w: %s", err.Cause, overflow)
	return err
}

// locateRefusal names where a refusal happened when it came from a validator
// the request leg shares with the response leg. Those validators see a text
// block or an image, never the message it sits in, so they cannot name the
// place themselves and the request caller has to.
//
// A refusal that already names a member keeps what it has.
func locateRefusal(err error, location, field string) error {
	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Parameter != "" {
		return err
	}
	located := NewFieldError(
		protocolError.Category, protocolError.Code, protocolError.Message, location, field,
	)
	if protocolError.Cause != nil {
		located.Cause = fmt.Errorf("%w: %w", located.Cause, protocolError.Cause)
	}
	located.RetryAfter = protocolError.RetryAfter
	return located
}
