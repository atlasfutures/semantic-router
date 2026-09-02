package llmprotocol

import "fmt"

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
