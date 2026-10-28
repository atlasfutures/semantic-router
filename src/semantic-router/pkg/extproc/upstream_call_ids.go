package extproc

import (
	"net/http"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
)

// The provider's own ids for a call, so a turn can be traced at the provider
// (an OpenRouter generation lookup, an Anthropic request id) from the cell's
// llm_usage and turn_failed lines. Only ids the upstream sent are kept, each
// bounded and printable; no body, header value other than these ids, or key
// is read into them.

// maxUpstreamIDLength bounds every provider id the cell logs.
const maxUpstreamIDLength = 128

// upstreamGenerationIDHeader is OpenRouter's generation id header, when it
// sends one.
const upstreamGenerationIDHeader = "x-generation-id"

// upstreamRequestIDHeaders are the provider request id headers the cell
// records, in the order it prefers them: Anthropic's, the common one, then
// Cloudflare's ray id, which OpenRouter's edge stamps on every response.
var upstreamRequestIDHeaders = []string{"request-id", "x-request-id", "cf-ray"}

// upstreamCallIDs are the ids the current call's upstream stated.
type upstreamCallIDs struct {
	// generationID is the response's own id: the x-generation-id header, or
	// the id of the buffered response or of the stream's first chunk. On
	// OpenRouter it is the generation id.
	generationID string
	// requestID is a provider request id header, and requestIDHeader names
	// the header it came from.
	requestID       string
	requestIDHeader string
}

// upstreamCallIDsFromHeaders reads the ids off the call's response headers.
// An x-request-id equal to the caller's own is the cell's or the caller's
// echoed back, not the provider's, and is not recorded.
func upstreamCallIDsFromHeaders(headerMap *core.HeaderMap, callerRequestID string) upstreamCallIDs {
	var ids upstreamCallIDs
	if headerMap == nil {
		return ids
	}
	values := make(map[string]string, len(upstreamRequestIDHeaders)+1)
	for _, header := range headerMap.Headers {
		name := strings.ToLower(header.GetKey())
		if name != upstreamGenerationIDHeader && !isUpstreamRequestIDHeader(name) {
			continue
		}
		value := header.GetValue()
		if value == "" {
			value = string(header.GetRawValue())
		}
		value = boundedUpstreamID(value)
		if value == "" || name == "x-request-id" && value == boundedUpstreamID(callerRequestID) {
			continue
		}
		if _, seen := values[name]; !seen {
			values[name] = value
		}
	}
	ids.generationID = values[upstreamGenerationIDHeader]
	for _, name := range upstreamRequestIDHeaders {
		if value := values[name]; value != "" {
			ids.requestID, ids.requestIDHeader = value, name
			break
		}
	}
	return ids
}

// upstreamCallIDsFromHTTPHeader reads the ids off a response the cell
// received itself (a re-send), as upstreamCallIDsFromHeaders does.
func upstreamCallIDsFromHTTPHeader(header http.Header, callerRequestID string) upstreamCallIDs {
	headerMap := &core.HeaderMap{}
	for _, name := range append([]string{upstreamGenerationIDHeader}, upstreamRequestIDHeaders...) {
		if value := header.Get(name); value != "" {
			headerMap.Headers = append(headerMap.Headers, &core.HeaderValue{Key: name, RawValue: []byte(value)})
		}
	}
	return upstreamCallIDsFromHeaders(headerMap, callerRequestID)
}

func isUpstreamRequestIDHeader(name string) bool {
	for _, candidate := range upstreamRequestIDHeaders {
		if name == candidate {
			return true
		}
	}
	return false
}

// headers restates the ids as response headers, so a re-sent
// call's ids reach the response pipeline with its synthetic headers.
func (ids upstreamCallIDs) headers() []*core.HeaderValue {
	var headers []*core.HeaderValue
	if ids.generationID != "" {
		headers = append(headers, &core.HeaderValue{Key: upstreamGenerationIDHeader, RawValue: []byte(ids.generationID)})
	}
	if ids.requestID != "" {
		headers = append(headers, &core.HeaderValue{Key: ids.requestIDHeader, RawValue: []byte(ids.requestID)})
	}
	return headers
}

// observeUpstreamResponseID records the response's own id as the call's
// generation id, when no header named one. It is called with the id the
// upstream wrote, before the Router replaces it with a public one.
func observeUpstreamResponseID(ctx *RequestContext, responseID string) {
	if ctx == nil || ctx.UpstreamCallIDs.generationID != "" {
		return
	}
	ctx.UpstreamCallIDs.generationID = boundedUpstreamID(responseID)
}

func boundedUpstreamID(value string) string {
	return boundedPrintable(strings.TrimSpace(value), maxUpstreamIDLength)
}

// addUpstreamCallIDFields puts the call's provider ids and attempt count on
// a turn_failed line; an id nobody stated is null.
func addUpstreamCallIDFields(fields map[string]interface{}, ctx *RequestContext) {
	ids := ctx.UpstreamCallIDs
	fields["upstream_generation_id"] = nullableString(ids.generationID)
	fields["upstream_request_id"] = nullableString(ids.requestID)
	fields["upstream_request_id_header"] = nullableString(ids.requestIDHeader)
	fields["upstream_provider"] = nullableString(boundedUpstreamID(upstreamProviderOf(ctx)))
	if ctx.UpstreamAttempts > 0 {
		fields["upstream_attempts"] = ctx.UpstreamAttempts
	} else {
		fields["upstream_attempts"] = nil
	}
}

// upstreamProviderOf is the provider the upstream named for the turn: on its
// response, or on the stream state of a stream that left none.
func upstreamProviderOf(ctx *RequestContext) string {
	if response := attributedResponse(ctx); response != nil {
		return response.UpstreamProvider
	}
	if ctx != nil && ctx.SemanticStreamState != nil {
		return ctx.SemanticStreamState.provider
	}
	return ""
}

func nullableString(value string) interface{} {
	if value == "" {
		return nil
	}
	return value
}
