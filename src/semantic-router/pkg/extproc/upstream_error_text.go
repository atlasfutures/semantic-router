package extproc

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// The text of a provider's 4xx, for the cell's own logs.
//
// A 400 that names its cause only in its text was logged as a bare status:
// OpenRouter's "encrypted reasoning or compaction content that can only be
// processed by the provider that created it (anthropic) or by azure/global,
// claude-on-aws, ..." on prod. So turn_failed and llm_usage carry the
// provider's error message, and the provider slugs OpenRouter states in the
// error's metadata (pinned_provider_slug, accepting_provider_slugs).
//
// Only those: never the request, the metadata's raw provider body, or any
// other member. The message is redacted as the client's copy is
// (llmprotocol.RedactProviderText), its links are dropped, its whitespace is
// folded, and it is cut at upstreamErrorTextMax bytes. An account refusal
// (401, 402) is left out: its text names the cell's provider account.

const (
	upstreamErrorTextMax  = 300
	upstreamErrorSlugMax  = 64
	upstreamErrorSlugsMax = 16
)

var (
	upstreamErrorLinkPattern = regexp.MustCompile(`https?://\S+`)
	upstreamErrorSlugPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)
)

// upstreamErrorText is what a provider's 4xx said, as the logs may carry it.
type upstreamErrorText struct {
	message        string
	pinnedSlug     string
	acceptingSlugs []string
}

// readUpstreamErrorText takes the loggable text of a 4xx: the decoded error's
// message, else the body's error.message, and the metadata's provider slugs.
func readUpstreamErrorText(status int, body []byte, protocolError *llmprotocol.ProtocolError) upstreamErrorText {
	if status < 400 || status >= 500 || status == http.StatusUnauthorized || status == http.StatusPaymentRequired {
		return upstreamErrorText{}
	}
	var wire struct {
		Error struct {
			Message  string `json:"message"`
			Metadata struct {
				PinnedProviderSlug     string   `json:"pinned_provider_slug"`
				AcceptingProviderSlugs []string `json:"accepting_provider_slugs"`
			} `json:"metadata"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &wire)
	message := wire.Error.Message
	if protocolError != nil && protocolError.Message != "" {
		message = protocolError.Message
	}
	text := upstreamErrorText{
		message:    loggableUpstreamMessage(message),
		pinnedSlug: loggableSlug(wire.Error.Metadata.PinnedProviderSlug),
	}
	for _, slug := range wire.Error.Metadata.AcceptingProviderSlugs {
		if len(text.acceptingSlugs) == upstreamErrorSlugsMax {
			break
		}
		if slug = loggableSlug(slug); slug != "" {
			text.acceptingSlugs = append(text.acceptingSlugs, slug)
		}
	}
	return text
}

func loggableUpstreamMessage(message string) string {
	message = llmprotocol.RedactProviderText(message)
	message = upstreamErrorLinkPattern.ReplaceAllString(message, "[link]")
	message = strings.Join(strings.Fields(message), " ")
	if len(message) <= upstreamErrorTextMax {
		return message
	}
	cut := upstreamErrorTextMax
	for cut > 0 && !utf8.RuneStart(message[cut]) {
		cut--
	}
	return message[:cut]
}

// loggableSlug is a provider slug, or "" for anything that is not one.
func loggableSlug(slug string) string {
	slug = strings.TrimSpace(slug)
	if len(slug) > upstreamErrorSlugMax || !upstreamErrorSlugPattern.MatchString(slug) {
		return ""
	}
	return slug
}

// addUpstreamErrorTextFields writes a 4xx's text into a log record, leaving
// out what it did not say.
func addUpstreamErrorTextFields(fields map[string]interface{}, text upstreamErrorText) {
	if text.message != "" {
		fields["upstream_error_message"] = text.message
	}
	if text.pinnedSlug != "" {
		fields["upstream_error_pinned_provider_slug"] = text.pinnedSlug
	}
	if len(text.acceptingSlugs) > 0 {
		fields["upstream_error_accepting_provider_slugs"] = text.acceptingSlugs
	}
}
