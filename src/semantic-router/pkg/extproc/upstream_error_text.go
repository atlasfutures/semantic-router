package extproc

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"unicode"
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
// (llmprotocol.RedactProviderText), its links are dropped, every quoted span
// is replaced by [quoted] (a provider that echoes a request value quotes it:
// "Invalid value: 'x'"), its whitespace is folded, and it is cut at
// upstreamErrorTextMax bytes. An echo the provider did not quote is not
// caught; the cut bounds it. An account refusal (401, 402) is left out: its
// text names the cell's provider account. A slug is kept only when it has a
// provider slug's shape and the redactor leaves it as it is.

const (
	upstreamErrorTextMax  = 300
	upstreamErrorSlugMax  = 48
	upstreamErrorSlugsMax = 16
)

var (
	upstreamErrorLinkPattern = regexp.MustCompile(`https?://\S+`)
	upstreamErrorSlugPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*(/[a-z0-9][a-z0-9.-]*)*$`)
	// upstreamErrorSecretPattern is a credential-shaped token anywhere in a
	// word (sk-..., rk_..., or any long unbroken token), which the shared
	// redactor's word boundary can miss.
	upstreamErrorSecretPattern = regexp.MustCompile(`(?i)(?:sk|rk|pk)[-_][A-Za-z0-9_-]{6,}|[A-Za-z0-9_-]{32,}`)
	upstreamErrorQuoteClosers  = map[rune]rune{'"': '"', '`': '`', '\'': '\'', '“': '”', '‘': '’'}
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
	message = withoutQuotedSpans(message)
	message = llmprotocol.RedactProviderText(message)
	message = upstreamErrorSecretPattern.ReplaceAllString(message, "[redacted]")
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

// withoutQuotedSpans replaces each quoted span with [quoted]. An apostrophe
// inside a word (can't) opens nothing; a backslash escapes the next rune; a
// span left open runs to the end of the message, so no part of a quoted
// value is ever kept.
func withoutQuotedSpans(message string) string {
	runes := []rune(message)
	var out strings.Builder
	isWord := func(index int) bool {
		return index >= 0 && index < len(runes) && (unicode.IsLetter(runes[index]) || unicode.IsDigit(runes[index]))
	}
	for index := 0; index < len(runes); index++ {
		closer, opens := upstreamErrorQuoteClosers[runes[index]]
		if !opens || runes[index] == '\'' && isWord(index-1) {
			out.WriteRune(runes[index])
			continue
		}
		end := -1
		for next := index + 1; next < len(runes); next++ {
			if runes[next] == '\\' {
				next++
				continue
			}
			if runes[next] == closer && (closer != '\'' || !isWord(next+1)) {
				end = next
				break
			}
		}
		out.WriteString("[quoted]")
		if end < 0 {
			break
		}
		index = end
	}
	return out.String()
}

// loggableSlug is a provider slug, or "" for anything that is not one.
func loggableSlug(slug string) string {
	slug = strings.ToLower(strings.TrimSpace(slug))
	if len(slug) > upstreamErrorSlugMax || !upstreamErrorSlugPattern.MatchString(slug) ||
		upstreamErrorSecretPattern.MatchString(slug) || llmprotocol.RedactProviderText(slug) != slug {
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
