package extproc

import (
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The Anthropic Messages base version has not moved since 2023-06-01. The
// protocol moves through anthropic-beta instead, so the effective protocol
// version of a request is the base version plus the beta set it declares. The
// cell passed that header through Envoy and never looked at it, which is why
// every member that appeared -- caller, cache_control on a tool block,
// citations, evict_on_complete -- cost an investigation to attribute rather
// than a lookup.
//
// Recording it is the whole of rule 5. The set is logged and nothing else:
// nothing refuses on a beta, and nothing routes on one. The gateway does not
// replay a body-level 400, so refusing an unrecognised beta would end a user's
// turn over a header the cell has no opinion about, and accept-by-default
// exists precisely to stop that. The dated inventories under
// pkg/protocolcodec/testdata/inventory are where a beta is given a meaning,
// and they are read by CI, never by the router.

const anthropicBetaHeader = "anthropic-beta"

// anthropicBetaSet returns the beta values a request declared, sorted and with
// duplicates removed, so two requests declaring the same set log the same
// line. It answers nil when the header is absent or declares nothing: those
// are the same state and must not read as different ones.
func anthropicBetaSet(headers map[string]string) []string {
	raw, declared := lookupHeaderIgnoringCase(headers, anthropicBetaHeader)
	if !declared {
		return nil
	}
	seen := make(map[string]struct{})
	betas := make([]string, 0, strings.Count(raw, ",")+1)
	for _, value := range strings.Split(raw, ",") {
		beta := strings.TrimSpace(value)
		if beta == "" {
			continue
		}
		if _, repeated := seen[beta]; repeated {
			continue
		}
		seen[beta] = struct{}{}
		betas = append(betas, beta)
	}
	if len(betas) == 0 {
		return nil
	}
	sort.Strings(betas)
	return betas
}

// lookupHeaderIgnoringCase reads one header whatever case it was stored in.
// Envoy lowercases header names, but a RequestContext is also built by the
// looper path and by tests, and a beta set missed because of a capital letter
// would be indistinguishable from one never sent.
func lookupHeaderIgnoringCase(headers map[string]string, name string) (string, bool) {
	if value, exact := headers[name]; exact {
		return value, true
	}
	for key, value := range headers {
		if strings.EqualFold(key, name) {
			return value, true
		}
	}
	return "", false
}

// logIngressBetaSet writes the beta set of one accepted request. A request
// declaring none writes nothing: most traffic on the two OpenAI formats
// carries no such header, and a line per request saying "none" would be the
// noise the provider field report already avoids.
func logIngressBetaSet(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	betas := anthropicBetaSet(ctx.Headers)
	if len(betas) == 0 {
		return
	}
	logging.ComponentEvent("extproc", "ingress_beta_set", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"format":          string(ctx.SourceFormat),
		"anthropic_betas": strings.Join(betas, ","),
	})
}
