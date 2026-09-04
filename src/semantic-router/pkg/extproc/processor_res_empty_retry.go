package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// emptyOutputItemCode is the refusal a completion with no content is raised
// with. See llmprotocol.validateOutputItem.
const emptyOutputItemCode = "empty_output_item"

// upstreamEmptyRetryTimeout bounds the second attempt. The first one already
// spent the turn's latency budget, and a client that has been waiting is not
// served by waiting again for as long.
const upstreamEmptyRetryTimeout = 120 * time.Second

// upstreamEmptyRetryBodyLimit caps what the retry will read back. It is the
// same order as a buffered completion the data plane already carries.
const upstreamEmptyRetryBodyLimit = 32 << 20

var upstreamEmptyRetryClient = &http.Client{Timeout: upstreamEmptyRetryTimeout}

// upstreamEmptyRetryPlan is what a second attempt needs: the bytes that were
// dispatched and where they went. It is kept only for a buffered turn, because
// a streamed one has already sent frames to the client by the time an empty
// completion is visible.
type upstreamEmptyRetryPlan struct {
	body    []byte
	profile *config.ProviderProfile
	model   string
	backend string
}

// retainUpstreamEmptyRetryPlan keeps the dispatched request beside the turn.
//
// The Router does not otherwise hold the bytes it rendered: Envoy carries them
// and the response phase sees only what came back. A retry has to send the
// identical request, so it is kept here, at the one boundary where the wire is
// known.
func retainUpstreamEmptyRetryPlan(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.UpstreamEmptyRetry = nil
	if dispatch == nil || ctx.ExpectStreamingResponse ||
		dispatch.targetFormat != llmprotocol.OpenAIChatV1 || dispatch.profile == nil {
		return
	}
	ctx.UpstreamEmptyRetry = &upstreamEmptyRetryPlan{
		body:    append([]byte(nil), body...),
		profile: dispatch.profile,
		model:   dispatch.logicalModel,
		backend: dispatch.backendName,
	}
}

// retryEmptyUpstreamCompletion re-issues, once, a turn the upstream answered
// with nothing.
//
// The condition is narrow on purpose. The response has to have decoded and
// then been refused for naming no content, and the upstream has to have billed
// at most one token for it: a turn that generated something was refused for
// another reason, and re-running it would pay twice and could discard a real
// answer. The upstream also has to have named the provider that served it,
// because the retry's only lever is telling the aggregator to use a different
// one, and there is nothing to exclude when nobody was named.
//
// It returns the retry's body and response, or nil when there was no retry to
// make or it produced nothing better. The caller fails on the original refusal
// in both of those cases -- a retry never turns a failure into a worse one.
func (r *OpenAIRouter) retryEmptyUpstreamCompletion(
	ctx *RequestContext,
	decodeErr error,
) ([]byte, *llmprotocol.Response) {
	if r == nil || ctx == nil || ctx.UpstreamEmptyRetry == nil || ctx.UpstreamEmptyRetries > 0 ||
		!isEmptyUpstreamCompletion(ctx.UpstreamDecodedRemnant, decodeErr) {
		return nil, nil
	}
	excluded := ctx.UpstreamDecodedRemnant.UpstreamProvider
	// The empty was billed whether or not the retry answers, so its counts are
	// carried before the retry overwrites the remnant they live in.
	ctx.UpstreamEmptyRetryPriorUsage = responseUsageFromSemanticUsage(ctx.UpstreamDecodedRemnant.Usage)
	ctx.UpstreamEmptyRetries = 1

	body, err := r.issueUpstreamEmptyRetry(ctx, ctx.UpstreamEmptyRetry, excluded)
	if err != nil {
		logUpstreamEmptyRetry(ctx, excluded, "", "transport_error")
		return nil, nil
	}
	response, decodeRetryErr := r.decodeClientResponse(body, ctx)
	if decodeRetryErr != nil {
		outcome := "decode_failed"
		if isEmptyUpstreamCompletion(ctx.UpstreamDecodedRemnant, decodeRetryErr) {
			outcome = "empty"
		}
		logUpstreamEmptyRetry(ctx, excluded, remnantProvider(ctx), outcome)
		return nil, nil
	}
	logUpstreamEmptyRetry(ctx, excluded, response.UpstreamProvider, "answered")
	return body, response
}

// isEmptyUpstreamCompletion reports the one outcome a retry answers: a
// response the codec decoded, refused for an output item that names no
// content, billed for no more than a stop token, and attributed to a provider
// the retry can exclude.
func isEmptyUpstreamCompletion(remnant *llmprotocol.Response, err error) bool {
	if remnant == nil || remnant.UpstreamProvider == "" {
		return false
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != emptyOutputItemCode {
		return false
	}
	completion := remnant.Usage.OutputTotal
	return completion.Value != nil && *completion.Value <= 1
}

func remnantProvider(ctx *RequestContext) string {
	if ctx == nil || ctx.UpstreamDecodedRemnant == nil {
		return ""
	}
	return ctx.UpstreamDecodedRemnant.UpstreamProvider
}

func logUpstreamEmptyRetry(ctx *RequestContext, excluded, retryProvider, outcome string) {
	logging.ComponentWarnEvent("extproc", "upstream_empty_retry", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"arm":               ctx.VSRSelectedModel,
		"excluded_provider": excluded,
		"retry_provider":    retryProvider,
		"outcome":           outcome,
	})
}

// issueUpstreamEmptyRetry sends the retry itself. The data plane dispatched
// the first attempt, but it has already delivered its answer by now, so the
// second one leaves from here.
func (r *OpenAIRouter) issueUpstreamEmptyRetry(
	ctx *RequestContext,
	plan *upstreamEmptyRetryPlan,
	excluded string,
) ([]byte, error) {
	body, err := excludeProviderFromRequest(plan.body, excluded)
	if err != nil {
		return nil, err
	}
	endpoint, err := upstreamChatEndpoint(plan.profile)
	if err != nil {
		return nil, err
	}
	callContext, cancel := context.WithTimeout(retryParentContext(ctx), upstreamEmptyRetryTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(callContext, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("content-type", "application/json")
	if err := r.authorizeUpstreamEmptyRetry(request, plan, ctx); err != nil {
		return nil, err
	}
	response, err := upstreamEmptyRetryClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	payload, err := io.ReadAll(io.LimitReader(response.Body, upstreamEmptyRetryBodyLimit))
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("upstream retry returned status %d", response.StatusCode)
	}
	return payload, nil
}

// retryParentContext keeps the retry tied to the client's request, so a
// disconnect cancels it rather than leaving it running against the provider.
func retryParentContext(ctx *RequestContext) context.Context {
	if ctx == nil || ctx.TraceContext == nil {
		return context.Background()
	}
	return ctx.TraceContext
}

// authorizeUpstreamEmptyRetry puts the same credential on the retry that the
// request phase put on the first attempt. A resolver that cannot answer stops
// the retry: sending the provider an unauthenticated copy of the turn is not a
// second attempt, it is a second failure.
func (r *OpenAIRouter) authorizeUpstreamEmptyRetry(
	request *http.Request,
	plan *upstreamEmptyRetryPlan,
	ctx *RequestContext,
) error {
	header, prefix, err := plan.profile.ResolveAuthHeader()
	if err != nil {
		return err
	}
	if r.CredentialResolver == nil {
		return nil
	}
	key, err := r.CredentialResolver.KeyForProvider(providerForProfile(plan.profile), plan.model, ctx.Headers)
	if err != nil {
		return err
	}
	if key == "" {
		return nil
	}
	if prefix != "" {
		key = prefix + " " + key
	}
	request.Header.Set(header, key)
	return nil
}

func providerForProfile(profile *config.ProviderProfile) authz.LLMProvider {
	provider, _, _, err := resolveProviderAuth(profile)
	if err != nil {
		return ""
	}
	return provider
}

// upstreamChatEndpoint is the absolute URL the first attempt was sent to. The
// request phase splits the same answer in two -- an Envoy cluster for the
// origin and a :path header for the rest -- and this rejoins them.
//
// ResolveChatPath already carries the base URL's own path, because that is
// what the :path header has to hold. So the retry resolves it against the
// origin and never appends that path twice.
func upstreamChatEndpoint(profile *config.ProviderProfile) (string, error) {
	if profile == nil || strings.TrimSpace(profile.BaseURL) == "" {
		return "", errors.New("provider profile has no base URL")
	}
	base, err := url.Parse(profile.BaseURL)
	if err != nil {
		return "", err
	}
	path, err := profile.ResolveChatPath()
	if err != nil {
		return "", err
	}
	target, err := url.Parse(path)
	if err != nil {
		return "", err
	}
	return base.ResolveReference(&url.URL{Path: target.Path, RawQuery: target.RawQuery}).String(), nil
}

// excludeProviderFromRequest adds OpenRouter's provider preference that keeps
// the retry off the upstream that answered nothing.
//
// The list is merged rather than replaced: a request that already excludes
// providers keeps those exclusions, and a repeated name is not added twice.
// Read 2026-09-04: https://openrouter.ai/docs/features/provider-routing
func excludeProviderFromRequest(body []byte, provider string) ([]byte, error) {
	requestMap := map[string]json.RawMessage{}
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, err
	}
	preferences := map[string]json.RawMessage{}
	if existing, present := requestMap["provider"]; present {
		if json.Unmarshal(existing, &preferences) != nil {
			preferences = map[string]json.RawMessage{}
		}
	}
	ignored := []string{}
	if raw, present := preferences["ignore"]; present {
		_ = json.Unmarshal(raw, &ignored)
	}
	for _, name := range ignored {
		if name == provider {
			provider = ""
			break
		}
	}
	if provider != "" {
		ignored = append(ignored, provider)
	}
	encodedIgnore, err := json.Marshal(ignored)
	if err != nil {
		return nil, err
	}
	preferences["ignore"] = encodedIgnore
	encodedPreferences, err := json.Marshal(preferences)
	if err != nil {
		return nil, err
	}
	requestMap["provider"] = encodedPreferences
	return json.Marshal(requestMap)
}
