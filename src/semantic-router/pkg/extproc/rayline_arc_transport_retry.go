/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"math"
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	envoyAttemptCountHeader       = "x-envoy-attempt-count"
	envoyMaxRetriesHeader         = "x-envoy-max-retries"
	envoyRetryOnHeader            = "x-envoy-retry-on"
	envoyRetryGRPCOnHeader        = "x-envoy-retry-grpc-on"
	envoyRetriableStatusHeader    = "x-envoy-retriable-status-codes"
	envoyRetriableHeadersHeader   = "x-envoy-retriable-header-names"
	envoyRequestTimeoutHeader     = "x-envoy-upstream-rq-timeout-ms"
	envoyPerTryTimeoutHeader      = "x-envoy-upstream-rq-per-try-timeout-ms"
	envoyHedgePerTryTimeoutHeader = "x-envoy-hedge-on-per-try-timeout"
)

var envoyRetryControlHeaders = []string{
	envoyMaxRetriesHeader,
	envoyRetryOnHeader,
	envoyRetryGRPCOnHeader,
	envoyRetriableStatusHeader,
	envoyRetriableHeadersHeader,
	envoyRequestTimeoutHeader,
	envoyPerTryTimeoutHeader,
	envoyHedgePerTryTimeoutHeader,
}

// enforceRaylineARCTransportRetryHeaders makes the signed worker contract the
// last writer of Envoy retry controls. The OpenRouter-only route topology owns
// the retryable statuses and backoff algorithm; these request headers supply
// the signed artifact's attempt/deadline budget. Envoy evaluates
// retriable_request_headers before request-body ext_proc mutations, so the
// deployment boundary—not a late header matcher—selects the retrying route.
//
// OpenAI-compatible self-hosted workers intentionally use routes with no retry
// policy. A 429 from a vLLM worker is server backpressure, and replaying it
// automatically can amplify overload. The caller must never be able to opt
// itself in or increase the artifact-owned retry budget.
func enforceRaylineARCTransportRetryHeaders(
	state *routeHeaderState,
	ctx *RequestContext,
) {
	if state == nil || ctx == nil || ctx.RaylineARCDispatch == nil {
		return
	}

	for _, name := range envoyRetryControlHeaders {
		state.setHeaders = removeSetHeader(state.setHeaders, name)
		state.removeHeaders = appendHeaderRemoval(state.removeHeaders, name)
	}

	worker := ctx.RaylineARCDispatch
	if worker.EffectiveDispatchBackend() != raylinearc.DispatchOpenRouter {
		return
	}

	state.setHeaders = append(
		state.setHeaders,
		overwriteHeader(envoyRetryOnHeader, "retriable-status-codes"),
		overwriteHeader(
			envoyMaxRetriesHeader,
			strconv.FormatUint(worker.OpenRouterMaxRetries, 10),
		),
	)
	if worker.AttemptDeadlineSeconds != nil {
		milliseconds := int64(math.Ceil(*worker.AttemptDeadlineSeconds * 1000))
		if milliseconds > 0 {
			state.setHeaders = append(
				state.setHeaders,
				overwriteHeader(
					envoyRequestTimeoutHeader,
					strconv.FormatInt(milliseconds, 10),
				),
			)
		}
	}
}

func overwriteHeader(name string, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      name,
			RawValue: []byte(value),
		},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	}
}

func removeSetHeader(
	headers []*core.HeaderValueOption,
	name string,
) []*core.HeaderValueOption {
	retained := headers[:0]
	for _, header := range headers {
		if strings.EqualFold(header.GetHeader().GetKey(), name) {
			continue
		}
		retained = append(retained, header)
	}
	return retained
}

func appendHeaderRemoval(headers []string, name string) []string {
	for _, header := range headers {
		if strings.EqualFold(header, name) {
			return headers
		}
	}
	return append(headers, name)
}

func captureRaylineARCProviderAttempts(
	headers *core.HeaderMap,
	ctx *RequestContext,
	statusCode int,
) {
	if ctx == nil ||
		ctx.RaylineARCDispatch == nil ||
		ctx.RaylineARCDispatch.EffectiveDispatchBackend() != raylinearc.DispatchOpenRouter {
		return
	}

	attempts := envoyAttemptCount(headers)
	ctx.UpstreamAttemptCount = attempts
	if attempts > 1 {
		ctx.UpstreamRetryCount = attempts - 1
	}
	ctx.UpstreamRetryExhausted = (statusCode == 429 || statusCode == 503) &&
		ctx.UpstreamRetryCount >= ctx.RaylineARCDispatch.OpenRouterMaxRetries

	outcome := "failed"
	if statusCode >= 200 && statusCode < 300 {
		outcome = "success"
	} else if ctx.UpstreamRetryExhausted {
		outcome = "retry_exhausted"
	}
	metrics.RecordRaylineARCProviderRequest(
		outcome,
		statusCode,
		attempts,
		ctx.UpstreamRetryExhausted,
	)
}

func envoyAttemptCount(headers *core.HeaderMap) uint64 {
	if headers == nil {
		return 1
	}
	for _, header := range headers.Headers {
		if !strings.EqualFold(header.GetKey(), envoyAttemptCountHeader) {
			continue
		}
		value := extractHeaderValue(header)
		attempts, err := strconv.ParseUint(value, 10, 64)
		if err == nil && attempts > 0 {
			return attempts
		}
	}
	return 1
}
