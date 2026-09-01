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
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// envoyAttemptCountHeader is the only Envoy retry header the router reads.
// The retry policy itself is route configuration, not request state.
const envoyAttemptCountHeader = "x-envoy-attempt-count"

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
