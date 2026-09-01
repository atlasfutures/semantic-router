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
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestCaptureRaylineARCProviderAttemptsSeparatesLogicalAndWireCounts(
	t *testing.T,
) {
	logicalBefore := testutil.ToFloat64(
		metrics.RaylineARCProviderLogicalRequests.WithLabelValues("success"),
	)
	attemptsBefore := testutil.ToFloat64(
		metrics.RaylineARCProviderAttempts.WithLabelValues("success"),
	)
	retriesBefore := testutil.ToFloat64(
		metrics.RaylineARCProviderRetries.WithLabelValues("success"),
	)
	ctx := &RequestContext{
		RaylineARCDispatch: &raylinearc.WorkerManifest{
			OpenRouterMaxRetries: 1,
		},
	}
	headers := &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: envoyAttemptCountHeader, Value: "2"},
	}}

	captureRaylineARCProviderAttempts(headers, ctx, 200)

	if ctx.UpstreamAttemptCount != 2 ||
		ctx.UpstreamRetryCount != 1 ||
		ctx.UpstreamRetryExhausted {
		t.Fatalf("attempt state = %#v", ctx)
	}
	if got := testutil.ToFloat64(
		metrics.RaylineARCProviderLogicalRequests.WithLabelValues("success"),
	) - logicalBefore; got != 1 {
		t.Fatalf("logical request delta = %v", got)
	}
	if got := testutil.ToFloat64(
		metrics.RaylineARCProviderAttempts.WithLabelValues("success"),
	) - attemptsBefore; got != 2 {
		t.Fatalf("attempt delta = %v", got)
	}
	if got := testutil.ToFloat64(
		metrics.RaylineARCProviderRetries.WithLabelValues("success"),
	) - retriesBefore; got != 1 {
		t.Fatalf("retry delta = %v", got)
	}
}

func TestCaptureRaylineARCProviderAttemptsMarksRetryExhaustion(t *testing.T) {
	exhaustedBefore := testutil.ToFloat64(
		metrics.RaylineARCProviderRetryExhaustions.WithLabelValues("429"),
	)
	ctx := &RequestContext{
		RaylineARCDispatch: &raylinearc.WorkerManifest{
			OpenRouterMaxRetries: 1,
		},
	}
	headers := &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: envoyAttemptCountHeader, RawValue: []byte("2")},
	}}

	captureRaylineARCProviderAttempts(headers, ctx, 429)

	if !ctx.UpstreamRetryExhausted || ctx.UpstreamRetryCount != 1 {
		t.Fatalf("retry exhaustion was not captured: %#v", ctx)
	}
	if got := testutil.ToFloat64(
		metrics.RaylineARCProviderRetryExhaustions.WithLabelValues("429"),
	) - exhaustedBefore; got != 1 {
		t.Fatalf("retry exhaustion delta = %v", got)
	}
}

func TestEnvoyAttemptCountDefaultsSafely(t *testing.T) {
	for name, headers := range map[string]*core.HeaderMap{
		"missing":   {},
		"zero":      {Headers: []*core.HeaderValue{{Key: envoyAttemptCountHeader, Value: "0"}}},
		"malformed": {Headers: []*core.HeaderValue{{Key: envoyAttemptCountHeader, Value: "many"}}},
	} {
		t.Run(name, func(t *testing.T) {
			if got := envoyAttemptCount(headers); got != 1 {
				t.Fatalf("attempt count = %d, want 1", got)
			}
		})
	}
}
