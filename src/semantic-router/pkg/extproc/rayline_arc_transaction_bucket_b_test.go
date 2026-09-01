//go:build vsr_next_bucket_b

// Parked until Bucket B re-seats the ARC dispatch hooks on upstream's
// prepareProviderDispatch / applyDispatchDecision seam. Build with
// -tags vsr_next_bucket_b once those symbols exist again.

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
	"context"
	"errors"
	"os"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCEpisodeFinalizerCoversEOFErrorCancelAndPanic(
	t *testing.T,
) {
	tests := []struct {
		name   string
		stream ext_proc.ExternalProcessor_ProcessServer
	}{
		{
			name:   "EOF",
			stream: NewMockStream(nil),
		},
		{
			name: "handler error",
			stream: &MockStream{
				Ctx:       context.Background(),
				RecvError: errors.New("synthetic receive failure"),
			},
		},
		{
			name: "cancel",
			stream: &MockStream{
				Ctx:       context.Background(),
				RecvError: status.Error(codes.Canceled, "synthetic cancel"),
			},
		},
		{
			name: "panic",
			stream: &panicOnRecvStream{
				MockStream: MockStream{Ctx: context.Background()},
				panicVal:   "synthetic panic",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router, requestContext, store, episode := newTestARCEpisodeTransaction(t)
			requestContext.RaylineARCTransaction.markSelection(1, 123)
			_ = router.processWithContext(test.stream, requestContext)
			assertARCEpisodeNotAdvanced(t, store, episode)
		})
	}
}

func TestRaylineARCTransactionRenewsRedisLease(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	episode := raylinearc.HashEpisodeID(
		t.Name() + time.Now().String(),
	)
	store, err := raylinearc.NewRedisEpisodeStore(
		raylinearc.RedisEpisodeStoreConfig{
			Address: address,
			KeyPrefix: "test:rayline-arc:" +
				episode + ":",
			LeaseTTL: 90 * time.Millisecond,
			IdleTTL:  time.Minute,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	lease, state, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	requestContext := &RequestContext{
		Headers: make(map[string]string),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(
			store,
			lease,
			state,
			episode,
			90*time.Millisecond,
			nil,
		),
	}
	requestContext.RaylineARCTransaction.markSelection(0, 77)
	time.Sleep(240 * time.Millisecond)
	if finalizeErr := finalizeRaylineARCResponseHeaders(
		requestContext,
		true,
	); finalizeErr != nil {
		t.Fatal(finalizeErr)
	}
	nextLease, nextState, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = store.Abort(context.Background(), nextLease)
	}()
	if nextState.TurnIndex != 1 {
		t.Fatalf("renewed transaction state = %#v", nextState)
	}
}
