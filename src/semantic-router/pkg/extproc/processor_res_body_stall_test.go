package extproc

import (
	"context"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An upstream that stops mid-body is not a message the Router is ever sent.
//
// The accumulation deadline was only ever read when a new chunk arrived, so
// an upstream that sent one non-terminal chunk and then went quiet was never
// refused. Envoy's stream_idle_timeout (620 s on the cell) eventually tears
// the stream down, so the bytes are not held forever, but ext_proc's
// message_timeout does not apply under FULL_DUPLEX_STREAMED and the Router's
// own 590 s response deadline covers only a streamed turn. The client
// therefore waits out the platform for a bare cut, which is the outcome
// decision 27 exists to remove.

// stallingStream delivers its queued messages and then blocks, the way a
// gRPC stream does when the upstream has stopped sending.
type stallingStream struct {
	grpc.ServerStream
	mu        sync.Mutex
	requests  []*ext_proc.ProcessingRequest
	responses []*ext_proc.ProcessingResponse
	released  chan struct{}
	ctx       context.Context
}

func newStallingStream(requests ...*ext_proc.ProcessingRequest) *stallingStream {
	return &stallingStream{
		requests: requests, released: make(chan struct{}), ctx: context.Background(),
	}
}

func (s *stallingStream) Send(response *ext_proc.ProcessingResponse) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.responses = append(s.responses, response)
	return nil
}

func (s *stallingStream) Recv() (*ext_proc.ProcessingRequest, error) {
	s.mu.Lock()
	if len(s.requests) > 0 {
		request := s.requests[0]
		s.requests = s.requests[1:]
		s.mu.Unlock()
		return request, nil
	}
	s.mu.Unlock()
	<-s.released
	return nil, context.Canceled
}

func (s *stallingStream) Context() context.Context { return s.ctx }

func (s *stallingStream) sent() []*ext_proc.ProcessingResponse {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*ext_proc.ProcessingResponse(nil), s.responses...)
}

func TestAStalledUpstreamIsRefusedAtTheAccumulationDeadline(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterOptions: config.RouterOptions{ResponseBodyTimeoutSec: 1},
	}}
	stream := newStallingStream(&ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(`{"partial":`), EndOfStream: false},
		},
	})
	ctx := &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: context.Background(),
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = router.processWithContext(stream, ctx)
	}()
	t.Cleanup(func() { close(stream.released); <-done })

	require.Eventually(t, func() bool { return len(stream.sent()) >= 2 }, 5*time.Second, 20*time.Millisecond,
		"the stalled response was never refused")

	sent := stream.sent()
	refusal := sent[1].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, refusal, "the refusal was not a shape the mode accepts")
	assert.True(t, refusal.GetEndOfStream(), "the stalled response was left open")
	assert.Contains(t, string(refusal.GetBody()), "response_body_timeout")
}

// With no deadline configured the loop waits, exactly as it did before.
func TestAStalledUpstreamIsNotRefusedWithoutADeadline(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	stream := newStallingStream(&ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(`{"partial":`), EndOfStream: false},
		},
	})
	ctx := &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		FullDuplexResponseBody: true, TraceContext: context.Background(),
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = router.processWithContext(stream, ctx)
	}()
	t.Cleanup(func() { close(stream.released); <-done })

	require.Never(t, func() bool { return len(stream.sent()) > 1 }, 500*time.Millisecond, 50*time.Millisecond,
		"an unconfigured router refused a stalled response")
}
