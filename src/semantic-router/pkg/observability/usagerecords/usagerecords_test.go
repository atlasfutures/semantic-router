package usagerecords

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type recordingXAdder struct {
	mu    sync.Mutex
	calls []*redis.XAddArgs
	block chan struct{}
}

func (fake *recordingXAdder) XAdd(_ context.Context, args *redis.XAddArgs) *redis.StringCmd {
	if fake.block != nil {
		<-fake.block
	}
	fake.mu.Lock()
	fake.calls = append(fake.calls, args)
	fake.mu.Unlock()
	return redis.NewStringResult("0-1", nil)
}

func (fake *recordingXAdder) recorded() []*redis.XAddArgs {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]*redis.XAddArgs(nil), fake.calls...)
}

func TestRedisStreamSinkAppendsEachRecordToACappedStream(t *testing.T) {
	fake := &recordingXAdder{}
	sink := newRedisStreamSink(fake, nil, config.UsageRecordsRedisConfig{Stream: "usage", MaxLen: 50})
	sink.Publish([]byte(`{"request_id":"rt_1","cost":null}`))
	sink.Publish([]byte(`{"request_id":"rt_2","cost":0.5}`))
	sink.Close(time.Second)

	calls := fake.recorded()
	if len(calls) != 2 {
		t.Fatalf("XADD calls = %d, want one per record", len(calls))
	}
	first := calls[0]
	if first.Stream != "usage" || first.MaxLen != 50 || !first.Approx {
		t.Fatalf("XADD args = %+v, want stream usage capped ~50", first)
	}
	values, _ := first.Values.([]interface{})
	if len(values) != 2 || values[0] != recordField || values[1] != `{"request_id":"rt_1","cost":null}` {
		t.Fatalf("XADD values = %#v", first.Values)
	}
}

func TestRedisStreamSinkDefaultsStreamAndCap(t *testing.T) {
	fake := &recordingXAdder{}
	sink := newRedisStreamSink(fake, nil, config.UsageRecordsRedisConfig{})
	sink.Publish([]byte(`{}`))
	sink.Close(time.Second)
	calls := fake.recorded()
	if len(calls) != 1 || calls[0].Stream != config.DefaultUsageRecordsStream ||
		calls[0].MaxLen != config.DefaultUsageRecordsMaxLen {
		t.Fatalf("XADD args = %+v", calls)
	}
}

// A full queue drops and counts; it never makes a response wait on Redis.
func TestRedisStreamSinkNeverBlocksTheCaller(t *testing.T) {
	fake := &recordingXAdder{block: make(chan struct{})}
	sink := newRedisStreamSink(fake, nil, config.UsageRecordsRedisConfig{QueueSize: 1})
	before := testutil.ToFloat64(recordsTotal.WithLabelValues("queue_full"))

	finished := make(chan struct{})
	go func() {
		for range 5 {
			sink.Publish([]byte(`{}`))
		}
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("Publish blocked behind a stalled Redis")
	}
	if dropped := testutil.ToFloat64(recordsTotal.WithLabelValues("queue_full")) - before; dropped < 3 {
		t.Fatalf("queue_full dropped = %v, want the records the queue could not hold", dropped)
	}
	close(fake.block)
	sink.Close(time.Second)
}

func TestPublishAfterCloseIsCountedNotSent(t *testing.T) {
	sink := newRedisStreamSink(&recordingXAdder{}, nil, config.UsageRecordsRedisConfig{})
	sink.Close(time.Second)
	before := testutil.ToFloat64(recordsTotal.WithLabelValues("closed"))
	sink.Publish([]byte(`{}`))
	if testutil.ToFloat64(recordsTotal.WithLabelValues("closed"))-before != 1 {
		t.Fatal("a record offered after Close was not counted")
	}
}

func TestInstallRoutesPublishAndRestores(t *testing.T) {
	fake := &recordingXAdder{}
	sink := newRedisStreamSink(fake, nil, config.UsageRecordsRedisConfig{})
	restore := Install(sink)
	if !Active() {
		t.Fatal("an installed sink is not active")
	}
	Publish([]byte(`{}`))
	restore()
	if Active() {
		t.Fatal("restore left the sink installed")
	}
	Publish([]byte(`{}`))
	sink.Close(time.Second)
	if calls := fake.recorded(); len(calls) != 1 {
		t.Fatalf("records reaching the sink = %d, want only the one published while installed", len(calls))
	}
}

func TestNewRedisStreamSinkRefusesAMissingPassword(t *testing.T) {
	_, err := NewRedisStreamSink(config.UsageRecordsRedisConfig{
		Address: "127.0.0.1:6379", PasswordEnv: "VSR_USAGE_RECORDS_TEST_UNSET_PASSWORD",
	})
	if err == nil {
		t.Fatal("a named password env that is unset was accepted")
	}
}

// Against a real Redis: the record read back from the stream is the record
// published, byte for byte.
func TestRedisStreamSinkRoundTripsThroughRedis(t *testing.T) {
	address := os.Getenv("VSR_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("VSR_TEST_REDIS_ADDR is not set")
	}
	stream := "vsr:llm_usage:test:" + time.Now().Format("150405.000000000")
	sink, err := NewRedisStreamSink(config.UsageRecordsRedisConfig{Address: address, Stream: stream, MaxLen: 10})
	if err != nil {
		t.Fatal(err)
	}
	record := `{"request_id":"rt_redis","cost":null,"pricing_snapshot":"sha256:00"}`
	sink.Publish([]byte(record))
	sink.Close(5 * time.Second)

	client := redis.NewClient(&redis.Options{Addr: address})
	defer client.Close()
	defer client.Del(context.Background(), stream)
	entries, err := client.XRange(context.Background(), stream, "-", "+").Result()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Values[recordField] != record {
		t.Fatalf("stream entries = %+v, want the one record", entries)
	}
}

// Shutdown that runs out of time counts what it leaves behind, so a durable
// record lost at exit is visible in the metric.
func TestCloseCountsRecordsItAbandons(t *testing.T) {
	fake := &recordingXAdder{block: make(chan struct{})}
	sink := newRedisStreamSink(fake, nil, config.UsageRecordsRedisConfig{QueueSize: 8})
	before := testutil.ToFloat64(recordsTotal.WithLabelValues("abandoned"))
	for range 3 {
		sink.Publish([]byte(`{}`))
	}

	closed := make(chan struct{})
	go func() {
		sink.Close(50 * time.Millisecond)
		close(closed)
	}()
	time.Sleep(200 * time.Millisecond) // past the drain timeout; the first write is still stalled
	close(fake.block)
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return")
	}
	if abandoned := testutil.ToFloat64(recordsTotal.WithLabelValues("abandoned")) - before; abandoned != 2 {
		t.Fatalf("abandoned = %v, want the 2 records still queued behind the stalled write", abandoned)
	}
}
