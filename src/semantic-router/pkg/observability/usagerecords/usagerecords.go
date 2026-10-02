// Package usagerecords sends the Router's per-call usage records to a durable
// sink. The llm_usage log line remains the record of last resort: a record the
// sink cannot take is counted and dropped here, never retried in the request
// path.
package usagerecords

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// recordField is the stream entry field that holds the record's JSON.
const recordField = "record"

const writeTimeout = 2 * time.Second

// failureLogInterval bounds the write-failure log. A down Redis fails every
// record; the counter counts each one, and the log names the first and then
// at most one per interval with how many it left out.
const failureLogInterval = time.Minute

var recordsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_usage_records_total",
	Help: "Usage records offered to the durable sink, by outcome (written, queue_full, write_failed, closed, abandoned).",
}, []string{"outcome"})

// Sink takes one encoded usage record. Publish must not block.
type Sink interface {
	Publish(record []byte)
}

type sinkHolder struct{ sink Sink }

var current atomic.Pointer[sinkHolder]

// Active reports whether a sink is installed, so a caller can skip encoding a
// record nobody will take.
func Active() bool {
	return current.Load() != nil
}

// Publish hands the record to the installed sink, if there is one.
func Publish(record []byte) {
	if holder := current.Load(); holder != nil {
		holder.sink.Publish(record)
	}
}

// Install makes sink the destination of Publish and returns a function that
// restores the previous one.
func Install(sink Sink) (restore func()) {
	var holder *sinkHolder
	if sink != nil {
		holder = &sinkHolder{sink: sink}
	}
	previous := current.Swap(holder)
	return func() { current.Store(previous) }
}

// redisXAdder is the one Redis call the stream sink makes.
type redisXAdder interface {
	XAdd(ctx context.Context, args *redis.XAddArgs) *redis.StringCmd
}

// RedisStreamSink appends each record to a capped Redis stream from one
// background writer.
type RedisStreamSink struct {
	client redisXAdder
	closer func() error
	stream string
	maxLen int64
	queue  chan []byte
	done   chan struct{}
	// abandon tells the writer that shutdown ran out of time: what is still
	// queued is counted abandoned rather than written.
	abandon chan struct{}
	// lastFailureLog and suppressedFailures belong to the writer goroutine.
	lastFailureLog     time.Time
	suppressedFailures int
	now                func() time.Time
	// mu orders Publish against Close: a record offered after Close is
	// counted dropped instead of sent on a closed queue.
	mu     sync.RWMutex
	closed bool
}

// NewRedisStreamSink connects lazily; an unreachable Redis shows up as
// write_failed records, never as a refused start.
func NewRedisStreamSink(cfg config.UsageRecordsRedisConfig) (*RedisStreamSink, error) {
	if cfg.Address == "" {
		return nil, errors.New("usage records: redis.address is required")
	}
	if cfg.MaxLen < 0 || cfg.QueueSize < 0 || cfg.DB < 0 {
		return nil, errors.New("usage records: max_len, queue_size and db must not be negative")
	}
	password := ""
	if cfg.PasswordEnv != "" {
		value, ok := os.LookupEnv(cfg.PasswordEnv)
		if !ok {
			return nil, fmt.Errorf("usage records: password env %s is not set", cfg.PasswordEnv)
		}
		password = value
	}
	options := &redis.Options{Addr: cfg.Address, DB: cfg.DB, Password: password}
	if cfg.UseTLS {
		options.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}
	client := redis.NewClient(options)
	return newRedisStreamSink(client, client.Close, cfg), nil
}

func newRedisStreamSink(client redisXAdder, closer func() error, cfg config.UsageRecordsRedisConfig) *RedisStreamSink {
	stream := cfg.Stream
	if stream == "" {
		stream = config.DefaultUsageRecordsStream
	}
	maxLen := cfg.MaxLen
	if maxLen == 0 {
		maxLen = config.DefaultUsageRecordsMaxLen
	}
	queueSize := cfg.QueueSize
	if queueSize == 0 {
		queueSize = config.DefaultUsageRecordsQueueSize
	}
	sink := &RedisStreamSink{
		client: client, closer: closer, stream: stream, maxLen: maxLen,
		queue: make(chan []byte, queueSize), done: make(chan struct{}), abandon: make(chan struct{}),
		now: time.Now,
	}
	go sink.run()
	return sink
}

// Publish queues the record, or counts it dropped when the queue is full.
func (sink *RedisStreamSink) Publish(record []byte) {
	sink.mu.RLock()
	defer sink.mu.RUnlock()
	if sink.closed {
		recordsTotal.WithLabelValues("closed").Inc()
		return
	}
	select {
	case sink.queue <- record:
	default:
		recordsTotal.WithLabelValues("queue_full").Inc()
	}
}

func (sink *RedisStreamSink) run() {
	defer close(sink.done)
	for record := range sink.queue {
		select {
		case <-sink.abandon:
			recordsTotal.WithLabelValues("abandoned").Inc()
		default:
			sink.write(record)
		}
	}
}

func (sink *RedisStreamSink) write(record []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	err := sink.client.XAdd(ctx, &redis.XAddArgs{
		Stream: sink.stream,
		MaxLen: sink.maxLen,
		Approx: true,
		Values: []interface{}{recordField, string(record)},
	}).Err()
	if err != nil {
		recordsTotal.WithLabelValues("write_failed").Inc()
		sink.logWriteFailure(err)
		return
	}
	recordsTotal.WithLabelValues("written").Inc()
}

// logWriteFailure logs the first failure and then at most one per interval,
// naming how many failures it did not log in between.
func (sink *RedisStreamSink) logWriteFailure(err error) {
	now := sink.now()
	if !sink.lastFailureLog.IsZero() && now.Sub(sink.lastFailureLog) < failureLogInterval {
		sink.suppressedFailures++
		return
	}
	logging.ComponentWarnEvent("usagerecords", "usage_record_write_failed", map[string]interface{}{
		"stream":                sink.stream,
		"error":                 err.Error(),
		"suppressed_since_last": sink.suppressedFailures,
	})
	sink.lastFailureLog = now
	sink.suppressedFailures = 0
}

// Close stops taking records, writes what is queued for up to timeout, counts
// whatever is still queued after that as abandoned, and closes the connection.
func (sink *RedisStreamSink) Close(timeout time.Duration) {
	sink.mu.Lock()
	if sink.closed {
		sink.mu.Unlock()
		return
	}
	sink.closed = true
	close(sink.queue)
	sink.mu.Unlock()
	select {
	case <-sink.done:
	case <-time.After(timeout):
		// Out of time: the writer counts what is left as abandoned. It is
		// at most one write from noticing, so wait that long for the count
		// to be complete before the process may exit.
		close(sink.abandon)
		select {
		case <-sink.done:
		case <-time.After(writeTimeout):
		}
	}
	if sink.closer != nil {
		_ = sink.closer()
	}
}
