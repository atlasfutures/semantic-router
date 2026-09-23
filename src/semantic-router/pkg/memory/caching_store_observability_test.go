package memory

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// storeWithUnreachableCache wraps an in-memory store in a CachingStore whose
// shared cache is unreachable, standing in for a Redis hot cache that is down
// or partitioned. The unix socket path does not exist, so every cache command
// fails at once instead of waiting for a dial timeout.
func storeWithUnreachableCache(t *testing.T) Store {
	t.Helper()
	t.Setenv(deterministicEmbeddingsEnv, "true")
	client := redis.NewClient(&redis.Options{
		Network:    "unix",
		Addr:       filepath.Join(t.TempDir(), "absent.sock"),
		MaxRetries: -1,
	})
	t.Cleanup(func() { _ = client.Close() })
	cache := &RedisCache{client: client, prefix: "memory_cache_test:", ttl: time.Minute}
	return NewCachingStore(NewInMemoryStore(), cache, "milvus")
}

// TestCachingStore_SharedCacheErrorIsNotACacheMiss shows that a failing shared
// cache is recorded as an ordinary cache miss on the read path.
func TestCachingStore_SharedCacheErrorIsNotACacheMiss(t *testing.T) {
	wrapped := storeWithUnreachableCache(t)
	opts := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}

	missesBefore := testutil.ToFloat64(MemoryCacheMisses.WithLabelValues("milvus"))
	_, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	missesAfter := testutil.ToFloat64(MemoryCacheMisses.WithLabelValues("milvus"))

	require.Equal(t, missesBefore, missesAfter,
		"an unreachable shared cache was counted as a cache miss: hit-rate metrics cannot tell a cold key from a cache outage")
	require.Equal(t, 1.0,
		testutil.ToFloat64(MemoryRetrievalCount.WithLabelValues("milvus", "cache_error", "u1")),
		"no metric records that the shared cache failed on the read path")
}

// TestCachingStore_FailedInvalidationIsSilent shows that a write whose cache
// invalidation fails still reports success, with nothing for an operator to
// alert on.
func TestCachingStore_FailedInvalidationIsSilent(t *testing.T) {
	wrapped := storeWithUnreachableCache(t)
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "user likes coffee", UserID: "u1"}

	require.NoError(t, wrapped.Store(context.Background(), mem))

	require.Equal(t, 1.0,
		testutil.ToFloat64(MemoryStoreOperations.WithLabelValues("milvus", "store", "cache_invalidate_error")),
		"the write succeeded but its cache invalidation failed unnoticed: every reader keeps serving the pre-write result until the cache TTL expires")
}
