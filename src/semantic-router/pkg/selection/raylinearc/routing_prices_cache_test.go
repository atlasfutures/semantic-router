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

package raylinearc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakePriceFeed struct {
	server *httptest.Server
	body   atomic.Value // []byte
	status atomic.Int32
	delay  atomic.Int64 // nanoseconds
	hits   atomic.Int32

	mu   sync.Mutex
	auth []string
}

func newFakePriceFeed(t *testing.T, body []byte) *fakePriceFeed {
	t.Helper()
	feed := &fakePriceFeed{}
	feed.body.Store(body)
	feed.status.Store(http.StatusOK)
	feed.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		feed.hits.Add(1)
		feed.mu.Lock()
		feed.auth = append(feed.auth, request.Header.Get("Authorization"))
		feed.mu.Unlock()
		if delay := time.Duration(feed.delay.Load()); delay > 0 {
			select {
			case <-time.After(delay):
			case <-request.Context().Done():
				return
			}
		}
		writer.WriteHeader(int(feed.status.Load()))
		_, _ = writer.Write(feed.body.Load().([]byte))
	}))
	t.Cleanup(feed.server.Close)
	return feed
}

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (clock *fakeClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

func (clock *fakeClock) advance(by time.Duration) {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	clock.now = clock.now.Add(by)
}

type fetchLog struct {
	mu      sync.Mutex
	classes []string
	status  []int
}

func (log *fetchLog) observe(class string, status int) {
	log.mu.Lock()
	defer log.mu.Unlock()
	log.classes = append(log.classes, class)
	log.status = append(log.status, status)
}

func (log *fetchLog) last() (string, int) {
	log.mu.Lock()
	defer log.mu.Unlock()
	return log.classes[len(log.classes)-1], log.status[len(log.status)-1]
}

func newTestPriceCache(feed *fakePriceFeed, clock *fakeClock, log *fetchLog) *RoutingPriceCache {
	return NewRoutingPriceCache(RoutingPriceCacheConfig{
		URL: feed.server.URL + "/arc/routing-prices", Credential: "test-token",
		TTL: time.Minute, RefreshAhead: 10 * time.Second, MaxAge: 10 * time.Minute,
		FetchTimeout: 2 * time.Second, Now: clock.Now, OnFetch: log.observe,
	})
}

// Fresh rates are served; past the TTL they are still served as stale until
// max age; past max age none are, and the request uses package prices.
func TestRoutingPriceCacheServesFreshThenStaleThenNothing(t *testing.T) {
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	clock := &fakeClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	log := &fetchLog{}
	cache := newTestPriceCache(feed, clock, log)

	if prices, status, _ := cache.Lookup(); prices != nil || status != RoutingPricesAbsent {
		t.Fatalf("before any fetch: %v %s", prices, status)
	}
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if class, _ := log.last(); class != "ok" {
		t.Fatalf("fetch observed as %s", class)
	}
	feed.mu.Lock()
	got := feed.auth[0]
	feed.mu.Unlock()
	if got != "Bearer test-token" {
		t.Fatalf("Authorization = %q", got)
	}
	prices, status, age := cache.Lookup()
	if prices == nil || status != RoutingPricesFresh || age != 0 || prices.StoreVersion != "v000000000042" ||
		len(prices.Rates) != 2 {
		t.Fatalf("fresh lookup: %+v %s %v", prices, status, age)
	}

	clock.advance(time.Minute)
	if prices, status, age := cache.Lookup(); prices == nil || status != RoutingPricesStale || age != time.Minute {
		t.Fatalf("stale lookup: %v %s %v", prices, status, age)
	}
	clock.advance(9*time.Minute - time.Nanosecond)
	if prices, status, _ := cache.Lookup(); prices == nil || status != RoutingPricesStale {
		t.Fatalf("lookup just inside max age: %v %s", prices, status)
	}
	clock.advance(time.Nanosecond)
	if prices, status, age := cache.Lookup(); prices != nil || status != RoutingPricesExpired || age != 10*time.Minute {
		t.Fatalf("lookup at max age: %v %s %v", prices, status, age)
	}

	// A successful refresh makes the rates fresh again.
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, status, _ := cache.Lookup(); status != RoutingPricesFresh {
		t.Fatalf("after refresh: %s", status)
	}
}

// A failed fetch keeps the last good rates and reports a bounded class.
func TestRoutingPriceCacheKeepsRatesThroughFetchFailures(t *testing.T) {
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	clock := &fakeClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	log := &fetchLog{}
	cache := newTestPriceCache(feed, clock, log)
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	clock.advance(2 * time.Minute)

	// Each refusal the feed can give is its own class.
	for status, class := range map[int32]string{
		http.StatusServiceUnavailable:  "unavailable",
		http.StatusUnauthorized:        "unauthorized",
		http.StatusForbidden:           "forbidden",
		http.StatusNotFound:            "not_found",
		http.StatusInternalServerError: "status",
	} {
		feed.status.Store(status)
		requireFetchFailure(t, cache, class)
		if observed, code := log.last(); observed != class || code != int(status) {
			t.Fatalf("observed %s %d, want %s %d", observed, code, class, status)
		}
	}

	feed.status.Store(http.StatusOK)
	feed.body.Store([]byte(`{"schema":"arc-routing-prices.v2"}`))
	requireFetchFailure(t, cache, "decode")

	feed.server.Close()
	requireFetchFailure(t, cache, "transport")

	if prices, status, _ := cache.Lookup(); prices == nil || status != RoutingPricesStale ||
		prices.StoreVersion != "v000000000042" {
		t.Fatalf("after failures: %v %s", prices, status)
	}
}

// A feed slower than the fetch timeout fails as a timeout within it.
func TestRoutingPriceCacheBoundsAFetchByItsTimeout(t *testing.T) {
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	feed.delay.Store(int64(2 * time.Second))
	log := &fetchLog{}
	cache := NewRoutingPriceCache(RoutingPriceCacheConfig{
		URL: feed.server.URL, TTL: time.Minute, MaxAge: time.Hour,
		FetchTimeout: 50 * time.Millisecond, OnFetch: log.observe,
	})
	started := time.Now()
	requireFetchFailure(t, cache, "timeout")
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("fetch took %v with a 50ms timeout", elapsed)
	}
	if prices, status, _ := cache.Lookup(); prices != nil || status != RoutingPricesAbsent {
		t.Fatalf("after a timed-out first fetch: %v %s", prices, status)
	}
}

// Run fetches at once and again ahead of the TTL, and stops with its context.
func TestRoutingPriceCacheRunRefreshesAheadOfTheTTL(t *testing.T) {
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	cache := NewRoutingPriceCache(RoutingPriceCacheConfig{
		URL: feed.server.URL, TTL: 60 * time.Millisecond, RefreshAhead: 40 * time.Millisecond,
		MaxAge: time.Hour, FetchTimeout: time.Second,
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		cache.Run(ctx)
		close(done)
	}()
	deadline := time.Now().Add(2 * time.Second)
	for feed.hits.Load() < 3 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	<-done
	if hits := feed.hits.Load(); hits < 3 {
		t.Fatalf("Run fetched %d times", hits)
	}
	if _, status, _ := cache.Lookup(); status != RoutingPricesFresh {
		t.Fatalf("after Run: %s", status)
	}
}

func requireFetchFailure(t *testing.T, cache *RoutingPriceCache, class string) {
	t.Helper()
	err := cache.Refresh(context.Background())
	var failure *RoutingPriceFetchError
	if !errors.As(err, &failure) || failure.Class != class {
		t.Fatalf("refresh error = %v, want class %s", err, class)
	}
}
