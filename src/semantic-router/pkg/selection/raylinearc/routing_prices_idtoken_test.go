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
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testFeedAudience = "https://routerproxy-abc123-uc.a.run.app"

// fakeMetadataServer stands in for the instance metadata server: it mints a
// distinct JWT-shaped token per call, valid for an hour from the clock, and
// refuses a request without Metadata-Flavor, another audience, or another
// format.
type fakeMetadataServer struct {
	server *httptest.Server
	clock  *fakeClock
	mints  atomic.Int32
	status atomic.Int32
	body   atomic.Value // string; overrides the minted token when set

	mu      sync.Mutex
	queries []string
}

func newFakeMetadataServer(t *testing.T, clock *fakeClock) *fakeMetadataServer {
	t.Helper()
	metadata := &fakeMetadataServer{clock: clock}
	metadata.status.Store(http.StatusOK)
	metadata.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		metadata.mu.Lock()
		metadata.queries = append(metadata.queries, request.URL.RawQuery)
		metadata.mu.Unlock()
		query := request.URL.Query()
		if request.Header.Get("Metadata-Flavor") != "Google" || query.Get("audience") != testFeedAudience ||
			query.Get("format") != "full" || request.URL.Path != "/computeMetadata/v1/instance/service-accounts/default/identity" {
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		if status := int(metadata.status.Load()); status != http.StatusOK {
			writer.WriteHeader(status)
			return
		}
		if body, ok := metadata.body.Load().(string); ok {
			_, _ = writer.Write([]byte(body))
			return
		}
		n := metadata.mints.Add(1)
		_, _ = writer.Write([]byte(testIDToken(n, clock.Now().Add(time.Hour))))
	}))
	t.Cleanup(metadata.server.Close)
	return metadata
}

func (metadata *fakeMetadataServer) endpoint() string {
	return metadata.server.URL + "/computeMetadata/v1/instance/service-accounts/default/identity"
}

func testIDToken(n int32, expiry time.Time) string {
	encode := base64.RawURLEncoding.EncodeToString
	payload := fmt.Sprintf(`{"aud":%q,"email":"vsr@example.iam.gserviceaccount.com","email_verified":true,"exp":%d,"n":%d}`,
		testFeedAudience, expiry.Unix(), n)
	return encode([]byte(`{"alg":"RS256","typ":"JWT"}`)) + "." + encode([]byte(payload)) + ".c2ln"
}

func newIDTokenPriceCache(feed *fakePriceFeed, metadata *fakeMetadataServer, clock *fakeClock, log *fetchLog) *RoutingPriceCache {
	return NewRoutingPriceCache(RoutingPriceCacheConfig{
		URL:         feed.server.URL + "/arc/routing-prices",
		TokenSource: NewMetadataIDTokenSource(testFeedAudience, metadata.endpoint(), clock.Now),
		TTL:         time.Minute, MaxAge: 10 * time.Minute, FetchTimeout: 2 * time.Second,
		Now: clock.Now, OnFetch: log.observe,
	})
}

func (feed *fakePriceFeed) lastAuth() string {
	feed.mu.Lock()
	defer feed.mu.Unlock()
	return feed.auth[len(feed.auth)-1]
}

// The feed gets the metadata server's identity token for its audience as
// the bearer; one token serves every fetch until shortly before it expires.
func TestRoutingPriceCacheSendsACachedMetadataIDToken(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	metadata := newFakeMetadataServer(t, clock)
	cache := newIDTokenPriceCache(feed, metadata, clock, &fetchLog{})

	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	first := "Bearer " + testIDToken(1, clock.Now().Add(time.Hour))
	if feed.lastAuth() != first {
		t.Fatalf("Authorization = %q", feed.lastAuth())
	}
	clock.advance(50 * time.Minute)
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if metadata.mints.Load() != 1 || feed.lastAuth() != first {
		t.Fatalf("token re-minted early: %d mints", metadata.mints.Load())
	}
	// Within the refresh margin of expiry a new token is minted.
	clock.advance(6 * time.Minute)
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if metadata.mints.Load() != 2 || feed.lastAuth() != "Bearer "+testIDToken(2, clock.Now().Add(time.Hour)) {
		t.Fatalf("token not re-minted near expiry: %d mints, %q", metadata.mints.Load(), feed.lastAuth())
	}
}

// A 401 drops the cached token, so the next fetch mints afresh instead of
// resending one the feed refused.
func TestRoutingPriceCacheRemintsAfterUnauthorized(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
	metadata := newFakeMetadataServer(t, clock)
	cache := newIDTokenPriceCache(feed, metadata, clock, &fetchLog{})

	feed.status.Store(http.StatusUnauthorized)
	requireFetchFailure(t, cache, "unauthorized")
	feed.status.Store(http.StatusForbidden)
	requireFetchFailure(t, cache, "forbidden")
	if metadata.mints.Load() != 2 {
		t.Fatalf("mints after a 401 then a 403 = %d, want 2", metadata.mints.Load())
	}
	feed.status.Store(http.StatusOK)
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if metadata.mints.Load() != 2 {
		t.Fatalf("a 403 dropped the token: %d mints", metadata.mints.Load())
	}
}

// No token, no fetch: a metadata failure or an unreadable token is the
// bounded class "token" and the feed is never called without a bearer.
func TestRoutingPriceCacheFailsWithoutAnIDToken(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	for name, setup := range map[string]func(*fakeMetadataServer){
		"metadata refuses": func(metadata *fakeMetadataServer) { metadata.status.Store(http.StatusNotFound) },
		"not a JWT":        func(metadata *fakeMetadataServer) { metadata.body.Store("opaque-token") },
		"no exp claim": func(metadata *fakeMetadataServer) {
			metadata.body.Store("e30." + base64.RawURLEncoding.EncodeToString([]byte(`{"aud":"x"}`)) + ".c2ln")
		},
		"already expired": func(metadata *fakeMetadataServer) {
			metadata.body.Store(testIDToken(1, clock.Now().Add(-time.Minute)))
		},
	} {
		t.Run(name, func(t *testing.T) {
			feed := newFakePriceFeed(t, readRoutingPriceFixture(t, "feed.v1.json"))
			metadata := newFakeMetadataServer(t, clock)
			setup(metadata)
			log := &fetchLog{}
			cache := newIDTokenPriceCache(feed, metadata, clock, log)
			requireFetchFailure(t, cache, "token")
			if class, _ := log.last(); class != "token" || feed.hits.Load() != 0 {
				t.Fatalf("observed %s, feed hits %d", class, feed.hits.Load())
			}
		})
	}
}
