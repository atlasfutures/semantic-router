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
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// RoutingPriceStatus is how old the cached feed was when a request read it.
// It is a closed set: it becomes a metric label.
type RoutingPriceStatus string

const (
	// RoutingPricesFresh is younger than the TTL.
	RoutingPricesFresh RoutingPriceStatus = "fresh"
	// RoutingPricesStale is past the TTL and still within max age: served.
	RoutingPricesStale RoutingPriceStatus = "stale"
	// RoutingPricesExpired is past max age: not served, the request falls
	// back to the package's prices.
	RoutingPricesExpired RoutingPriceStatus = "expired"
	// RoutingPricesAbsent is a feed never fetched: package prices.
	RoutingPricesAbsent RoutingPriceStatus = "absent"
)

const (
	maxRoutingPriceFeedBytes = 4 << 20
	// routingPriceRetryBackoff spaces fetch attempts after a failure.
	routingPriceRetryBackoff = 5 * time.Second
)

// RoutingPriceCacheConfig is the resolved feed configuration. The bearer
// is TokenSource's token when set (google_id_token auth), otherwise
// Credential, a static token the caller read from the environment; neither
// is ever logged.
type RoutingPriceCacheConfig struct {
	URL          string
	Credential   string
	TokenSource  RoutingPriceTokenSource
	TTL          time.Duration
	RefreshAhead time.Duration
	MaxAge       time.Duration
	FetchTimeout time.Duration
	Now          func() time.Time
	// OnFetch observes every fetch: class "ok", or the bounded failure
	// class and the HTTP status when there was one.
	OnFetch func(class string, status int)
}

// RoutingPriceFetchError is a failed fetch with a bounded class: transport,
// timeout, token (no identity token could be minted), unauthorized (401),
// forbidden (403), not_found (404: no store version at that as_of),
// unavailable (503: the feed is not configured or down), status (any other
// non-200), decode or request.
type RoutingPriceFetchError struct {
	Class  string
	Status int
}

func (err *RoutingPriceFetchError) Error() string {
	return fmt.Sprintf("routing price feed fetch failed (class=%s status=%d)", err.Class, err.Status)
}

// RoutingPriceCache holds the latest feed in process. Run refreshes it in
// the background ahead of its TTL; Lookup never fetches, so a request is
// never held by the feed.
type RoutingPriceCache struct {
	config RoutingPriceCacheConfig
	http   *http.Client

	mu         sync.RWMutex
	prices     *PolicyRoutingPrices
	settlement *RoutingPriceSettlement
	fetchedAt  time.Time
}

func NewRoutingPriceCache(config RoutingPriceCacheConfig) *RoutingPriceCache {
	if config.Now == nil {
		config.Now = time.Now
	}
	return &RoutingPriceCache{config: config, http: &http.Client{}}
}

// Lookup returns the rates to attach to a decision request with their
// status and age. Past max age, or before any successful fetch, it returns
// no rates: the service then scores with the package's prices.
func (cache *RoutingPriceCache) Lookup() (*PolicyRoutingPrices, RoutingPriceStatus, time.Duration) {
	cache.mu.RLock()
	defer cache.mu.RUnlock()
	if cache.prices == nil {
		return nil, RoutingPricesAbsent, 0
	}
	age := cache.config.Now().Sub(cache.fetchedAt)
	switch {
	case age >= cache.config.MaxAge:
		return nil, RoutingPricesExpired, age
	case age >= cache.config.TTL:
		return cache.prices, RoutingPricesStale, age
	default:
		return cache.prices, RoutingPricesFresh, age
	}
}

// Refresh fetches the feed once. A failure keeps the cached rates, which
// stay servable until max age.
func (cache *RoutingPriceCache) Refresh(ctx context.Context) error {
	prices, settlement, err := cache.fetch(ctx)
	if cache.config.OnFetch != nil {
		var failure *RoutingPriceFetchError
		switch {
		case err == nil:
			cache.config.OnFetch("ok", 0)
		case errors.As(err, &failure):
			cache.config.OnFetch(failure.Class, failure.Status)
		default:
			cache.config.OnFetch("transport", 0)
		}
	}
	if err != nil {
		return err
	}
	cache.mu.Lock()
	cache.prices, cache.settlement, cache.fetchedAt = prices, settlement, cache.config.Now()
	cache.mu.Unlock()
	return nil
}

// Now is the cache's clock, the one its freshness and a settled row's
// validity window are read on.
func (cache *RoutingPriceCache) Now() time.Time {
	return cache.config.Now()
}

// Settlement returns the feed to settle a call at, or why there is none: no
// feed yet (feed_absent), or one past max age by its fetch or by its own
// as_of (feed_expired). A feed a decide request would still be sent is one
// a call may be settled at.
func (cache *RoutingPriceCache) Settlement() (*RoutingPriceSettlement, string) {
	cache.mu.RLock()
	defer cache.mu.RUnlock()
	if cache.settlement == nil {
		return nil, SettlementFeedAbsent
	}
	now := cache.config.Now()
	if now.Sub(cache.fetchedAt) >= cache.config.MaxAge || now.Sub(cache.settlement.asOf) >= cache.config.MaxAge {
		return nil, SettlementFeedExpired
	}
	return cache.settlement, ""
}

// Run refreshes until ctx ends: at once, then RefreshAhead before each TTL
// runs out, and after a failure again within routingPriceRetryBackoff.
func (cache *RoutingPriceCache) Run(ctx context.Context) {
	interval := cache.config.TTL - cache.config.RefreshAhead
	if interval <= 0 {
		interval = cache.config.TTL
	}
	for {
		wait := interval
		if cache.Refresh(ctx) != nil && routingPriceRetryBackoff < wait {
			wait = routingPriceRetryBackoff
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (cache *RoutingPriceCache) fetch(ctx context.Context) (*PolicyRoutingPrices, *RoutingPriceSettlement, error) {
	ctx, cancel := context.WithTimeout(ctx, cache.config.FetchTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, cache.config.URL, nil)
	if err != nil {
		return nil, nil, &RoutingPriceFetchError{Class: "request"}
	}
	request.Header.Set("Accept", "application/json")
	bearer := cache.config.Credential
	if cache.config.TokenSource != nil {
		if bearer, err = cache.config.TokenSource.Token(ctx); err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				return nil, nil, &RoutingPriceFetchError{Class: "timeout"}
			}
			return nil, nil, &RoutingPriceFetchError{Class: "token"}
		}
	}
	if bearer != "" {
		request.Header.Set("Authorization", "Bearer "+bearer)
	}
	response, err := cache.http.Do(request)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, nil, &RoutingPriceFetchError{Class: "timeout"}
		}
		return nil, nil, &RoutingPriceFetchError{Class: "transport"}
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		if response.StatusCode == http.StatusUnauthorized && cache.config.TokenSource != nil {
			// The next attempt mints afresh rather than resend a refused token.
			cache.config.TokenSource.Invalidate()
		}
		return nil, nil, &RoutingPriceFetchError{Class: routingPriceStatusClass(response.StatusCode), Status: response.StatusCode}
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxRoutingPriceFeedBytes+1))
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, nil, &RoutingPriceFetchError{Class: "timeout"}
		}
		return nil, nil, &RoutingPriceFetchError{Class: "transport"}
	}
	if len(body) > maxRoutingPriceFeedBytes {
		return nil, nil, &RoutingPriceFetchError{Class: "decode"}
	}
	feed, err := DecodeRoutingPriceFeed(body)
	if err != nil {
		return nil, nil, &RoutingPriceFetchError{Class: "decode"}
	}
	return feed.RequestPrices(), newRoutingPriceSettlement(feed, body), nil
}

// routingPriceStatusClass names the feed's refusals apart, since each needs
// a different fix: 401 a token the feed cannot verify (wrong audience), 403
// an identity missing from its invoker allowlist, 404 no store version at
// the requested as_of, 503 a feed with no audience or allowlist set.
func routingPriceStatusClass(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "unauthorized"
	case http.StatusForbidden:
		return "forbidden"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusServiceUnavailable:
		return "unavailable"
	default:
		return "status"
	}
}
