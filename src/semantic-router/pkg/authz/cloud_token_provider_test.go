package authz

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// fakeTokenSource mints "tok-N" tokens valid for ttl from the fake clock.
type fakeTokenSource struct {
	clock *fakeClock
	ttl   time.Duration
	calls atomic.Int32
	fail  atomic.Bool
	gate  chan struct{} // when set, Token blocks until it is closed
}

func (s *fakeTokenSource) Token(ctx context.Context) (AccessToken, error) {
	n := s.calls.Add(1)
	if s.gate != nil {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return AccessToken{}, ctx.Err()
		}
	}
	if s.fail.Load() {
		return AccessToken{}, errors.New("metadata unavailable")
	}
	return AccessToken{Value: fmt.Sprintf("tok-%d", n), Expiry: s.clock.Now().Add(s.ttl)}, nil
}

func newTestCloudProvider(src *fakeTokenSource, clock *fakeClock) *CloudTokenProvider {
	return NewCloudTokenProvider(src, WithCloudTokenClock(clock.Now))
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition not met in time")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestCloudTokenProviderServesOnlyConfiguredProviders(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour}
	p := newTestCloudProvider(src, clock)

	if got := p.GetKey(ProviderOpenAI, "m", nil); got != "" {
		t.Fatalf("GetKey(openai) = %q, want empty", got)
	}
	if src.calls.Load() != 0 {
		t.Fatalf("token source called for an unserved provider")
	}
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
		t.Fatalf("GetKey(vertex-ai) = %q, want tok-1", got)
	}
	if p.Name() != CloudTokenProviderType || p.HeadersToStrip() != nil {
		t.Fatalf("unexpected name or headers to strip")
	}
}

func TestCloudTokenProviderCachesUntilRefreshWindow(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour}
	p := newTestCloudProvider(src, clock)

	for i := 0; i < 5; i++ {
		if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
			t.Fatalf("GetKey = %q, want tok-1", got)
		}
	}
	clock.Advance(50 * time.Minute) // 10 min left, outside the 5 min window
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
		t.Fatalf("GetKey = %q, want cached tok-1", got)
	}
	if n := src.calls.Load(); n != 1 {
		t.Fatalf("token source calls = %d, want 1", n)
	}
}

func TestCloudTokenProviderRefreshesAheadInBackground(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour}
	p := newTestCloudProvider(src, clock)
	_ = p.GetKey(ProviderVertexAI, "m", nil)

	clock.Advance(57 * time.Minute) // 3 min left: inside refreshAhead, still usable
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
		t.Fatalf("GetKey = %q, want the still-valid tok-1 while refreshing", got)
	}
	waitFor(t, func() bool { return p.GetKey(ProviderVertexAI, "m", nil) == "tok-2" })
	if n := src.calls.Load(); n != 2 {
		t.Fatalf("token source calls = %d, want 2", n)
	}
}

func TestCloudTokenProviderBlocksWhenExpiredAndSharesOneRefresh(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour, gate: make(chan struct{})}
	p := newTestCloudProvider(src, clock)

	const callers = 16
	results := make(chan string, callers)
	for i := 0; i < callers; i++ {
		go func() { results <- p.GetKey(ProviderVertexAI, "m", nil) }()
	}
	waitFor(t, func() bool { return src.calls.Load() == 1 })
	close(src.gate)
	for i := 0; i < callers; i++ {
		if got := <-results; got != "tok-1" {
			t.Fatalf("caller got %q, want tok-1", got)
		}
	}
	if n := src.calls.Load(); n != 1 {
		t.Fatalf("token source calls = %d, want 1 shared refresh", n)
	}
}

func TestCloudTokenProviderKeepsUsableTokenWhenRefreshFails(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour}
	p := newTestCloudProvider(src, clock)
	_ = p.GetKey(ProviderVertexAI, "m", nil)

	src.fail.Store(true)
	clock.Advance(57 * time.Minute)
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
		t.Fatalf("GetKey = %q, want tok-1 kept through a failed refresh", got)
	}
	waitFor(t, func() bool { return src.calls.Load() == 2 })
	waitFor(t, func() bool {
		p.mu.Lock()
		defer p.mu.Unlock()
		return p.inflight == nil
	})
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "tok-1" {
		t.Fatalf("GetKey = %q, want tok-1", got)
	}

	// Past expiry with the source still failing: no credential.
	clock.Advance(3 * time.Minute)
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "" {
		t.Fatalf("GetKey = %q, want empty once the token is unusable", got)
	}
	// The source recovers: the next call mints a new token.
	src.fail.Store(false)
	if got := p.GetKey(ProviderVertexAI, "m", nil); !strings.HasPrefix(got, "tok-") || got == "tok-1" {
		t.Fatalf("GetKey = %q, want a fresh token", got)
	}
}

func TestCloudTokenProviderFailsClosedInChain(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_000_000, 0)}
	src := &fakeTokenSource{clock: clock, ttl: time.Hour}
	src.fail.Store(true)
	resolver := NewCredentialResolver(newTestCloudProvider(src, clock))
	if _, err := resolver.KeyForProvider(ProviderVertexAI, "m", nil); err == nil {
		t.Fatal("KeyForProvider succeeded with no token; want fail-closed error")
	}
	src.fail.Store(false)
	key, err := resolver.KeyForProvider(ProviderVertexAI, "m", nil)
	if err != nil || key == "" {
		t.Fatalf("KeyForProvider = %q, %v; want a token", key, err)
	}
}

func TestMetadataTokenSource(t *testing.T) {
	var gotPath, gotFlavor, gotScopes string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotFlavor, gotScopes = r.URL.Path, r.Header.Get("Metadata-Flavor"), r.URL.Query().Get("scopes")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"ya29.test","expires_in":3599,"token_type":"Bearer"}`))
	}))
	defer srv.Close()

	src := &MetadataTokenSource{Host: srv.URL, Scopes: []string{googleCloudPlatformScope}, Client: srv.Client()}
	before := time.Now()
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok.Value != "ya29.test" {
		t.Fatalf("token = %q", tok.Value)
	}
	if tok.Expiry.Before(before.Add(3598*time.Second)) || tok.Expiry.After(time.Now().Add(3600*time.Second)) {
		t.Fatalf("expiry %v not about 3599s ahead", tok.Expiry)
	}
	if gotPath != "/computeMetadata/v1/instance/service-accounts/default/token" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotFlavor != "Google" {
		t.Fatalf("Metadata-Flavor = %q", gotFlavor)
	}
	if gotScopes != googleCloudPlatformScope {
		t.Fatalf("scopes = %q", gotScopes)
	}
}

func TestMetadataTokenSourceRejectsBadResponses(t *testing.T) {
	cases := map[string]struct {
		status int
		body   string
	}{
		"status":     {http.StatusNotFound, `not found`},
		"not json":   {http.StatusOK, `<html>`},
		"no token":   {http.StatusOK, `{"expires_in":3599}`},
		"no expiry":  {http.StatusOK, `{"access_token":"x"}`},
		"not bearer": {http.StatusOK, `{"access_token":"x","expires_in":10,"token_type":"MAC"}`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			src := &MetadataTokenSource{Host: srv.URL, Client: srv.Client()}
			if _, err := src.Token(context.Background()); err == nil {
				t.Fatal("Token succeeded; want error")
			}
		})
	}
}

func TestNewMetadataTokenSourceHonoursHostOverride(t *testing.T) {
	t.Setenv("GCE_METADATA_HOST", "metadata.test:8080")
	if got := NewMetadataTokenSource().Host; got != "metadata.test:8080" {
		t.Fatalf("Host = %q", got)
	}
	t.Setenv("GCE_METADATA_HOST", "")
	if got := NewMetadataTokenSource().Host; got != "169.254.169.254" {
		t.Fatalf("Host = %q", got)
	}
}

func TestCloudTokenProviderWithMetadataServer(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := hits.Add(1)
		_, _ = fmt.Fprintf(w, `{"access_token":"ya29.%d","expires_in":3599,"token_type":"Bearer"}`, n)
	}))
	defer srv.Close()
	p := NewCloudTokenProvider(&MetadataTokenSource{Host: srv.URL, Client: srv.Client()})
	for i := 0; i < 3; i++ {
		if got := p.GetKey(ProviderVertexAI, "m", nil); got != "ya29.1" {
			t.Fatalf("GetKey = %q, want ya29.1", got)
		}
	}
	if hits.Load() != 1 {
		t.Fatalf("metadata hits = %d, want 1", hits.Load())
	}
}
