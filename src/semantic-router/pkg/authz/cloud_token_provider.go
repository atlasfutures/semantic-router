package authz

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CloudTokenProviderType is the authz.providers[].type that selects the
// CloudTokenProvider.
const CloudTokenProviderType = "google-access-token"

// googleCloudPlatformScope is the OAuth scope Vertex AI accepts.
const googleCloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// AccessToken is a short-lived bearer credential and the time it stops working.
type AccessToken struct {
	Value  string
	Expiry time.Time
}

// TokenSource mints access tokens. Implementations are called at most once at
// a time by CloudTokenProvider.
type TokenSource interface {
	Token(ctx context.Context) (AccessToken, error)
}

// CloudTokenProvider hands out a Google OAuth access token for the providers
// it serves (Vertex AI by default), so no static key is configured. The token
// comes from the workload's ambient identity (on GKE, Workload Identity through
// the metadata server), is cached, and is refreshed before it expires:
//
//   - with more than refreshAhead left, the cached token is returned;
//   - inside refreshAhead but not yet expired, the cached token is returned and
//     one background refresh is started;
//   - with no usable token (none yet, or within minValidity of expiry), the
//     caller waits for a synchronous refresh; concurrent callers share it.
//
// A failed refresh keeps the cached token while it is still usable. When there
// is none, GetKey returns "" and the credential chain decides (fail closed by
// default).
type CloudTokenProvider struct {
	source       TokenSource
	providers    map[LLMProvider]bool
	refreshAhead time.Duration
	minValidity  time.Duration
	fetchTimeout time.Duration
	now          func() time.Time

	mu         sync.Mutex
	cached     AccessToken
	inflight   chan struct{}
	lastErrLog time.Time
}

// CloudTokenOption configures a CloudTokenProvider.
type CloudTokenOption func(*CloudTokenProvider)

// WithCloudTokenProviders sets which LLM providers receive the token.
func WithCloudTokenProviders(providers ...LLMProvider) CloudTokenOption {
	return func(p *CloudTokenProvider) {
		p.providers = make(map[LLMProvider]bool, len(providers))
		for _, provider := range providers {
			p.providers[provider] = true
		}
	}
}

// WithCloudTokenClock replaces the clock (tests).
func WithCloudTokenClock(now func() time.Time) CloudTokenOption {
	return func(p *CloudTokenProvider) { p.now = now }
}

// WithCloudTokenRefreshAhead sets how long before expiry a background refresh
// starts, and how close to expiry a token stops being handed out.
func WithCloudTokenRefreshAhead(refreshAhead, minValidity time.Duration) CloudTokenOption {
	return func(p *CloudTokenProvider) {
		p.refreshAhead = refreshAhead
		p.minValidity = minValidity
	}
}

// NewCloudTokenProvider creates a provider that serves tokens from source.
func NewCloudTokenProvider(source TokenSource, opts ...CloudTokenOption) *CloudTokenProvider {
	p := &CloudTokenProvider{
		source:       source,
		providers:    map[LLMProvider]bool{ProviderVertexAI: true},
		refreshAhead: 5 * time.Minute,
		minValidity:  30 * time.Second,
		fetchTimeout: 10 * time.Second,
		now:          time.Now,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func (p *CloudTokenProvider) Name() string {
	return CloudTokenProviderType
}

// HeadersToStrip returns nil: the token never comes from a request header.
func (p *CloudTokenProvider) HeadersToStrip() []string {
	return nil
}

// GetKey returns a current access token for a served provider, else "".
func (p *CloudTokenProvider) GetKey(provider LLMProvider, _ string, _ map[string]string) string {
	if p == nil || p.source == nil || !p.providers[provider] {
		return ""
	}
	p.mu.Lock()
	tok := p.cached
	usable := p.usableLocked(tok)
	if usable && p.now().Before(tok.Expiry.Add(-p.refreshAhead)) {
		p.mu.Unlock()
		return tok.Value
	}
	wait := p.inflight
	if wait == nil {
		wait = make(chan struct{})
		p.inflight = wait
		go p.refresh(wait)
	}
	p.mu.Unlock()
	if usable {
		return tok.Value
	}
	<-wait
	p.mu.Lock()
	defer p.mu.Unlock()
	if tok = p.cached; p.usableLocked(tok) {
		return tok.Value
	}
	// The refresh failed and nothing usable is cached.
	return ""
}

func (p *CloudTokenProvider) usableLocked(tok AccessToken) bool {
	return tok.Value != "" && p.now().Before(tok.Expiry.Add(-p.minValidity))
}

func (p *CloudTokenProvider) refresh(done chan struct{}) {
	ctx, cancel := context.WithTimeout(context.Background(), p.fetchTimeout)
	defer cancel()
	tok, err := p.source.Token(ctx)
	if err == nil && (tok.Value == "" || tok.Expiry.IsZero()) {
		err = errors.New("token source returned an empty token or no expiry")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err == nil {
		p.cached = tok
	} else if now := p.now(); now.Sub(p.lastErrLog) > time.Minute {
		p.lastErrLog = now
		logging.ComponentErrorEvent("authz", "cloud_token_refresh_failed", map[string]interface{}{
			"provider": CloudTokenProviderType,
			"error":    err.Error(),
		})
	}
	p.inflight = nil
	close(done)
}

// MetadataTokenSource reads the default service account's access token from
// the GCE/GKE metadata server. On GKE with Workload Identity that is the
// token of the identity bound to the pod's Kubernetes service account.
// GCE_METADATA_HOST overrides the host, as in Google's client libraries.
type MetadataTokenSource struct {
	Host   string
	Scopes []string
	Client *http.Client
}

// NewMetadataTokenSource returns a source for the cloud-platform scope.
func NewMetadataTokenSource() *MetadataTokenSource {
	host := strings.TrimSpace(os.Getenv("GCE_METADATA_HOST"))
	if host == "" {
		host = "169.254.169.254"
	}
	return &MetadataTokenSource{
		Host:   host,
		Scopes: []string{googleCloudPlatformScope},
		// The metadata server is link-local: never reach it through an
		// HTTP(S)_PROXY, which would not exempt it by itself.
		Client: &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{Proxy: nil}},
	}
}

type metadataTokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int64  `json:"expires_in"`
	TokenType   string `json:"token_type"`
}

// Token fetches one token.
func (s *MetadataTokenSource) Token(ctx context.Context) (AccessToken, error) {
	host := s.Host
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	endpoint := strings.TrimRight(host, "/") + "/computeMetadata/v1/instance/service-accounts/default/token"
	if len(s.Scopes) > 0 {
		endpoint += "?scopes=" + url.QueryEscape(strings.Join(s.Scopes, ","))
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return AccessToken{}, fmt.Errorf("metadata token request: %w", err)
	}
	req.Header.Set("Metadata-Flavor", "Google")
	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return AccessToken{}, fmt.Errorf("metadata token request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		return AccessToken{}, fmt.Errorf("metadata token response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		// The body is not echoed: it can name the identity.
		return AccessToken{}, fmt.Errorf("metadata token response: status %d", resp.StatusCode)
	}
	var parsed metadataTokenResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return AccessToken{}, fmt.Errorf("metadata token response: invalid JSON")
	}
	if parsed.AccessToken == "" || parsed.ExpiresIn <= 0 {
		return AccessToken{}, errors.New("metadata token response: missing access_token or expires_in")
	}
	if parsed.TokenType != "" && !strings.EqualFold(parsed.TokenType, "Bearer") {
		return AccessToken{}, fmt.Errorf("metadata token response: token_type %q is not Bearer", parsed.TokenType)
	}
	return AccessToken{
		Value:  parsed.AccessToken,
		Expiry: start.Add(time.Duration(parsed.ExpiresIn) * time.Second),
	}, nil
}
