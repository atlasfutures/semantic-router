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

// The routing price feed's google_id_token auth: a Google-signed identity
// token for the runtime service account, minted by the instance metadata
// server for the feed's audience. The feed checks the audience and that the
// token's verified email is on its invoker allowlist; nothing here is a
// stored secret.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// DefaultMetadataIdentityURL is the metadata server's identity endpoint for
// the instance's default service account (Cloud Run, GKE, GCE).
const DefaultMetadataIdentityURL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"

const (
	// idTokenRefreshMargin re-mints a token this long before it expires,
	// so a fetch never sends one that lapses in flight. Metadata tokens
	// live an hour.
	idTokenRefreshMargin = 5 * time.Minute
	maxIDTokenBytes      = 16 << 10
)

// RoutingPriceTokenSource yields the feed's bearer token for one fetch.
// Invalidate drops a cached token the feed refused.
type RoutingPriceTokenSource interface {
	Token(ctx context.Context) (string, error)
	Invalidate()
}

// MetadataIDTokenSource mints identity tokens from the metadata server and
// caches each until idTokenRefreshMargin before its exp claim.
type MetadataIDTokenSource struct {
	endpoint string
	audience string
	now      func() time.Time
	http     *http.Client

	mu     sync.Mutex
	token  string
	expiry time.Time
}

// NewMetadataIDTokenSource mints for audience from endpoint, the metadata
// identity URL (DefaultMetadataIdentityURL outside tests).
func NewMetadataIDTokenSource(audience string, endpoint string, now func() time.Time) *MetadataIDTokenSource {
	if now == nil {
		now = time.Now
	}
	return &MetadataIDTokenSource{endpoint: endpoint, audience: audience, now: now, http: &http.Client{}}
}

func (source *MetadataIDTokenSource) Token(ctx context.Context) (string, error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	if source.token != "" && source.now().Before(source.expiry.Add(-idTokenRefreshMargin)) {
		return source.token, nil
	}
	token, expiry, err := source.mint(ctx)
	if err != nil {
		return "", err
	}
	source.token, source.expiry = token, expiry
	return token, nil
}

func (source *MetadataIDTokenSource) Invalidate() {
	source.mu.Lock()
	source.token, source.expiry = "", time.Time{}
	source.mu.Unlock()
}

func (source *MetadataIDTokenSource) mint(ctx context.Context) (string, time.Time, error) {
	query := url.Values{"audience": {source.audience}, "format": {"full"}}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source.endpoint+"?"+query.Encode(), nil)
	if err != nil {
		return "", time.Time{}, err
	}
	request.Header.Set("Metadata-Flavor", "Google")
	response, err := source.http.Do(request)
	if err != nil {
		return "", time.Time{}, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("metadata identity status %d", response.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxIDTokenBytes+1))
	if err != nil {
		return "", time.Time{}, err
	}
	if len(body) > maxIDTokenBytes {
		return "", time.Time{}, errors.New("metadata identity token too long")
	}
	token := strings.TrimSpace(string(body))
	expiry, err := idTokenExpiry(token)
	if err != nil {
		return "", time.Time{}, err
	}
	if !source.now().Before(expiry) {
		return "", time.Time{}, errors.New("metadata identity token already expired")
	}
	return token, expiry, nil
}

// idTokenExpiry reads a JWT's exp claim. The token is not verified: this
// process minted it from its own metadata server, and the feed verifies it.
func idTokenExpiry(token string) (time.Time, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return time.Time{}, errors.New("metadata identity token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return time.Time{}, errors.New("metadata identity token payload is not base64url")
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil || claims.Exp <= 0 {
		return time.Time{}, errors.New("metadata identity token has no exp")
	}
	return time.Unix(claims.Exp, 0), nil
}
