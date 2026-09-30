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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const maxPolicyServiceResponseBytes = 4 << 20

// PolicyServiceConfig is the resolved client configuration; credentials are
// values here, read from the environment by readiness.
type PolicyServiceConfig struct {
	BaseURL      string
	ModalKey     string
	ModalSecret  string
	TotalTimeout time.Duration
	// ConnectTimeout bounds the dial and the TLS handshake; zero selects
	// DefaultPolicyServiceConnectTimeout.
	ConnectTimeout time.Duration
}

// DefaultPolicyServiceConnectTimeout matches the encoder's shipped connect
// timeout.
const DefaultPolicyServiceConnectTimeout = 5 * time.Second

// PolicyServiceClient calls the policy service's decide and packages endpoints.
type PolicyServiceClient struct {
	config PolicyServiceConfig
	http   *http.Client
}

// PolicyServiceError is a failure with a bounded class for metrics and logs:
// the service's error code when it answered one, otherwise transport, status
// or decode.
type PolicyServiceError struct {
	Class  string
	Status int
}

func (err *PolicyServiceError) Error() string {
	return fmt.Sprintf("policy service failed (class=%s status=%d)", err.Class, err.Status)
}

// policyServiceErrorCodes is the contract's closed set of error codes
// (pathfinder arc_policy_contract.ErrorCode). A class becomes a metric label
// and a log field, so a code outside the set is never passed through.
var policyServiceErrorCodes = map[string]bool{
	"invalid_request":                  true,
	"package_not_loaded":               true,
	"package_hash_mismatch":            true,
	"session_busy":                     true,
	"unsupported_request":              true,
	"selection_refused":                true,
	"context_exceeds_encoder_capacity": true,
	"session_capacity":                 true,
	"backend_unavailable":              true,
}

// PolicyServiceErrorClass bounds a service error code to the contract's set;
// anything else is "service_error".
func PolicyServiceErrorClass(code string) string {
	if policyServiceErrorCodes[code] {
		return code
	}
	return "service_error"
}

func NewPolicyServiceClient(config PolicyServiceConfig) *PolicyServiceClient {
	connect := config.ConnectTimeout
	if connect <= 0 {
		connect = DefaultPolicyServiceConnectTimeout
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: connect, KeepAlive: 30 * time.Second}).DialContext
	transport.TLSHandshakeTimeout = connect
	return &PolicyServiceClient{config: config, http: &http.Client{Transport: transport}}
}

// Packages reads the service's loaded packages, naming the pin it needs. A
// store-mode service (one app hot-loading packages from a volume) loads and
// lists a package only for the pin that names it, and a service that serves
// a fixed set ignores the query and lists what it has.
func (client *PolicyServiceClient) Packages(ctx context.Context, alias, sha256 string) (*PolicyPackagesResponse, error) {
	path := "/v1/rayline/arc/policy/packages"
	if alias != "" {
		path += "?" + url.Values{"alias": {alias}, "package_sha256": {sha256}}.Encode()
	}
	body, err := client.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	response, err := DecodePolicyPackagesResponse(body)
	if err != nil {
		return nil, &PolicyServiceError{Class: "decode"}
	}
	return response, nil
}

// RequirePackage fails unless the service has loaded exactly this package.
func (client *PolicyServiceClient) RequirePackage(ctx context.Context, alias, sha256 string) error {
	packages, err := client.Packages(ctx, alias, sha256)
	if err != nil {
		return err
	}
	for _, loaded := range packages.Packages {
		if loaded.Alias != alias {
			continue
		}
		if loaded.PackageSHA256 != sha256 {
			return &PolicyServiceError{Class: "package_hash_mismatch"}
		}
		if loaded.State != "loaded" {
			return &PolicyServiceError{Class: "package_not_loaded"}
		}
		return nil
	}
	return &PolicyServiceError{Class: "package_not_loaded"}
}

// Decide asks the service for one decision.
func (client *PolicyServiceClient) Decide(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, &PolicyServiceError{Class: "request"}
	}
	body, err := client.do(ctx, http.MethodPost, "/v1/rayline/arc/policy/decide", payload)
	if err != nil {
		return nil, err
	}
	response, err := DecodePolicyDecisionResponse(body)
	if err != nil {
		return nil, &PolicyServiceError{Class: "decode"}
	}
	return response, nil
}

func (client *PolicyServiceClient) do(
	ctx context.Context,
	method string,
	path string,
	payload []byte,
) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, client.config.TotalTimeout)
	defer cancel()
	var reader io.Reader
	if payload != nil {
		reader = bytes.NewReader(payload)
	}
	endpoint := strings.TrimRight(client.config.BaseURL, "/") + path
	request, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, &PolicyServiceError{Class: "request"}
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if client.config.ModalKey != "" {
		request.Header.Set("Modal-Key", client.config.ModalKey)
		request.Header.Set("Modal-Secret", client.config.ModalSecret)
	}
	response, err := client.http.Do(request)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, &PolicyServiceError{Class: "timeout"}
		}
		return nil, &PolicyServiceError{Class: "transport"}
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxPolicyServiceResponseBytes+1))
	if err != nil || len(body) > maxPolicyServiceResponseBytes {
		return nil, &PolicyServiceError{Class: "transport", Status: response.StatusCode}
	}
	if response.StatusCode != http.StatusOK {
		var failure PolicyErrorResponse
		if json.Unmarshal(body, &failure) == nil && failure.Error != "" {
			return nil, &PolicyServiceError{Class: PolicyServiceErrorClass(failure.Error), Status: response.StatusCode}
		}
		return nil, &PolicyServiceError{Class: "status", Status: response.StatusCode}
	}
	return body, nil
}
