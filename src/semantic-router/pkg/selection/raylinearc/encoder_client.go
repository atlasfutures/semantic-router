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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

const (
	encoderRequestSchema        = "rayline.arc.pooling-request.v1"
	encoderSessionRequestSchema = "rayline.arc.session-pooling-request.v1"
	encoderSessionCapability    = "resumable_causal_mean"
	encoderServingRungA         = "A"
	encoderServingRungB         = "B"
	encoderDimension            = 1024
	maxEncoderResponse          = 256 * 1024
	maxEncoderCloseResponse     = 4096

	// EncoderTokenizerSHA256 is a public file-integrity fingerprint, not a credential.
	EncoderTokenizerSHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42" // #nosec G101
	EncoderEOSTokenID      = 248046
)

type EncoderFailureClass string

const (
	EncoderFailureRequest   EncoderFailureClass = "request"
	EncoderFailureTransport EncoderFailureClass = "transport"
	EncoderFailureTimeout   EncoderFailureClass = "timeout"
	EncoderFailureStatus    EncoderFailureClass = "status"
	EncoderFailureDecode    EncoderFailureClass = "decode"
	EncoderFailureContract  EncoderFailureClass = "contract"
)

// EncoderFailure is deliberately bounded: it reports a stable class and stage
// but never includes request text, episode identity, response bodies, or keys.
type EncoderFailure struct {
	Class      EncoderFailureClass
	Stage      string
	StatusCode int
}

func (failure *EncoderFailure) Error() string {
	if failure == nil {
		return "ARC encoder failed"
	}
	return fmt.Sprintf(
		"ARC encoder failed (class=%s, stage=%s)",
		failure.Class,
		failure.Stage,
	)
}

type EncoderClientConfig struct {
	BaseURL               string
	Model                 string
	ModelRevision         string
	TokenizerRevision     string
	TokenizerSHA256       string
	EOSTokenID            int
	ExpectedBuildID       string
	ExpectedPluginVersion string
	SerializerVersion     string
	ServingRung           string
	RequiredCapabilities  []string
	RetainedSession       bool
	ModalKey              string
	ModalSecret           string
	ConnectTimeout        time.Duration
	TotalTimeout          time.Duration
	MaxRetries            int
}

type EncoderResult struct {
	Embedding            []float32
	SerializedTokens     int
	FullHistoryTokens    int
	TruncatedTokens      int
	CachedPrefixTokens   int
	RetainedPrefixTokens int
	AppendedTokens       int
	SessionAction        string
	SessionRevision      int
	ModelRevision        string
	EngineBuildID        string
	IOPluginVersion      string
	PoolingCapabilities  []string
	// ReplicaID and VisitedReplicaIDs are internal transactional affinity.
	// Callers must not export them as telemetry labels or log fields.
	ReplicaID         string
	ReplicaIndex      int
	ReplicaAttempts   int
	ReplicaFailover   bool
	VisitedReplicaIDs []string
}

type EncoderClient struct {
	config     EncoderClientConfig
	endpoint   string
	httpClient *http.Client
}

type encoderWireRequest struct {
	Task string             `json:"task"`
	Data encoderRequestData `json:"data"`
}

type encoderRequestData struct {
	SchemaVersion string `json:"schema_version"`
	Serializer    string `json:"serializer_version"`
	ServingRung   string `json:"serving_rung"`
	EpisodeIDHash string `json:"episode_id_hash"`
	Turns         []Turn `json:"turns"`
}

// encoderWireResponse is the exact envelope the pinned vLLM engine emits for
// IO-processor pooling responses: plugin-owned payload under "data" plus the
// engine-owned "request_id" and "created_at" correlation fields.
type encoderWireResponse struct {
	RequestID string              `json:"request_id"`
	CreatedAt int64               `json:"created_at"`
	Data      encoderResponseData `json:"data"`
}

type encoderResponseData struct {
	Embedding           []float64 `json:"embedding"`
	SerializedTokens    int       `json:"serialized_tokens"`
	FullHistoryTokens   int       `json:"full_history_tokens"`
	TruncatedTokens     int       `json:"truncated_tokens"`
	CachedPrefixTokens  int       `json:"cached_prefix_tokens"`
	SerializerVersion   string    `json:"serializer_version"`
	Model               string    `json:"model"`
	ModelRevision       string    `json:"model_revision"`
	TokenizerRevision   string    `json:"tokenizer_revision"`
	TokenizerSHA256     string    `json:"tokenizer_sha256"`
	EOSTokenID          int       `json:"eos_token_id"`
	EngineBuildID       string    `json:"engine_build_id"`
	IOPluginVersion     string    `json:"io_plugin_version"`
	PoolingCapabilities []string  `json:"pooling_capabilities"`
}

type encoderSessionCloseResponse struct {
	Closed *bool `json:"closed"`
}

func NewEncoderClient(config EncoderClientConfig) (*EncoderClient, error) {
	dialer := &net.Dialer{Timeout: config.ConnectTimeout}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = dialer.DialContext
	return newEncoderClient(config, &http.Client{Transport: transport})
}

func newEncoderClient(
	config EncoderClientConfig,
	httpClient *http.Client,
) (*EncoderClient, error) {
	endpoint, err := encoderEndpoint(config.BaseURL, config.RetainedSession)
	if err != nil {
		return nil, err
	}
	if err := validateEncoderClientConfig(config, httpClient); err != nil {
		return nil, err
	}
	config.RequiredCapabilities = append(
		[]string(nil),
		config.RequiredCapabilities...,
	)
	return &EncoderClient{
		config:     config,
		endpoint:   endpoint,
		httpClient: httpClient,
	}, nil
}

func validateEncoderClientConfig(
	config EncoderClientConfig,
	httpClient *http.Client,
) error {
	if err := validateEncoderHTTPPolicy(config, httpClient); err != nil {
		return err
	}
	if encoderReadinessContractIncomplete(config) {
		return errors.New("ARC encoder readiness contract is incomplete")
	}
	return validateEncoderServingContract(config)
}

func validateEncoderHTTPPolicy(
	config EncoderClientConfig,
	httpClient *http.Client,
) error {
	if config.ConnectTimeout <= 0 || config.TotalTimeout <= 0 ||
		config.ConnectTimeout > config.TotalTimeout {
		return errors.New("invalid ARC encoder timeout configuration")
	}
	if config.MaxRetries < 0 || config.MaxRetries > 3 {
		return errors.New("invalid ARC encoder retry configuration")
	}
	if httpClient == nil {
		return errors.New("ARC encoder HTTP client is required")
	}
	if (config.ModalKey == "") != (config.ModalSecret == "") {
		return errors.New("ARC encoder Modal credentials must be paired")
	}
	return nil
}

func validateEncoderServingContract(config EncoderClientConfig) error {
	if config.ServingRung != encoderServingRungA &&
		config.ServingRung != encoderServingRungB {
		return errors.New("ARC encoder serving rung must be A or B")
	}
	if config.RetainedSession && config.ServingRung != encoderServingRungB {
		return errors.New("ARC retained sessions require serving rung B")
	}
	if config.RetainedSession &&
		!slices.Contains(config.RequiredCapabilities, encoderSessionCapability) {
		return errors.New("ARC retained sessions require resumable capability")
	}
	return nil
}

func encoderReadinessContractIncomplete(config EncoderClientConfig) bool {
	return config.SerializerVersion == "" ||
		config.Model == "" ||
		config.ModelRevision == "" ||
		config.TokenizerRevision == "" ||
		config.TokenizerSHA256 == "" ||
		config.ExpectedBuildID == "" ||
		config.ExpectedPluginVersion == "" ||
		config.ServingRung == "" ||
		len(config.RequiredCapabilities) == 0
}

func encoderEndpoint(baseURL string, retainedSession bool) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", errors.New("invalid ARC encoder base URL")
	}
	endpointPath := "/pooling"
	if retainedSession {
		endpointPath = "/v1/rayline/arc/session/pooling"
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + endpointPath
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func HashEpisodeID(raw string) string {
	digest := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(digest[:])
}

func (client *EncoderClient) Encode(
	parent context.Context,
	episodeIDHash string,
	turns []Turn,
) (*EncoderResult, error) {
	payload, err := client.buildPayload(episodeIDHash, turns)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(parent, client.config.TotalTimeout)
	defer cancel()
	response, err := client.doWithRetries(ctx, payload)
	if err != nil {
		return nil, err
	}
	return client.consumeResponse(response)
}

func (client *EncoderClient) buildPayload(
	episodeIDHash string,
	turns []Turn,
) ([]byte, error) {
	if client == nil {
		return nil, encoderFailure(EncoderFailureRequest, "client")
	}
	if !validEpisodeIDHash(episodeIDHash) {
		return nil, encoderFailure(EncoderFailureRequest, "episode_hash")
	}
	if len(turns) == 0 {
		return nil, encoderFailure(EncoderFailureRequest, "turns")
	}
	requestData := encoderRequestData{
		SchemaVersion: encoderRequestSchema,
		Serializer:    client.config.SerializerVersion,
		ServingRung:   client.config.ServingRung,
		EpisodeIDHash: episodeIDHash,
		Turns:         turns,
	}
	var wireRequest any = encoderWireRequest{
		Task: "plugin",
		Data: requestData,
	}
	if client.config.RetainedSession {
		requestData.SchemaVersion = encoderSessionRequestSchema
		wireRequest = requestData
	}
	payload, err := json.Marshal(wireRequest)
	if err != nil {
		return nil, encoderFailure(EncoderFailureRequest, "encode")
	}
	return payload, nil
}

func validEpisodeIDHash(value string) bool {
	if len(value) != sha256.Size*2 ||
		value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func (client *EncoderClient) doWithRetries(
	ctx context.Context,
	payload []byte,
) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		response, err := client.doAttempt(ctx, payload)
		if err == nil {
			return response, nil
		}
		failure, retry := requestAttemptFailure(
			ctx,
			response,
			err,
			attempt < client.config.MaxRetries,
		)
		if !retry {
			return nil, failure
		}
	}
}

func (client *EncoderClient) doAttempt(
	ctx context.Context,
	payload []byte,
) (*http.Response, error) {
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		client.endpoint,
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, encoderFailure(EncoderFailureRequest, "construct")
	}
	request.Header.Set("content-type", "application/json")
	if client.config.ModalKey != "" {
		request.Header.Set("Modal-Key", client.config.ModalKey)
		request.Header.Set("Modal-Secret", client.config.ModalSecret)
	}
	return client.httpClient.Do(request)
}

func (client *EncoderClient) closeSession(
	parent context.Context,
	episodeIDHash string,
) error {
	endpoint, err := encoderSessionCloseEndpoint(
		client.config.BaseURL,
		episodeIDHash,
	)
	if err != nil {
		return encoderFailure(EncoderFailureRequest, "close_endpoint")
	}
	ctx, cancel := context.WithTimeout(parent, client.config.TotalTimeout)
	defer cancel()
	for attempt := 0; ; attempt++ {
		response, requestErr := client.doCloseAttempt(ctx, endpoint)
		if requestErr == nil {
			return client.consumeCloseResponse(response)
		}
		failure, retry := requestAttemptFailure(
			ctx,
			response,
			requestErr,
			attempt < client.config.MaxRetries,
		)
		if !retry {
			return failure
		}
	}
}

// CloseSession explicitly releases a retained encoder session. It is public
// so a replica pool can fan the close out to every replica that has owned the
// episode. The client still enforces the same bounded response contract and
// never includes episode identity or response bodies in returned errors.
func (client *EncoderClient) CloseSession(
	parent context.Context,
	episodeIDHash string,
) error {
	if client == nil {
		return encoderFailure(EncoderFailureRequest, "client")
	}
	if !client.config.RetainedSession {
		return encoderFailure(EncoderFailureRequest, "close_not_retained")
	}
	return client.closeSession(parent, episodeIDHash)
}

// Close releases idle HTTP connections owned by this client.
func (client *EncoderClient) Close() {
	if client == nil || client.httpClient == nil {
		return
	}
	client.httpClient.CloseIdleConnections()
}

func encoderSessionCloseEndpoint(baseURL, episodeIDHash string) (string, error) {
	if !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC encoder episode hash")
	}
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", errors.New("invalid ARC encoder base URL")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") +
		"/v1/rayline/arc/session/" + episodeIDHash
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func (client *EncoderClient) doCloseAttempt(
	ctx context.Context,
	endpoint string,
) (*http.Response, error) {
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodDelete,
		endpoint,
		nil,
	)
	if err != nil {
		return nil, encoderFailure(EncoderFailureRequest, "close_construct")
	}
	request.Header.Set("accept", "application/json")
	if client.config.ModalKey != "" {
		request.Header.Set("Modal-Key", client.config.ModalKey)
		request.Header.Set("Modal-Secret", client.config.ModalSecret)
	}
	return client.httpClient.Do(request)
}

func (client *EncoderClient) consumeCloseResponse(response *http.Response) error {
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK ||
		response.StatusCode >= http.StatusMultipleChoices {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return encoderStatusFailure("close_http_status", response.StatusCode)
	}
	body, err := io.ReadAll(
		io.LimitReader(response.Body, maxEncoderCloseResponse+1),
	)
	if err != nil {
		return encoderFailure(EncoderFailureDecode, "close_read")
	}
	if len(body) > maxEncoderCloseResponse {
		return encoderFailure(EncoderFailureDecode, "close_response_limit")
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var wire encoderSessionCloseResponse
	if err := decoder.Decode(&wire); err != nil {
		return encoderFailure(EncoderFailureDecode, "close_json")
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return encoderFailure(EncoderFailureDecode, "close_json_trailing")
	}
	if wire.Closed == nil || !*wire.Closed {
		return encoderFailure(EncoderFailureContract, "close_status")
	}
	return nil
}

func requestAttemptFailure(
	ctx context.Context,
	response *http.Response,
	err error,
	canRetry bool,
) (*EncoderFailure, bool) {
	var bounded *EncoderFailure
	if errors.As(err, &bounded) {
		return bounded, false
	}
	if response != nil {
		closeEncoderResponse(response)
		return encoderFailure(EncoderFailureTransport, "response_error"), false
	}
	if ctx.Err() != nil {
		return encoderFailure(EncoderFailureTimeout, "request"), false
	}
	if canRetry {
		return nil, true
	}
	return encoderFailure(EncoderFailureTransport, "pre_response"), false
}

func closeEncoderResponse(response *http.Response) {
	if response.Body != nil {
		_ = response.Body.Close()
	}
}

// Probe exercises the exact plugin path and full metadata contract. The input
// is fixed public canary content and the caller-provided correlation is hashed.
func (client *EncoderClient) Probe(
	ctx context.Context,
	correlation string,
) error {
	if client == nil {
		return encoderFailure(EncoderFailureRequest, "client")
	}
	episodeIDHash := HashEpisodeID(correlation)
	_, encodeErr := client.Encode(
		ctx,
		episodeIDHash,
		[]Turn{{Role: "user", Text: "Rayline ARC readiness probe"}},
	)
	if !client.config.RetainedSession {
		return encodeErr
	}
	closeErr := client.closeSession(context.WithoutCancel(ctx), episodeIDHash)
	if encodeErr != nil && closeErr != nil {
		return errors.Join(encodeErr, closeErr)
	}
	if encodeErr != nil {
		return encodeErr
	}
	return closeErr
}

func (client *EncoderClient) consumeResponse(
	response *http.Response,
) (*EncoderResult, error) {
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK ||
		response.StatusCode >= http.StatusMultipleChoices {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return nil, encoderStatusFailure("http_status", response.StatusCode)
	}
	limited := io.LimitReader(response.Body, maxEncoderResponse+1)
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, encoderFailure(EncoderFailureDecode, "read")
	}
	if len(body) > maxEncoderResponse {
		return nil, encoderFailure(EncoderFailureDecode, "response_limit")
	}
	if client.config.RetainedSession {
		return client.consumeSessionResponse(body)
	}
	var wire encoderWireResponse
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return nil, encoderFailure(EncoderFailureDecode, "json")
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, encoderFailure(EncoderFailureDecode, "json_trailing")
	}
	// Online vLLM always stamps both envelope fields; requiring them keeps a
	// nonconforming proxy or fake from satisfying readiness.
	if wire.RequestID == "" || wire.CreatedAt <= 0 {
		return nil, encoderFailure(EncoderFailureContract, "envelope")
	}
	return client.validateResponse(&wire.Data)
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	return errors.New("unexpected trailing JSON")
}

func (client *EncoderClient) validateResponse(
	data *encoderResponseData,
) (*EncoderResult, error) {
	embedding, err := validatedEmbedding(data.Embedding)
	if err != nil {
		return nil, err
	}
	if err := validateEncoderTokenCounts(data); err != nil {
		return nil, err
	}
	if err := client.validateEncoderMetadata(data); err != nil {
		return nil, err
	}
	return &EncoderResult{
		Embedding:           embedding,
		SerializedTokens:    data.SerializedTokens,
		FullHistoryTokens:   data.FullHistoryTokens,
		TruncatedTokens:     data.TruncatedTokens,
		CachedPrefixTokens:  data.CachedPrefixTokens,
		ModelRevision:       data.ModelRevision,
		EngineBuildID:       data.EngineBuildID,
		IOPluginVersion:     data.IOPluginVersion,
		PoolingCapabilities: append([]string(nil), data.PoolingCapabilities...),
	}, nil
}

func validatedEmbedding(values []float64) ([]float32, error) {
	if len(values) != encoderDimension {
		return nil, encoderFailure(EncoderFailureContract, "embedding_dimension")
	}
	embedding := make([]float32, len(values))
	for index, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, encoderFailure(EncoderFailureContract, "embedding_finite")
		}
		embedding[index] = float32(value)
		if math.IsInf(float64(embedding[index]), 0) {
			return nil, encoderFailure(EncoderFailureContract, "embedding_float32")
		}
	}
	return embedding, nil
}

func validateEncoderTokenCounts(data *encoderResponseData) error {
	if data.SerializedTokens <= 0 ||
		data.FullHistoryTokens <= 0 ||
		data.TruncatedTokens < 0 ||
		data.CachedPrefixTokens < 0 ||
		data.SerializedTokens+data.TruncatedTokens != data.FullHistoryTokens {
		return encoderFailure(EncoderFailureContract, "token_counts")
	}
	return nil
}

func (client *EncoderClient) validateEncoderMetadata(
	data *encoderResponseData,
) error {
	checks := []struct {
		valid bool
		stage string
	}{
		{data.SerializerVersion == client.config.SerializerVersion, "serializer"},
		{data.Model == client.config.Model, "model"},
		{data.ModelRevision == client.config.ModelRevision, "model_revision"},
		{data.TokenizerRevision == client.config.TokenizerRevision, "tokenizer_revision"},
		{data.TokenizerSHA256 == client.config.TokenizerSHA256, "tokenizer_sha256"},
		{data.EOSTokenID == client.config.EOSTokenID, "eos_token_id"},
		{data.EngineBuildID == client.config.ExpectedBuildID, "engine_build"},
		{data.IOPluginVersion == client.config.ExpectedPluginVersion, "io_plugin"},
		{
			equalCapabilities(
				data.PoolingCapabilities,
				client.config.RequiredCapabilities,
			),
			"capabilities",
		},
	}
	for _, check := range checks {
		if !check.valid {
			return encoderFailure(EncoderFailureContract, check.stage)
		}
	}
	return nil
}

func equalCapabilities(actual []string, expected []string) bool {
	if len(actual) != len(expected) {
		return false
	}
	actualCopy := append([]string(nil), actual...)
	expectedCopy := append([]string(nil), expected...)
	slices.Sort(actualCopy)
	slices.Sort(expectedCopy)
	return slices.Equal(actualCopy, expectedCopy)
}

func encoderFailure(
	class EncoderFailureClass,
	stage string,
) *EncoderFailure {
	return &EncoderFailure{Class: class, Stage: stage}
}

func encoderStatusFailure(stage string, statusCode int) *EncoderFailure {
	return &EncoderFailure{
		Class:      EncoderFailureStatus,
		Stage:      stage,
		StatusCode: statusCode,
	}
}
