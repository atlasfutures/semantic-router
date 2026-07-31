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
	"encoding/json"
)

const encoderSessionResponseSchema = "rayline.arc.session-pooling-response.v1"

type encoderSessionResponseData struct {
	SchemaVersion        string    `json:"schema_version"`
	Embedding            []float64 `json:"embedding"`
	SerializedTokens     int       `json:"serialized_tokens"`
	FullHistoryTokens    int       `json:"full_history_tokens"`
	TruncatedTokens      int       `json:"truncated_tokens"`
	RetainedPrefixTokens int       `json:"retained_prefix_tokens"`
	AppendedTokens       int       `json:"appended_tokens"`
	SessionAction        string    `json:"session_action"`
	SessionRevision      int       `json:"session_revision"`
	SerializerVersion    string    `json:"serializer_version"`
	Model                string    `json:"model"`
	ModelRevision        string    `json:"model_revision"`
	TokenizerRevision    string    `json:"tokenizer_revision"`
	TokenizerSHA256      string    `json:"tokenizer_sha256"`
	EOSTokenID           int       `json:"eos_token_id"`
	EngineBuildID        string    `json:"engine_build_id"`
	IOPluginVersion      string    `json:"io_plugin_version"`
	PoolingCapabilities  []string  `json:"pooling_capabilities"`
}

func (client *EncoderClient) consumeSessionResponse(
	body []byte,
) (*EncoderResult, error) {
	var data encoderSessionResponseData
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&data); err != nil {
		return nil, encoderFailure(EncoderFailureDecode, "json")
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, encoderFailure(EncoderFailureDecode, "json_trailing")
	}
	if data.SchemaVersion != encoderSessionResponseSchema {
		return nil, encoderFailure(EncoderFailureContract, "session_schema")
	}
	if err := validateSessionTokenCounts(&data); err != nil {
		return nil, err
	}
	if err := client.validateEncoderMetadata(data.statelessMetadata()); err != nil {
		return nil, err
	}
	embedding, err := validatedEmbedding(data.Embedding)
	if err != nil {
		return nil, err
	}
	return &EncoderResult{
		Embedding:            embedding,
		SerializedTokens:     data.SerializedTokens,
		FullHistoryTokens:    data.FullHistoryTokens,
		TruncatedTokens:      data.TruncatedTokens,
		CachedPrefixTokens:   0,
		RetainedPrefixTokens: data.RetainedPrefixTokens,
		AppendedTokens:       data.AppendedTokens,
		SessionAction:        data.SessionAction,
		SessionRevision:      data.SessionRevision,
		ModelRevision:        data.ModelRevision,
		EngineBuildID:        data.EngineBuildID,
		IOPluginVersion:      data.IOPluginVersion,
		PoolingCapabilities:  append([]string(nil), data.PoolingCapabilities...),
	}, nil
}

func (data *encoderSessionResponseData) statelessMetadata() *encoderResponseData {
	return &encoderResponseData{
		SerializerVersion:   data.SerializerVersion,
		Model:               data.Model,
		ModelRevision:       data.ModelRevision,
		TokenizerRevision:   data.TokenizerRevision,
		TokenizerSHA256:     data.TokenizerSHA256,
		EOSTokenID:          data.EOSTokenID,
		EngineBuildID:       data.EngineBuildID,
		IOPluginVersion:     data.IOPluginVersion,
		PoolingCapabilities: data.PoolingCapabilities,
	}
}

func validateSessionTokenCounts(data *encoderSessionResponseData) error {
	if invalidSessionTokenCounts(data) {
		return encoderFailure(EncoderFailureContract, "session_token_counts")
	}
	if !validSessionAction(data) {
		return encoderFailure(EncoderFailureContract, "session_action")
	}
	return nil
}

func invalidSessionTokenCounts(data *encoderSessionResponseData) bool {
	return data.SerializedTokens <= 0 ||
		data.FullHistoryTokens <= 0 ||
		data.TruncatedTokens < 0 ||
		data.RetainedPrefixTokens < 0 ||
		data.AppendedTokens < 0 ||
		data.SessionRevision <= 0 ||
		data.SerializedTokens+data.TruncatedTokens != data.FullHistoryTokens ||
		data.RetainedPrefixTokens+data.AppendedTokens != data.SerializedTokens
}

func validSessionAction(data *encoderSessionResponseData) bool {
	switch data.SessionAction {
	case "created", "rebuilt":
		return data.RetainedPrefixTokens == 0 &&
			data.AppendedTokens == data.SerializedTokens
	case "appended":
		return data.RetainedPrefixTokens > 0 && data.AppendedTokens > 0
	case "reused":
		return data.RetainedPrefixTokens == data.SerializedTokens &&
			data.AppendedTokens == 0
	default:
		return false
	}
}
