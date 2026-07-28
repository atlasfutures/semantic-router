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
	"errors"
	"fmt"
	"math"
)

var requiredTensorNames = []string{
	"model_encoder.meta_mean",
	"model_encoder.meta_std",
	"model_encoder.all_metas",
	"model_encoder.meta_mlp.0.weight",
	"model_encoder.meta_mlp.0.bias",
	"model_encoder.meta_mlp.2.weight",
	"model_encoder.meta_mlp.2.bias",
	"model_encoder.residual_embed.weight",
	"model_encoder.output_proj.0.weight",
	"model_encoder.output_proj.0.bias",
	"model_encoder.output_proj.1.weight",
	"model_encoder.output_proj.1.bias",
	"q_network.backbone.0.weight",
	"q_network.backbone.0.bias",
	"q_network.backbone.3.weight",
	"q_network.backbone.3.bias",
	"q_network.head.weight",
	"q_network.head.bias",
}

type Head struct {
	architecture  ArchitectureManifest
	tensors       map[string]tensor
	armEmbeddings [][]float32
}

func loadHead(data []byte, manifest *Manifest) (*Head, error) {
	tensors, err := decodeSafeTensors(data, requiredTensorNames)
	if err != nil {
		return nil, err
	}
	head := &Head{
		architecture: manifest.Architecture,
		tensors:      tensors,
	}
	if err := head.validateShapes(len(manifest.Workers)); err != nil {
		return nil, err
	}
	head.armEmbeddings = make([][]float32, len(manifest.Workers))
	for index := range manifest.Workers {
		embedding, err := head.armEmbedding(index)
		if err != nil {
			return nil, fmt.Errorf("compute arm embedding %d: %w", index, err)
		}
		head.armEmbeddings[index] = embedding
	}
	return head, nil
}

func (head *Head) Scores(
	history []float32,
	previousArm *int,
	turnIndex uint64,
) ([]float32, error) {
	if len(history) != head.architecture.HistoryDimension {
		return nil, fmt.Errorf(
			"history embedding has %d dimensions; expected %d",
			len(history),
			head.architecture.HistoryDimension,
		)
	}
	if previousArm != nil &&
		(*previousArm < 0 || *previousArm >= len(head.armEmbeddings)) {
		return nil, errors.New("previous arm index is out of range")
	}
	normalized := append([]float32(nil), history...)
	if err := normalize(normalized); err != nil {
		return nil, err
	}
	previous := make([]float32, head.architecture.ArmEmbeddingDimension)
	if previousArm != nil {
		copy(previous, head.armEmbeddings[*previousArm])
	}
	scores := make([]float32, len(head.armEmbeddings))
	for index, candidate := range head.armEmbeddings {
		input := make([]float32, 0, head.architecture.JointInputDimension)
		input = append(input, normalized...)
		input = append(input, candidate...)
		input = append(input, previous...)
		if previousArm != nil && *previousArm == index {
			input = append(input, 1)
		} else {
			input = append(input, 0)
		}
		input = append(
			input,
			float32(math.Log1p(float64(float32(turnIndex)))),
		)
		hidden, err := head.linear("q_network.backbone.0", input)
		if err != nil {
			return nil, err
		}
		relu(hidden)
		hidden, err = head.linear("q_network.backbone.3", hidden)
		if err != nil {
			return nil, err
		}
		relu(hidden)
		output, err := head.linear("q_network.head", hidden)
		if err != nil {
			return nil, err
		}
		scores[index] = output[0]
	}
	return scores, nil
}

func (head *Head) validateShapes(workerCount int) error {
	metaMean := head.tensors["model_encoder.meta_mean"]
	if len(metaMean.shape) != 1 {
		return errors.New("model_encoder.meta_mean must be one-dimensional")
	}
	metaDimension := metaMean.shape[0]
	metaHidden := head.tensors["model_encoder.meta_mlp.0.bias"]
	if len(metaHidden.shape) != 1 {
		return errors.New("model_encoder.meta_mlp.0.bias must be one-dimensional")
	}
	metaHiddenDimension := metaHidden.shape[0]
	residual := head.tensors["model_encoder.residual_embed.weight"]
	if len(residual.shape) != 2 || residual.shape[0] != workerCount {
		return errors.New(
			"model_encoder.residual_embed.weight worker dimension is invalid",
		)
	}
	residualDimension := residual.shape[1]
	architecture := &head.architecture
	expected := map[string][]int{
		"model_encoder.meta_mean":             {metaDimension},
		"model_encoder.meta_std":              {metaDimension},
		"model_encoder.all_metas":             {workerCount, metaDimension},
		"model_encoder.meta_mlp.0.weight":     {metaHiddenDimension, metaDimension},
		"model_encoder.meta_mlp.0.bias":       {metaHiddenDimension},
		"model_encoder.meta_mlp.2.weight":     {metaHiddenDimension, metaHiddenDimension},
		"model_encoder.meta_mlp.2.bias":       {metaHiddenDimension},
		"model_encoder.residual_embed.weight": {workerCount, residualDimension},
		"model_encoder.output_proj.0.weight":  {architecture.ArmEmbeddingDimension, metaHiddenDimension + residualDimension},
		"model_encoder.output_proj.0.bias":    {architecture.ArmEmbeddingDimension},
		"model_encoder.output_proj.1.weight":  {architecture.ArmEmbeddingDimension},
		"model_encoder.output_proj.1.bias":    {architecture.ArmEmbeddingDimension},
		"q_network.backbone.0.weight":         {architecture.HiddenDimensions[0], architecture.JointInputDimension},
		"q_network.backbone.0.bias":           {architecture.HiddenDimensions[0]},
		"q_network.backbone.3.weight":         {architecture.HiddenDimensions[1], architecture.HiddenDimensions[0]},
		"q_network.backbone.3.bias":           {architecture.HiddenDimensions[1]},
		"q_network.head.weight":               {1, architecture.HiddenDimensions[1]},
		"q_network.head.bias":                 {1},
	}
	for name, shape := range expected {
		actual, ok := head.tensors[name]
		if !ok {
			return fmt.Errorf("missing required tensor %q", name)
		}
		if !equalShape(actual.shape, shape) {
			return fmt.Errorf(
				"tensor %q has shape %v; expected %v",
				name,
				actual.shape,
				shape,
			)
		}
	}
	return nil
}

func (head *Head) armEmbedding(index int) ([]float32, error) {
	meta, err := tensorRow(head.tensors["model_encoder.all_metas"], index)
	if err != nil {
		return nil, err
	}
	metaEmbedding, err := head.linear("model_encoder.meta_mlp.0", meta)
	if err != nil {
		return nil, err
	}
	relu(metaEmbedding)
	metaEmbedding, err = head.linear("model_encoder.meta_mlp.2", metaEmbedding)
	if err != nil {
		return nil, err
	}
	residual, err := tensorRow(
		head.tensors["model_encoder.residual_embed.weight"],
		index,
	)
	if err != nil {
		return nil, err
	}
	input := append(metaEmbedding, residual...)
	output, err := head.linear("model_encoder.output_proj.0", input)
	if err != nil {
		return nil, err
	}
	if err := head.layerNorm("model_encoder.output_proj.1", output); err != nil {
		return nil, err
	}
	return output, nil
}

func (head *Head) linear(prefix string, input []float32) ([]float32, error) {
	weight, ok := head.tensors[prefix+".weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", prefix+".weight")
	}
	bias, ok := head.tensors[prefix+".bias"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", prefix+".bias")
	}
	if len(weight.shape) != 2 || weight.shape[1] != len(input) {
		return nil, fmt.Errorf(
			"%s expected %d inputs, got %d",
			prefix,
			weight.shape[1],
			len(input),
		)
	}
	if !equalShape(bias.shape, []int{weight.shape[0]}) {
		return nil, fmt.Errorf("%s bias shape is invalid", prefix)
	}
	output := make([]float32, weight.shape[0])
	for row := range output {
		sum := bias.values[row]
		offset := row * len(input)
		for column, value := range input {
			sum = fma32(weight.values[offset+column], value, sum)
		}
		output[row] = sum
	}
	return output, nil
}

func (head *Head) layerNorm(prefix string, values []float32) error {
	if len(values) == 0 {
		return errors.New("cannot layer-normalize an empty vector")
	}
	weight := head.tensors[prefix+".weight"]
	bias := head.tensors[prefix+".bias"]
	if !equalShape(weight.shape, []int{len(values)}) ||
		!equalShape(bias.shape, []int{len(values)}) {
		return fmt.Errorf("%s layer normalization shape is invalid", prefix)
	}
	var mean float32
	for _, value := range values {
		mean += value
	}
	mean /= float32(len(values))
	var variance float32
	for _, value := range values {
		delta := value - mean
		variance += delta * delta
	}
	variance /= float32(len(values))
	denominator := float32(math.Sqrt(float64(variance + 1e-5)))
	for index := range values {
		values[index] = ((values[index]-mean)/denominator)*weight.values[index] +
			bias.values[index]
	}
	return nil
}

func tensorRow(value tensor, index int) ([]float32, error) {
	if len(value.shape) != 2 || index < 0 || index >= value.shape[0] {
		return nil, fmt.Errorf("invalid row %d for tensor shape %v", index, value.shape)
	}
	width := value.shape[1]
	return value.values[index*width : (index+1)*width], nil
}

func normalize(values []float32) error {
	var squaredNorm float32
	for _, value := range values {
		if float32IsNaNOrInf(value) {
			return errors.New("history embedding contains a non-finite value")
		}
		squaredNorm += value * value
	}
	norm := float32(math.Sqrt(float64(squaredNorm)))
	if float32IsNaNOrInf(norm) || norm <= 0 {
		return errors.New("history embedding norm must be finite and positive")
	}
	for index := range values {
		values[index] /= norm
	}
	return nil
}

func relu(values []float32) {
	for index := range values {
		if values[index] < 0 {
			values[index] = 0
		}
	}
}

func fma32(left, right, sum float32) float32 {
	return float32(math.FMA(float64(left), float64(right), float64(sum)))
}

func equalShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
