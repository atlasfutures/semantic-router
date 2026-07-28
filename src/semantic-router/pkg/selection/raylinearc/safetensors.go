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
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

const (
	maxSafeTensorFileBytes = 256 << 20
	maxSafeTensorHeader    = 8 << 20
)

type tensor struct {
	shape  []int
	values []float32
}

type tensorDescriptor struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type decodedTensorDescriptor struct {
	shape       []int
	dataOffsets []uint64
}

type tensorRange struct {
	name  string
	start uint64
	end   uint64
}

func decodeSafeTensors(data []byte, expected []string) (map[string]tensor, error) {
	if len(data) < 8 {
		return nil, errors.New("SafeTensors file is shorter than its length prefix")
	}
	headerLength := binary.LittleEndian.Uint64(data[:8])
	if headerLength == 0 || headerLength > maxSafeTensorHeader {
		return nil, fmt.Errorf("invalid SafeTensors header length %d", headerLength)
	}
	// A slice length is nonnegative and therefore losslessly representable as
	// uint64 on every supported Go platform.
	dataLength := uint64(len(data)) //nolint:gosec
	if headerLength > dataLength-8 {
		return nil, errors.New("SafeTensors header extends beyond the file")
	}
	headerEnd := uint64(8) + headerLength
	rawEntries, err := decodeHeaderObject(data[8:headerEnd])
	if err != nil {
		return nil, fmt.Errorf("decode SafeTensors header: %w", err)
	}
	expectedSet := make(map[string]struct{}, len(expected))
	for _, name := range expected {
		if name == "" || name == "__metadata__" {
			return nil, fmt.Errorf("invalid expected tensor name %q", name)
		}
		if _, duplicate := expectedSet[name]; duplicate {
			return nil, fmt.Errorf("duplicate expected tensor %q", name)
		}
		expectedSet[name] = struct{}{}
	}

	descriptors := make(map[string]decodedTensorDescriptor, len(expected))
	ranges := make([]tensorRange, 0, len(expected))
	for name, raw := range rawEntries {
		if name == "__metadata__" {
			var metadata map[string]string
			if err := json.Unmarshal(raw, &metadata); err != nil {
				return nil, fmt.Errorf("invalid SafeTensors metadata: %w", err)
			}
			continue
		}
		if _, ok := expectedSet[name]; !ok {
			return nil, fmt.Errorf("unexpected tensor %q", name)
		}
		var descriptor tensorDescriptor
		if err := decodeStrictJSON(raw, &descriptor); err != nil {
			return nil, fmt.Errorf("tensor %q descriptor: %w", name, err)
		}
		if descriptor.DType != "F32" {
			return nil, fmt.Errorf(
				"tensor %q dtype must be F32, got %q",
				name,
				descriptor.DType,
			)
		}
		if len(descriptor.Shape) == 0 {
			return nil, fmt.Errorf("tensor %q must have at least one dimension", name)
		}
		if len(descriptor.DataOffsets) != 2 {
			return nil, fmt.Errorf("tensor %q must have two data offsets", name)
		}
		start, end := descriptor.DataOffsets[0], descriptor.DataOffsets[1]
		if start > end {
			return nil, fmt.Errorf("tensor %q data offsets are reversed", name)
		}
		elements, shape, err := checkedElementCount(descriptor.Shape)
		if err != nil {
			return nil, fmt.Errorf("tensor %q shape: %w", name, err)
		}
		if elements > math.MaxUint64/4 || end-start != elements*4 {
			return nil, fmt.Errorf("tensor %q byte length does not match its shape", name)
		}
		if end > uint64(len(data))-headerEnd {
			return nil, fmt.Errorf("tensor %q data extends beyond the file", name)
		}
		descriptors[name] = decodedTensorDescriptor{
			shape:       shape,
			dataOffsets: descriptor.DataOffsets,
		}
		ranges = append(ranges, tensorRange{name: name, start: start, end: end})
	}
	for name := range expectedSet {
		if _, ok := descriptors[name]; !ok {
			return nil, fmt.Errorf("missing required tensor %q", name)
		}
	}
	sort.Slice(ranges, func(left, right int) bool {
		if ranges[left].start == ranges[right].start {
			return ranges[left].end < ranges[right].end
		}
		return ranges[left].start < ranges[right].start
	})
	if len(ranges) == 0 || ranges[0].start != 0 {
		return nil, errors.New("SafeTensors data must start at offset zero")
	}
	for index := 1; index < len(ranges); index++ {
		if ranges[index].start < ranges[index-1].end {
			return nil, fmt.Errorf(
				"tensor %q overlaps tensor %q",
				ranges[index].name,
				ranges[index-1].name,
			)
		}
		if ranges[index].start != ranges[index-1].end {
			return nil, fmt.Errorf(
				"gap before tensor %q is not allowed",
				ranges[index].name,
			)
		}
	}
	if ranges[len(ranges)-1].end != uint64(len(data))-headerEnd {
		return nil, errors.New(
			"SafeTensors data has unclaimed trailing bytes",
		)
	}

	tensors := make(map[string]tensor, len(descriptors))
	for name, descriptor := range descriptors {
		start := headerEnd + descriptor.dataOffsets[0]
		end := headerEnd + descriptor.dataOffsets[1]
		raw := data[start:end]
		values := make([]float32, len(raw)/4)
		for index := range values {
			bits := binary.LittleEndian.Uint32(raw[index*4 : index*4+4])
			value := math.Float32frombits(bits)
			if float32IsNaNOrInf(value) {
				return nil, fmt.Errorf("tensor %q contains non-finite values", name)
			}
			values[index] = value
		}
		tensors[name] = tensor{
			shape:  append([]int(nil), descriptor.shape...),
			values: values,
		}
	}
	return tensors, nil
}

func decodeHeaderObject(data []byte) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return nil, errors.New("SafeTensors header must be a JSON object")
	}
	result := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok {
			return nil, errors.New("SafeTensors tensor name must be a string")
		}
		if _, duplicate := result[name]; duplicate {
			return nil, fmt.Errorf("duplicate SafeTensors entry %q", name)
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return nil, err
		}
		result[name] = raw
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("unexpected trailing SafeTensors header value")
		}
		return nil, err
	}
	return result, nil
}

func checkedElementCount(dimensions []uint64) (uint64, []int, error) {
	count := uint64(1)
	shape := make([]int, len(dimensions))
	maxInt := uint64(^uint(0) >> 1)
	for index, dimension := range dimensions {
		if dimension == 0 {
			return 0, nil, errors.New("zero-sized dimensions are not supported")
		}
		if dimension > maxInt {
			return 0, nil, errors.New("dimension exceeds platform integer range")
		}
		if count > math.MaxUint64/dimension {
			return 0, nil, errors.New("element count overflows uint64")
		}
		count *= dimension
		// The platform-range guard above makes this conversion lossless.
		shape[index] = int(dimension) //nolint:gosec
	}
	return count, shape, nil
}
