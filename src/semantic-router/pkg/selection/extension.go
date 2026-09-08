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

package selection

// ExtensionKey identifies one entry in an Extensions bag. Owners must
// namespace their keys with the package that defines them, for example
// "example.com/router-ext.selector-input", so two independent extensions
// cannot collide.
type ExtensionKey string

// Extensions is an opaque, request-scoped bag carried by SelectionContext
// and SelectionResult on behalf of a Selector registered through
// Registry.Register outside the default set.
//
// The bag carries data across the existing Selector boundary. It is not a
// discovery or activation mechanism: registration already exists, and this
// changes nothing about it.
//
// Contract:
//
//   - The core is blind to the contents. No code in this repository reads a
//     value, branches on a key, or gives any key a meaning. The bag exists so
//     such a Selector can carry its own typed input into Select and its own
//     typed trace back out, without adding a field or an import to the shared
//     structs.
//   - The bag is nil by default. Nothing allocates it until a caller sets a
//     key, so the zero value costs one nil pointer.
//   - Values are not serialised. The router never writes them to logs,
//     metrics, traces, or any API response. An owner that wants a value in
//     telemetry must emit it from its own code.
//   - The bag is a map, so a shallow copy of the enclosing struct shares it.
//     Treat values as request-scoped and do not mutate them after Select
//     returns. Helpers that rebuild a struct field by field must copy the
//     Extensions field across.
//   - Keys and values are owned by whoever sets them. Two extensions must not
//     read each other's keys.
type Extensions map[ExtensionKey]any

// Extension reads the value stored at key and asserts it to T. It reports
// false when the bag is nil, the key is absent, or the stored value has a
// different type, and returns the zero value of T in each of those cases.
func Extension[T any](extensions Extensions, key ExtensionKey) (T, bool) {
	var zero T
	if extensions == nil {
		return zero, false
	}
	value, found := extensions[key]
	if !found {
		return zero, false
	}
	typed, ok := value.(T)
	if !ok {
		return zero, false
	}
	return typed, true
}

// SetExtension stores value at key, allocating the bag on first use.
// It is a no-op on a nil context.
func (c *SelectionContext) SetExtension(key ExtensionKey, value any) {
	if c == nil {
		return
	}
	if c.Extensions == nil {
		c.Extensions = make(Extensions, 1)
	}
	c.Extensions[key] = value
}

// SetExtension stores value at key, allocating the bag on first use.
// It is a no-op on a nil result.
func (r *SelectionResult) SetExtension(key ExtensionKey, value any) {
	if r == nil {
		return
	}
	if r.Extensions == nil {
		r.Extensions = make(Extensions, 1)
	}
	r.Extensions[key] = value
}
