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

package extproc

// An appended_mismatch (policyDeltaFor) says only that the client's echo of
// the appended reply digests otherwise than the reply VSR staged. To tell
// which part of it differs, the staged turn keeps the reply's shape: per
// message, a digest of its role and of each other member, and per content
// block its type and a digest of each member, all of the canonical form the
// delta digest reads (policyDeltaCanonical). The shape is compared only when
// the turn's reason is appended_mismatch, and what is logged is one value of
// a fixed set (appended_mismatch_part), never content or a digest of it.
// The decision to send a turn in full or as a delta does not read it.

import (
	"bytes"
	"encoding/json"
	"hash/fnv"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Values of appended_mismatch_part.
const (
	// The staged turn carries no shape: it was staged by another instance
	// (the shared record holds digests only) or was too large to keep one.
	appendedMismatchUnavailable = "unavailable"
	appendedMismatchRole        = "role"
	// One side's content is a string, the other's a list of blocks (or
	// null).
	appendedMismatchContentForm = "content_form"
	appendedMismatchBlockCount  = "block_count"
	// The same block types in another order.
	appendedMismatchBlockOrder = "block_order"
	appendedMismatchBlockType  = "block_type"
	// A text block's text, or a string content.
	appendedMismatchText = "text"
	// A thinking block's text, a redacted_thinking block's data, a
	// reasoning item's summary or encrypted content.
	appendedMismatchThinking = "thinking"
	// A tool call's id, name or input (arguments, call_id, tool_calls).
	appendedMismatchToolUse = "tool_use"
	// Any other member of a content block (citations, an unknown member).
	appendedMismatchBlockMetadata = "block_metadata"
	// Any other member of the message (an extra member, a Responses item's
	// status).
	appendedMismatchMessageMetadata = "message_metadata"
	// No part of the shape differs, or a message did not parse.
	appendedMismatchOther = "other"
)

// Bounds past which a staged reply keeps no shape.
const (
	maxPolicyAppendedShapeMessages = 64
	maxPolicyAppendedShapeBlocks   = 512
)

// policyAppendedShape is the appended messages' shapes, behind a pointer so
// a staged turn stays comparable.
type policyAppendedShape struct {
	messages []policyAppendedMessageShape
}

// policyAppendedMessageShape is one appended message as its parts digest.
type policyAppendedMessageShape struct {
	parsed  bool
	role    uint64
	form    byte // 'a' a block list, 's' a string, 'n' null or absent, 'o' other
	content uint64
	blocks  []policyAppendedBlockShape
	members map[string]uint64
}

type policyAppendedBlockShape struct {
	kind    string
	members map[string]uint64
}

func policyPartDigest(value any) uint64 {
	encoded, err := raylinearc.MarshalUnescaped(value)
	if err != nil {
		return 0
	}
	hash := fnv.New64a()
	_, _ = hash.Write(encoded)
	return hash.Sum64()
}

func policyMemberDigests(fields map[string]any, skip ...string) map[string]uint64 {
	members := make(map[string]uint64, len(fields))
	for key, value := range fields {
		if !slices.Contains(skip, key) {
			members[key] = policyPartDigest(value)
		}
	}
	return members
}

// policyAppendedShapeOf is the shape of the appended messages, nil past the
// bounds.
func policyAppendedShapeOf(messages []json.RawMessage) *policyAppendedShape {
	if len(messages) > maxPolicyAppendedShapeMessages {
		return nil
	}
	shape := make([]policyAppendedMessageShape, 0, len(messages))
	blocks := 0
	for _, message := range messages {
		one := policyAppendedMessageShapeOf(message)
		blocks += len(one.blocks)
		if blocks > maxPolicyAppendedShapeBlocks {
			return nil
		}
		shape = append(shape, one)
	}
	return &policyAppendedShape{messages: shape}
}

func policyAppendedMessageShapeOf(message json.RawMessage) policyAppendedMessageShape {
	var object map[string]any
	decoder := json.NewDecoder(bytes.NewReader(policyDeltaCanonical(message)))
	decoder.UseNumber()
	if decoder.Decode(&object) != nil || object == nil {
		return policyAppendedMessageShape{}
	}
	shape := policyAppendedMessageShape{parsed: true, members: policyMemberDigests(object, "role", "content")}
	if role, ok := object["role"]; ok {
		shape.role = policyPartDigest(role)
	}
	switch content := object["content"].(type) {
	case []any:
		shape.form = 'a'
		shape.blocks = make([]policyAppendedBlockShape, 0, len(content))
		for _, block := range content {
			fields, ok := block.(map[string]any)
			if !ok {
				shape.blocks = append(shape.blocks, policyAppendedBlockShape{members: map[string]uint64{"": policyPartDigest(block)}})
				continue
			}
			kind, _ := fields["type"].(string)
			shape.blocks = append(shape.blocks, policyAppendedBlockShape{kind: kind, members: policyMemberDigests(fields, "type")})
		}
	case string:
		shape.form, shape.content = 's', policyPartDigest(content)
	case nil:
		shape.form = 'n'
	default:
		shape.form, shape.content = 'o', policyPartDigest(content)
	}
	return shape
}

// policyMismatchPartOfMember is the part a member that differs stands for;
// metadata is what any other member reads as at its level.
func policyMismatchPartOfMember(key, metadata string) string {
	switch key {
	case "text":
		return appendedMismatchText
	case "thinking", "data", "summary", "encrypted_content":
		return appendedMismatchThinking
	case "id", "name", "input", "arguments", "call_id", "tool_calls", "tool_use_id":
		return appendedMismatchToolUse
	case "type":
		return appendedMismatchBlockType
	default:
		return metadata
	}
}

// policyMismatchPartPriority orders the parts of one block or message that
// differ, so the part reported does not hang on map order.
var policyMismatchPartPriority = []string{
	appendedMismatchBlockType, appendedMismatchToolUse, appendedMismatchThinking,
	appendedMismatchText, appendedMismatchBlockMetadata, appendedMismatchMessageMetadata,
}

// policyMembersMismatch is the highest-priority part whose member differs
// between staged and echoed, "" when none does.
func policyMembersMismatch(staged, echoed map[string]uint64, metadata string) string {
	found := map[string]bool{}
	for key, digest := range staged {
		if other, ok := echoed[key]; !ok || other != digest {
			found[policyMismatchPartOfMember(key, metadata)] = true
		}
	}
	for key := range echoed {
		if _, ok := staged[key]; !ok {
			found[policyMismatchPartOfMember(key, metadata)] = true
		}
	}
	for _, part := range policyMismatchPartPriority {
		if found[part] {
			return part
		}
	}
	return ""
}

// policyAppendedShapeMismatch is the first part, message by message, in
// which the echo differs from the staged shape.
func policyAppendedShapeMismatch(stagedShape, echoedShape *policyAppendedShape) string {
	if stagedShape == nil || echoedShape == nil || len(stagedShape.messages) != len(echoedShape.messages) {
		return appendedMismatchOther
	}
	staged, echoed := stagedShape.messages, echoedShape.messages
	for index := range staged {
		want, got := staged[index], echoed[index]
		if !want.parsed || !got.parsed {
			return appendedMismatchOther
		}
		if want.role != got.role {
			return appendedMismatchRole
		}
		if want.form != got.form {
			return appendedMismatchContentForm
		}
		if want.form == 'a' {
			if part := policyBlocksMismatch(want.blocks, got.blocks); part != "" {
				return part
			}
		} else if want.content != got.content {
			return appendedMismatchText
		}
		if part := policyMembersMismatch(want.members, got.members, appendedMismatchMessageMetadata); part != "" {
			return part
		}
	}
	return appendedMismatchOther
}

func policyBlocksMismatch(staged, echoed []policyAppendedBlockShape) string {
	if len(staged) != len(echoed) {
		return appendedMismatchBlockCount
	}
	kinds := func(blocks []policyAppendedBlockShape) []string {
		out := make([]string, len(blocks))
		for index, block := range blocks {
			out[index] = block.kind
		}
		return out
	}
	wantKinds, gotKinds := kinds(staged), kinds(echoed)
	if !slices.Equal(wantKinds, gotKinds) {
		slices.Sort(wantKinds)
		slices.Sort(gotKinds)
		if slices.Equal(wantKinds, gotKinds) {
			return appendedMismatchBlockOrder
		}
		return appendedMismatchBlockType
	}
	for index := range staged {
		if part := policyMembersMismatch(staged[index].members, echoed[index].members, appendedMismatchBlockMetadata); part != "" {
			return part
		}
	}
	return ""
}

// policyAppendedMismatchPart is which part of the appended reply the
// request's echo of it differs in, for a turn whose delta reason is
// appended_mismatch: the request's messages from the staged reply's start
// to the staged count, against the staged shape.
func policyAppendedMismatchPart(scorer *policyServiceScorer, episodeIDHash string, messages []json.RawMessage) string {
	staged, ok := scorer.staged.get(episodeIDHash)
	if !ok || staged.appendedShape == nil || staged.appendedFrom < 0 ||
		staged.appendedFrom > staged.count || staged.count > len(messages) {
		return appendedMismatchUnavailable
	}
	return policyAppendedShapeMismatch(staged.appendedShape, policyAppendedShapeOf(messages[staged.appendedFrom:staged.count]))
}
