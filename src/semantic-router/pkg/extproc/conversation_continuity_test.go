//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Conversation continuity: the provider reads the conversation the client
// sent.
//
// Real client bodies run through the real request phases, headers and then
// body, on a router built from a config file: the scrubbed captures under
// testdata/continuity, with the headers they were sent with, and the captured
// Messages bodies of the v5 golden corpus. The provider-bound body is then
// compared with the client's. The comparison walks the raw JSON rather than
// the codec under test. It keeps every message's role and every block's type,
// tool link and cache breakpoint, in order. Only a transform the case names
// may change that sequence.
//
// A body the cell cannot serve must be refused with a 400, before any policy
// decision, and must not reach a provider. One example is a message-threads
// continuation, which expects the history to be held server-side
// (req_thread_continuation.go). Another is a tool result whose call was cut
// off. The cut-down variants of each capture check the second.
//
// A case directory holds case.json and headers.json. Its body is client.json,
// or a fixture another test already keeps, named by the case's body member,
// so a capture is stored once.
const continuityDir = "testdata/continuity"

// orphanToolResultRefusal starts the message of the cell's 400 for a tool
// result with no call before it. Claude Code answers this 400 by resending the
// whole conversation, which is why a thread continuation gets it too.
const orphanToolResultRefusal = "tool result must follow its request tool call or carry a retained-history link"

// The transforms the cell may apply to a conversation. A case names the ones
// it allows. A transform may leave the sequence unchanged, but a change no
// named transform explains fails.
const (
	// The steer of an "up" action: one text block appended to the last
	// message.
	transformSteerTailText = "steer_tail_text"
	// Messages has no developer role, so a developer message is sent as a
	// system message.
	transformDeveloperAsSystem = "developer_as_system"
	// Claude Code's mid-conversation tool changes leave the transcript for
	// the top-level tools on OpenRouter (req_filter_tool_changes.go). A
	// system message that is left empty goes too.
	transformToolChangesToTools = "tool_changes_to_tools"
	// An arm reached over Messages is sent no thinking from an earlier
	// assistant turn the cell did not serve on its model (rayline_arc_thinking_signature.go):
	// a captured conversation's thinking was signed elsewhere, so every
	// assistant message loses its thinking but the active tool loop's latest
	// on Claude, and a message left empty goes.
	transformUnservedThinkingStripped = "unserved_thinking_stripped"
	// A worker that is not Claude loses the active loop's thinking too.
	transformUnservedLoopThinkingStripped = "unserved_loop_thinking_stripped"
)

type continuityCase struct {
	Description string `json:"description"`
	Client      string `json:"client"`
	Captured    string `json:"captured"`
	Path        string `json:"path"`
	// Body names the body file, relative to the case directory, when it is
	// not the case's own client.json.
	Body string `json:"body,omitempty"`
	// Expect is "served" or "refused".
	Expect string `json:"expect"`
	// RefusalBody names the exact refusal, relative to the case directory. A
	// refusal always starts with orphanToolResultRefusal.
	RefusalBody string   `json:"refusal_body,omitempty"`
	Transforms  []string `json:"transforms,omitempty"`
	// DroppedFields lists the top-level members the cell does not forward. A
	// member of the client body that is neither forwarded nor listed here
	// fails the case: a member the cell ignores is written down, not
	// discovered in production.
	DroppedFields []string `json:"dropped_fields,omitempty"`
}

type continuityFixture struct {
	Name    string
	Case    continuityCase
	Client  []byte
	Headers map[string]string
	// Refusal is the exact refusal body, when the case names one.
	Refusal []byte
}

func loadContinuityFixtures(t *testing.T) []continuityFixture {
	t.Helper()
	entries, err := os.ReadDir(continuityDir)
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []continuityFixture
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(continuityDir, entry.Name())
		fixture := continuityFixture{Name: entry.Name()}
		for file, into := range map[string]any{"case.json": &fixture.Case, "headers.json": &fixture.Headers} {
			raw, err := os.ReadFile(filepath.Join(dir, file))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, into); err != nil {
				t.Fatalf("%s/%s: %v", entry.Name(), file, err)
			}
		}
		body := "client.json"
		if fixture.Case.Body != "" {
			body = fixture.Case.Body
		}
		if fixture.Client, err = os.ReadFile(filepath.Join(dir, body)); err != nil {
			t.Fatal(err)
		}
		// The captures name their own model; the route is the auto alias.
		// Only that value changes, so every other byte is the capture's.
		if fixture.Client, err = sjson.SetBytes(fixture.Client, "model", "auto"); err != nil {
			t.Fatal(err)
		}
		if fixture.Case.RefusalBody != "" {
			if fixture.Refusal, err = os.ReadFile(filepath.Join(dir, fixture.Case.RefusalBody)); err != nil {
				t.Fatal(err)
			}
			fixture.Refusal = bytes.TrimSpace(fixture.Refusal)
		}
		if fixture.Case.Expect != "served" && fixture.Case.Expect != "refused" {
			t.Fatalf("%s: expect = %q, want served or refused", entry.Name(), fixture.Case.Expect)
		}
		fixtures = append(fixtures, fixture)
	}
	if len(fixtures) == 0 {
		t.Fatal("the continuity corpus is empty")
	}
	return fixtures
}

// continuityActions are the v5 actions a captured body is dispatched with.
// Each arm reads Messages: Opus natively, and GLM at the neutral level and at
// "up", where the steer is written into the conversation.
var continuityActions = []struct {
	Name, ID string
}{
	{"opus", v5OpusAction},
	{"glm-none", v5GLMNone},
	{"glm-up", v5GLMUp},
}

// lastMessageOnlyFailure is how the cell fails a Messages body left with
// only a trailing system message: no turn for the thinking control to place.
const lastMessageOnlyFailure = "the messages body has no conversational unit to govern"

// continuityResult is what one request did. A refusal has a status and a body
// and no provider body. A served turn has the provider body.
type continuityResult struct {
	Status   int
	Refusal  []byte
	Provider []byte
	// Decided reports whether the policy service was asked for a decision.
	Decided bool
}

// continuityTurn runs one client body, sent with its captured headers, through
// the stream loop's request phases (processRequestHeaders and
// processRequestBody) and reads what they send. A served turn is committed as
// a 200 so that the episode's next turn starts clean, as v5Turn does.
func continuityTurn(t *testing.T, router *OpenAIRouter, fake *fakePolicyService, action, episode, path string,
	client []byte, captured map[string]string,
) continuityResult {
	t.Helper()
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
	decisions := len(fake.received())
	if path == "" {
		path = "/v1/messages"
	}
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: fmt.Sprintf("continuity-%s-%d", episode, time.Now().UnixNano()),
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	values := []*core.HeaderValue{
		{Key: ":method", Value: "POST"}, {Key: ":path", Value: path}, {Key: "x-rayline-session", Value: episode},
	}
	names := make([]string, 0, len(captured))
	for name := range captured {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		values = append(values, &core.HeaderValue{Key: name, Value: captured[name]})
	}
	// The stream loop's own phase handlers run each message, and what they
	// send is what Envoy would act on. A phase that returns an error fails
	// the exchange, which Envoy, failing closed, answers with a 500.
	stream := NewMockStream(nil)
	decided := func() bool { return len(fake.received()) > decisions }
	sent := func(phase string) (*ext_proc.ProcessingResponse, bool) {
		if len(stream.Responses) != 1 {
			t.Fatalf("%s phase sent %d responses, want 1", phase, len(stream.Responses))
		}
		response := stream.Responses[0]
		stream.Responses = nil
		return response, response.GetImmediateResponse() != nil
	}
	refusal := func(response *ext_proc.ProcessingResponse) continuityResult {
		immediate := response.GetImmediateResponse()
		return continuityResult{
			Status: int(immediate.GetStatus().GetCode()), Refusal: immediate.GetBody(), Decided: decided(),
		}
	}
	if err := router.processRequestHeaders(stream, &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: values}},
	}, ctx); err != nil {
		t.Fatalf("request headers: %v", err)
	}
	if response, immediate := sent("request headers"); immediate {
		return refusal(response)
	}
	if err := router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: client, EndOfStream: true},
	}, ctx); err != nil {
		return continuityResult{Status: 500, Refusal: []byte(err.Error()), Decided: decided()}
	}
	response, immediate := sent("request body")
	if immediate {
		return refusal(response)
	}
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("the turn was neither refused nor given a provider body")
	}
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
	return continuityResult{Provider: mutation.GetBody(), Decided: decided()}
}

// blockShape is what continuity keeps of one content block: its type, its
// tool link (a call's id or a result's call id), its cache breakpoint, and a
// digest of the rest of it, nested tool-result content included. Two blocks of
// one type that swap, or one whose content is replaced, differ.
type blockShape struct {
	Type   string
	Link   string
	Cache  string
	Digest string
}

type messageShape struct {
	Role   string
	Blocks []blockShape
}

func (m messageShape) String() string {
	parts := make([]string, 0, len(m.Blocks))
	for _, block := range m.Blocks {
		part := block.Type
		if block.Link != "" {
			part += "#" + block.Link
		}
		if block.Cache != "" {
			part += "+cache" + block.Cache
		}
		parts = append(parts, part+"@"+block.Digest)
	}
	return m.Role + ":[" + strings.Join(parts, ",") + "]"
}

// topLevelSystem is the role given to the request's top-level system prompt,
// which the comparison reads as the conversation's first message.
const topLevelSystem = "(system)"

// canonicalJSON re-encodes a value with sorted keys, so two encodings of one
// value compare equal.
func canonicalJSON(raw json.RawMessage) (string, error) {
	// Numbers stay as written, so a changed digit is never rounded away.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(value)
	return string(encoded), err
}

// continuityBlocks reads one content member: a string is one text block, which
// is the same content as far as the provider is concerned.
func continuityBlocks(raw json.RawMessage) ([]blockShape, error) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		raw, _ = json.Marshal([]map[string]string{{"type": "text", "text": text}})
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}
	shapes := make([]blockShape, 0, len(blocks))
	for _, block := range blocks {
		var shape blockShape
		_ = json.Unmarshal(block["type"], &shape.Type)
		var id, callID string
		_ = json.Unmarshal(block["id"], &id)
		_ = json.Unmarshal(block["tool_use_id"], &callID)
		shape.Link = id + callID
		if cache, present := block["cache_control"]; present && string(cache) != "null" {
			canonical, err := canonicalJSON(cache)
			if err != nil {
				return nil, err
			}
			shape.Cache = canonical
		}
		delete(block, "cache_control")
		// A tool result's string content is one text block, as above.
		var nested string
		if json.Unmarshal(block["content"], &nested) == nil {
			block["content"], _ = json.Marshal([]map[string]string{{"type": "text", "text": nested}})
		}
		encoded, err := json.Marshal(block)
		if err != nil {
			return nil, err
		}
		rest, err := canonicalJSON(encoded)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256([]byte(rest))
		shape.Digest = hex.EncodeToString(sum[:4])
		shapes = append(shapes, shape)
	}
	return shapes, nil
}

// conversationShape reads a Messages body's conversation from its raw JSON:
// the top-level system prompt, then every message.
func conversationShape(raw []byte) ([]messageShape, error) {
	var body struct {
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	var shapes []messageShape
	if len(body.System) > 0 && string(body.System) != "null" {
		blocks, err := continuityBlocks(body.System)
		if err != nil {
			return nil, fmt.Errorf("system: %w", err)
		}
		shapes = append(shapes, messageShape{Role: topLevelSystem, Blocks: blocks})
	}
	for index, message := range body.Messages {
		blocks, err := continuityBlocks(message.Content)
		if err != nil {
			return nil, fmt.Errorf("message %d content: %w", index, err)
		}
		shapes = append(shapes, messageShape{Role: message.Role, Blocks: blocks})
	}
	return shapes, nil
}

// expectedConversation is the client's conversation after the transforms that
// rewrite it on the way in. The steer is matched against the provider body
// instead, in conversationKept.
func expectedConversation(client []messageShape, transforms map[string]bool) []messageShape {
	out := make([]messageShape, 0, len(client))
	loopStart, loopEnd := activeLoopShape(client)
	for index, message := range client {
		if transforms[transformUnservedThinkingStripped] && message.Role == "assistant" &&
			(loopEnd < 0 || index < loopStart || index > loopEnd || transforms[transformUnservedLoopThinkingStripped]) {
			kept := make([]blockShape, 0, len(message.Blocks))
			for _, block := range message.Blocks {
				if block.Type != "thinking" && block.Type != "redacted_thinking" {
					kept = append(kept, block)
				}
			}
			if len(kept) == 0 {
				continue
			}
			message.Blocks = kept
		}
		if transforms[transformDeveloperAsSystem] && message.Role == "developer" {
			message.Role = "system"
		}
		if transforms[transformToolChangesToTools] && message.Role == "system" {
			kept := make([]blockShape, 0, len(message.Blocks))
			movedCache := ""
			for _, block := range message.Blocks {
				if block.Type == "tool_addition" || block.Type == "tool_removal" {
					if block.Cache != "" {
						movedCache = block.Cache
					}
					continue
				}
				kept = append(kept, block)
			}
			if len(kept) == 0 {
				continue
			}
			// The breakpoint a dropped block carried moves to the last block
			// its message keeps, unless that block has its own.
			if movedCache != "" && kept[len(kept)-1].Cache == "" {
				kept[len(kept)-1].Cache = movedCache
			}
			message.Blocks = kept
		}
		out = append(out, message)
	}
	return out
}

// activeLoopShape is the active tool loop's latest reply as the indexes of
// its first and last assistant message, or -1, -1: the consecutive assistant
// messages before the trailing results, when those answer any of its calls.
func activeLoopShape(messages []messageShape) (int, int) {
	last := len(messages) - 1
	for last >= 0 && messages[last].Role != "assistant" {
		last--
	}
	if last < 0 || last == len(messages)-1 {
		return -1, -1
	}
	answered := map[string]bool{}
	for _, message := range messages[last+1:] {
		for _, block := range message.Blocks {
			if block.Type == "tool_result" && block.Link != "" {
				answered[block.Link] = true
			}
		}
	}
	start := last
	for start > 0 && messages[start-1].Role == "assistant" {
		start--
	}
	for _, message := range messages[start : last+1] {
		for _, block := range message.Blocks {
			if block.Type == "tool_use" && answered[block.Link] {
				return start, last
			}
		}
	}
	return -1, -1
}

// conversationKept reports how the provider's conversation differs from the
// client's, or nil when only the named transforms tell them apart.
func conversationKept(client, provider []byte, transforms []string) error {
	allowed := map[string]bool{}
	for _, name := range transforms {
		allowed[name] = true
	}
	sent, err := conversationShape(client)
	if err != nil {
		return fmt.Errorf("client body: %w", err)
	}
	received, err := conversationShape(provider)
	if err != nil {
		return fmt.Errorf("provider body: %w", err)
	}
	want := expectedConversation(sent, allowed)
	if allowed[transformSteerTailText] {
		received = withoutSteer(want, received)
	}
	if len(received) != len(want) {
		return fmt.Errorf("the provider received %d messages, the client sent %d (%d after transforms)\n client:   %v\n provider: %v",
			len(received), len(sent), len(want), want, received)
	}
	for index := range want {
		if want[index].String() != received[index].String() {
			return fmt.Errorf("message %d is %v at the provider, %v from the client\n client:   %v\n provider: %v",
				index, received[index], want[index], want, received)
		}
	}
	return nil
}

// withoutSteer removes the steer from the provider's conversation: one text
// block, with no cache breakpoint, after the content of the last user message
// the client sent. Anything else is left for the comparison to report.
func withoutSteer(want, received []messageShape) []messageShape {
	if len(received) != len(want) {
		return received
	}
	for index := len(want) - 1; index >= 0; index-- {
		if want[index].Role != "user" {
			continue
		}
		blocks := received[index].Blocks
		if len(blocks) != len(want[index].Blocks)+1 {
			return received
		}
		if last := blocks[len(blocks)-1]; last.Type != "text" || last.Link != "" || last.Cache != "" {
			return received
		}
		out := append([]messageShape(nil), received...)
		out[index].Blocks = blocks[:len(blocks)-1]
		return out
	}
	return received
}

// toolChangesLanded reports where the provider's top-level tools disagree
// with the client's mid-conversation tool changes. The changes are applied in
// transcript order over the client's own tools, as the fold does: a
// definition adds or replaces its tool, a removal drops the named tool. Every
// tool in force at the end must reach the provider with the definition the
// client gave it.
func toolChangesLanded(client, provider []byte) error {
	var sent struct {
		Tools    []json.RawMessage `json:"tools"`
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	var received struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(client, &sent); err != nil {
		return err
	}
	if err := json.Unmarshal(provider, &received); err != nil {
		return err
	}
	toolName := func(raw json.RawMessage) string {
		var tool struct {
			Name string `json:"name"`
		}
		_ = json.Unmarshal(raw, &tool)
		return tool.Name
	}
	// declared keeps every definition the client gave, so a tool removed and
	// later loaded again by reference comes back as it was declared.
	inForce, declared := map[string]json.RawMessage{}, map[string]json.RawMessage{}
	for _, tool := range sent.Tools {
		inForce[toolName(tool)], declared[toolName(tool)] = tool, tool
	}
	for _, message := range sent.Messages {
		var blocks []struct {
			Type string `json:"type"`
			Tool struct {
				Type       string          `json:"type"`
				Name       string          `json:"name"`
				Definition json.RawMessage `json:"definition"`
			} `json:"tool"`
		}
		if json.Unmarshal(message.Content, &blocks) != nil {
			continue
		}
		for _, block := range blocks {
			switch {
			case block.Type == "tool_addition" && block.Tool.Type == "tool_definition":
				name := toolName(block.Tool.Definition)
				inForce[name], declared[name] = block.Tool.Definition, block.Tool.Definition
			case block.Type == "tool_addition" && block.Tool.Type == "tool_reference":
				// A reference loads a tool the request declares: it is sent as
				// declared, loaded.
				if definition, present := declared[block.Tool.Name]; present {
					loaded, err := sjson.DeleteBytes(definition, "defer_loading")
					if err != nil {
						return err
					}
					inForce[block.Tool.Name] = loaded
				}
			case block.Type == "tool_removal":
				delete(inForce, block.Tool.Name)
			}
		}
	}
	have := map[string]string{}
	for _, tool := range received.Tools {
		canonical, err := canonicalJSON(tool)
		if err != nil {
			return err
		}
		have[toolName(tool)] = canonical
	}
	var problems []string
	for name, definition := range inForce {
		want, err := canonicalJSON(definition)
		if err != nil {
			return err
		}
		got, present := have[name]
		switch {
		case !present:
			problems = append(problems, name+" is missing")
		case got != want:
			problems = append(problems, name+" arrived changed")
		}
		delete(have, name)
	}
	for name := range have {
		problems = append(problems, name+" was never in force")
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("provider tools disagree with the client's tool changes: %s", strings.Join(problems, "; "))
	}
	return nil
}

// dispatchOwnedFields are the top-level members an ARC arm writes itself: its
// thinking control states the thinking and effort the action chose, and the
// client's never reach the provider (the v5 golden corpus,
// claude_code_body_default).
var dispatchOwnedFields = []string{"thinking", "output_config"}

// comparedElsewhere are the members whose provider value another comparison
// checks, absence included: the model is the route's, and the conversation
// and the system prompt are conversationKept's. The tools are
// toolChangesLanded's when the case names the tool-changes transform, since
// a removal can leave none; otherwise they must arrive as sent.
var comparedElsewhere = map[string]bool{"model": true, "messages": true, "system": true}

// fieldsForwarded reports the client's top-level members that do not reach
// the provider as sent: missing, or changed, without being owned by dispatch,
// compared elsewhere, or listed in dropped.
func fieldsForwarded(client, provider []byte, dropped []string, toolsCompared bool) error {
	var sent, received map[string]json.RawMessage
	if err := json.Unmarshal(client, &sent); err != nil {
		return err
	}
	if err := json.Unmarshal(provider, &received); err != nil {
		return err
	}
	known := map[string]bool{}
	for _, name := range dispatchOwnedFields {
		known[name] = true
	}
	for _, name := range dropped {
		known[name] = true
	}
	var problems []string
	for name, value := range sent {
		if known[name] || comparedElsewhere[name] || name == "tools" && toolsCompared {
			continue
		}
		got, forwarded := received[name]
		if !forwarded {
			problems = append(problems, name+" is missing")
			continue
		}
		want, err := canonicalJSON(value)
		if err != nil {
			return err
		}
		if have, err := canonicalJSON(got); err != nil || have != want {
			problems = append(problems, name+" arrived changed")
		}
	}
	for _, name := range dropped {
		if _, forwarded := received[name]; forwarded {
			problems = append(problems, name+" is listed as dropped but forwarded")
		}
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("top-level members not forwarded as sent: %s", strings.Join(problems, "; "))
	}
	return nil
}

// refusedBeforeTheProvider reports whether a result is the cell's 400 for a
// body it cannot serve: an invalid_request_error whose message starts with
// orphanToolResultRefusal, given before any policy decision, with no provider
// body. When exact is set, the refusal must be those bytes.
func refusedBeforeTheProvider(result continuityResult, exact []byte) error {
	if result.Provider != nil {
		return errors.New("the cell served the body; it reached the provider instead of being refused")
	}
	if result.Status != 400 {
		return fmt.Errorf("refused with %d, want 400: %s", result.Status, result.Refusal)
	}
	if result.Decided {
		return errors.New("the policy service was asked to decide a turn the cell refuses")
	}
	var refusal struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(result.Refusal, &refusal); err != nil {
		return fmt.Errorf("refusal body: %w: %s", err, result.Refusal)
	}
	if refusal.Error.Type != "invalid_request_error" || !strings.HasPrefix(refusal.Error.Message, orphanToolResultRefusal) {
		return fmt.Errorf("refusal = %s, want an invalid_request_error starting %q", result.Refusal, orphanToolResultRefusal)
	}
	if exact != nil && !bytes.Equal(result.Refusal, exact) {
		return fmt.Errorf("refusal = %s, want %s", result.Refusal, exact)
	}
	return nil
}

// Every captured body is served with its whole conversation, on every arm,
// or refused before any provider when the cell cannot serve it.
func TestConversationContinuityOfCapturedClients(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	router, fake := v5RouterForCaptures(t)
	for _, fixture := range loadContinuityFixtures(t) {
		for _, action := range continuityActions {
			t.Run(fixture.Name+"/"+action.Name, func(t *testing.T) {
				result := continuityTurn(t, router, fake, action.ID, "continuity-"+fixture.Name+"-"+action.Name,
					fixture.Case.Path, fixture.Client, fixture.Headers)
				if fixture.Case.Expect == "refused" {
					if err := refusedBeforeTheProvider(result, fixture.Refusal); err != nil {
						t.Error(err)
					}
					return
				}
				if result.Provider == nil {
					t.Fatalf("refused with %d: %s", result.Status, result.Refusal)
				}
				transforms := fixture.Case.Transforms
				if action.ID == v5GLMUp {
					transforms = append([]string{transformSteerTailText}, transforms...)
				}
				// Every arm here reads Messages, which takes thinking only signed.
				transforms = append([]string{transformUnservedThinkingStripped}, transforms...)
				if action.ID != v5OpusAction {
					transforms = append([]string{transformUnservedLoopThinkingStripped}, transforms...)
				}
				err := errors.Join(
					conversationKept(fixture.Client, result.Provider, transforms),
					fieldsForwarded(fixture.Client, result.Provider, fixture.Case.DroppedFields,
						slices.Contains(transforms, transformToolChangesToTools)),
				)
				if slices.Contains(transforms, transformToolChangesToTools) {
					err = errors.Join(err, toolChangesLanded(fixture.Client, result.Provider))
				}
				if err != nil {
					t.Error(err)
				}
			})
		}
	}
}

// The ways a captured conversation is cut down. Each cut leaves a body the
// cell can tell is not a whole conversation, so none may be served.
const (
	// cutToolCall keeps only what follows the last tool call: the tool
	// result and anything after it. That is a message-threads continuation
	// with its thread member removed.
	cutToolCall = "tool_call_cut"
	// cutToolUseMessage removes the assistant message that holds the last
	// tool call and keeps everything else.
	cutToolUseMessage = "tool_use_message_removed"
	// cutToLastMessage keeps only the last message.
	cutToLastMessage = "last_message_only"
)

func cutConversation(t *testing.T, client []byte, cut string) ([]byte, bool) {
	t.Helper()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(client, &body); err != nil {
		t.Fatal(err)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatal(err)
	}
	shapes, err := conversationShape(client)
	if err != nil {
		t.Fatal(err)
	}
	if len(shapes) > 0 && shapes[0].Role == topLevelSystem {
		shapes = shapes[1:]
	}
	call := -1
	for index, shape := range shapes {
		for _, block := range shape.Blocks {
			if block.Type == "tool_use" {
				call = index
			}
		}
	}
	switch cut {
	case cutToolCall:
		if call < 0 || call+1 >= len(messages) {
			return nil, false
		}
		messages = messages[call+1:]
	case cutToolUseMessage:
		if call < 0 || call+1 >= len(messages) {
			return nil, false
		}
		messages = append(append([]json.RawMessage(nil), messages[:call]...), messages[call+1:]...)
	case cutToLastMessage:
		if len(messages) < 2 {
			return nil, false
		}
		messages = messages[len(messages)-1:]
	default:
		t.Fatalf("unknown cut %q", cut)
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	body["messages"] = encoded
	out, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return out, true
}

// A captured conversation cut down is never served. A tool result whose call
// was cut off is refused as an orphan, rather than answered as a fragment.
func TestCutDownConversationsAreNeverServed(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	router, fake := v5RouterForCaptures(t)
	ran := map[string]int{}
	for _, fixture := range loadContinuityFixtures(t) {
		if fixture.Case.Expect != "served" {
			continue
		}
		for _, cut := range []string{cutToolCall, cutToolUseMessage, cutToLastMessage} {
			body, applies := cutConversation(t, fixture.Client, cut)
			if !applies {
				continue
			}
			ran[cut]++
			for _, action := range continuityActions {
				t.Run(fixture.Name+"/"+cut+"/"+action.Name, func(t *testing.T) {
					result := continuityTurn(t, router, fake, action.ID, "continuity-cut-"+fixture.Name+"-"+cut+"-"+action.Name,
						fixture.Case.Path, body, fixture.Headers)
					if cut == cutToLastMessage {
						// Only the last message is left: a system message after
						// the turn. It is never served. Today the cell fails the
						// exchange on it rather than refusing it, which the
						// deployed cells, failing closed (failure_mode_allow:
						// false), answer with a 500. That is pinned here so that
						// any other outcome, a forwarded body above all, fails;
						// a 400 is the answer it should get
						// (atlasfutures/semantic-router#326).
						if result.Provider != nil {
							t.Fatal("served a conversation cut down to its last message")
						}
						if result.Status != 500 || !strings.Contains(string(result.Refusal), lastMessageOnlyFailure) {
							t.Errorf("answered %d %q, want the pinned failure %q", result.Status, result.Refusal, lastMessageOnlyFailure)
						}
						return
					}
					if err := refusedBeforeTheProvider(result, nil); err != nil {
						t.Error(err)
					}
				})
			}
		}
	}
	for _, cut := range []string{cutToolCall, cutToolUseMessage, cutToLastMessage} {
		if ran[cut] == 0 {
			t.Errorf("no captured conversation to cut with %s", cut)
		}
	}
}

// The comparison itself: a lost, reordered or unpaired message is reported,
// and a transform passes only when the case names it.
func TestConversationKeptReportsWhatChanged(t *testing.T) {
	const client = `{"system":[{"type":"text","text":"rules"}],"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"go"},{"type":"text","text":"now"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok","cache_control":{"type":"ephemeral"}}]},` +
		`{"role":"developer","content":"note"}]}`
	cases := []struct {
		name, provider, client string
		transforms             []string
		kept                   bool
	}{
		{name: "unchanged", provider: client, kept: true},
		{
			name:     "string content as one text block",
			provider: strings.Replace(client, `"content":"note"`, `"content":[{"type":"text","text":"note"}]`, 1),
			kept:     true,
		},
		{
			name:     "a tool result's string content as one text block",
			provider: strings.Replace(client, `"content":"ok"`, `"content":[{"type":"text","text":"ok"}]`, 1),
			kept:     true,
		},
		{name: "two blocks of one type swapped", provider: strings.Replace(client,
			`{"type":"text","text":"go"},{"type":"text","text":"now"}`, `{"type":"text","text":"now"},{"type":"text","text":"go"}`, 1)},
		{name: "a block's content replaced", provider: strings.Replace(client, `"text":"go"`, `"text":"gone"`, 1)},
		{name: "a tool result's content emptied", provider: strings.Replace(client, `"content":"ok"`, `"content":[]`, 1)},
		{name: "the system prompt emptied", provider: strings.Replace(client, `"system":[{"type":"text","text":"rules"}]`, `"system":[]`, 1)},
		{name: "a breakpoint set to null", provider: strings.Replace(client, `"cache_control":{"type":"ephemeral"}`, `"cache_control":null`, 1)},
		{name: "a large number changed in its last digit", provider: strings.Replace(client, `"input":{}`, `"input":{"id":9007199254740993}`, 1),
			client: strings.Replace(client, `"input":{}`, `"input":{"id":9007199254740992}`, 1)},
		{name: "a breakpoint's ttl changed", provider: strings.Replace(client,
			`"cache_control":{"type":"ephemeral"}`, `"cache_control":{"type":"ephemeral","ttl":"1h"}`, 1)},
		{name: "the history before the result is gone", provider: `{"system":[{"type":"text","text":"rules"}],"messages":[` +
			`{"role":"user","content":[{"type":"text","text":"ok","cache_control":{"type":"ephemeral"}}]},` +
			`{"role":"developer","content":"note"}]}`},
		{name: "the result lost its call", provider: strings.Replace(client, `"tool_use_id":"t1"`, `"tool_use_id":"t2"`, 1)},
		{name: "the breakpoint moved", provider: strings.Replace(client, `,"cache_control":{"type":"ephemeral"}`, ``, 1)},
		{name: "a steer nobody named", provider: strings.Replace(client, `"content":"ok","cache_control":{"type":"ephemeral"}}]`,
			`"content":"ok","cache_control":{"type":"ephemeral"}},{"type":"text","text":"steer"}]`, 1)},
		{
			name: "a named steer", transforms: []string{transformSteerTailText},
			provider: strings.Replace(client, `"content":"ok","cache_control":{"type":"ephemeral"}}]`,
				`"content":"ok","cache_control":{"type":"ephemeral"}},{"type":"text","text":"steer"}]`, 1),
			kept: true,
		},
		{name: "developer sent as system, unnamed", provider: strings.Replace(client, `"developer"`, `"system"`, 1)},
		{
			name: "developer sent as system, named", transforms: []string{transformDeveloperAsSystem},
			provider: strings.Replace(client, `"developer"`, `"system"`, 1), kept: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sent := client
			if tc.client != "" {
				sent = tc.client
			}
			err := conversationKept([]byte(sent), []byte(tc.provider), tc.transforms)
			if tc.kept && err != nil {
				t.Fatalf("reported a change: %v", err)
			}
			if !tc.kept && err == nil {
				t.Fatal("the change was not reported")
			}
		})
	}
}

// Tool changes are applied in transcript order: a tool added and later
// removed is not owed to the provider, and an added tool must arrive with the
// definition the client gave it.
func TestToolChangesLandedFollowsTheTranscript(t *testing.T) {
	const addition = `{"type":"tool_addition","tool":{"type":"tool_definition","definition":` +
		`{"name":"Docs","description":"d","input_schema":{"type":"object"}}}}`
	const removal = `{"type":"tool_removal","tool":{"type":"tool_reference","name":"Docs"}}`
	body := func(blocks ...string) []byte {
		return []byte(`{"tools":[{"name":"Bash","input_schema":{"type":"object"}}],"messages":[` +
			`{"role":"user","content":"go"},{"role":"system","content":[` + strings.Join(blocks, ",") + `]}]}`)
	}
	const bash = `{"name":"Bash","input_schema":{"type":"object"}}`
	const docs = `{"name":"Docs","description":"d","input_schema":{"type":"object"}}`
	cases := []struct {
		name, provider string
		client         []byte
		landed         bool
	}{
		{"an addition lands", `{"tools":[` + bash + `,` + docs + `]}`, body(addition), true},
		{"an addition is missing", `{"tools":[` + bash + `]}`, body(addition), false},
		{"an addition arrives without its schema", `{"tools":[` + bash + `,{"name":"Docs"}]}`, body(addition), false},
		{"an addition later removed is not owed", `{"tools":[` + bash + `]}`, body(addition, removal), true},
		{"a removed tool still sent", `{"tools":[` + bash + `,` + docs + `]}`, body(addition, removal), false},
		{"the last tool removed leaves no tools", `{}`, body(`{"type":"tool_removal","tool":{"type":"tool_reference","name":"Bash"}}`), true},
		{
			"a removed tool loaded again by reference", `{"tools":[` + bash + `,` + docs + `]}`,
			body(addition, removal, `{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}`), true,
		},
		{
			"a removed tool loaded again but not sent", `{"tools":[` + bash + `]}`,
			body(addition, removal, `{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}`), false,
		},
		{
			"a reference loads a deferred tool", `{"tools":[` + bash + `,` + docs + `]}`,
			[]byte(`{"tools":[` + bash + `,{"name":"Docs","description":"d","input_schema":{"type":"object"},"defer_loading":true}],` +
				`"messages":[{"role":"user","content":"go"},{"role":"system","content":[` +
				`{"type":"tool_addition","tool":{"type":"tool_reference","name":"Docs"}}]}]}`), true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := toolChangesLanded(tc.client, []byte(tc.provider))
			if tc.landed && err != nil {
				t.Fatalf("reported: %v", err)
			}
			if !tc.landed && err == nil {
				t.Fatal("the disagreement was not reported")
			}
		})
	}
}
