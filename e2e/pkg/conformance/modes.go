package conformance

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/textproto"
	"sort"
	"strconv"
	"strings"
)

// compareExact requires raw byte identity plus the expected status and headers.
// Nothing is normalized: exact is how a same-protocol path proves it did not
// reserialize the body.
func compareExact(want, got Payload) []Mismatch {
	mismatches := compareStatus(want.Status, got.Status)
	mismatches = append(mismatches, compareHeaders(want.Headers, got.Headers, nil)...)

	if !bytes.Equal(want.Body, got.Body) {
		mismatches = append(mismatches, Mismatch{
			Path:   "body",
			Want:   strconv.Itoa(len(want.Body)) + " bytes",
			Got:    strconv.Itoa(len(got.Body)) + " bytes",
			Reason: "raw bytes differ: " + firstByteDifference(want.Body, got.Body),
		})
	}
	return mismatches
}

// compareExactExcept requires structural identity except the declared exclusions.
func compareExactExcept(cmp Comparison, want, got Payload) ([]Mismatch, error) {
	pointers, headers := splitExclusions(cmp.Exclude)

	mismatches := compareStatus(want.Status, got.Status)
	mismatches = append(mismatches, compareHeaders(want.Headers, got.Headers, headers)...)

	body, err := compareBodies(want, got, func(path string, wantValue, gotValue any, out *[]Mismatch) error {
		for _, pointer := range pointers {
			if err := deletePointer(wantValue, pointer); err != nil {
				return err
			}
			if err := deletePointer(gotValue, pointer); err != nil {
				return err
			}
		}
		applyInvariants(cmp.Invariants, wantValue, gotValue)
		diffJSON(path, wantValue, gotValue, nil, out)
		return nil
	}, func(want, got SSEEvent, coordinate string, out *[]Mismatch) {
		if want.ID != got.ID {
			*out = append(*out, Mismatch{Path: coordinate + ".id", Want: want.ID, Got: got.ID, Reason: "event id differs"})
		}
		compareEventData(want, got, coordinate, nil, cmp.Invariants, out)
	})
	if err != nil {
		return nil, err
	}
	return append(mismatches, body...), nil
}

// compareReject requires the declared status, headers, and error body, and nothing else.
func compareReject(cmp Comparison, got Payload) ([]Mismatch, error) {
	mismatches := compareStatus(cmp.Reject.Status, got.Status)
	mismatches = append(mismatches, compareHeaders(cmp.Reject.Headers, got.Headers, nil)...)

	if cmp.Reject.Body == nil {
		return mismatches, nil
	}
	gotBody, err := decodeJSON(got.Body)
	if err != nil {
		return nil, fmt.Errorf("observed rejection body: %w", err)
	}

	// Round-trip the declared body through JSON so both sides use the same number
	// representation as every other comparison.
	wantBody, err := remarshalJSON(cmp.Reject.Body)
	if err != nil {
		return nil, fmt.Errorf("declared reject_body: %w", err)
	}
	diffJSON("", wantBody, gotBody, volatileSet(cmp.Volatile), &mismatches)
	return mismatches, nil
}

// compareBodies routes a body pair to the JSON or SSE comparison, rejecting a pair
// whose encodings disagree.
func compareBodies(
	want, got Payload,
	compareJSON func(path string, want, got any, out *[]Mismatch) error,
	compareEvent func(want, got SSEEvent, coordinate string, out *[]Mismatch),
) ([]Mismatch, error) {
	if want.Stream != got.Stream {
		return []Mismatch{{
			Path:   "body",
			Want:   encodingName(want.Stream),
			Got:    encodingName(got.Stream),
			Reason: "body encoding differs",
		}}, nil
	}

	var mismatches []Mismatch
	if !want.Stream {
		wantValue, err := decodeJSON(want.Body)
		if err != nil {
			return nil, fmt.Errorf("expected body: %w", err)
		}
		gotValue, err := decodeJSON(got.Body)
		if err != nil {
			return nil, fmt.Errorf("observed body: %w", err)
		}
		if err := compareJSON("", wantValue, gotValue, &mismatches); err != nil {
			return nil, err
		}
		return mismatches, nil
	}

	wantEvents, err := ParseSSE(want.Body)
	if err != nil {
		return nil, fmt.Errorf("expected stream: %w", err)
	}
	gotEvents, err := ParseSSE(got.Body)
	if err != nil {
		return nil, fmt.Errorf("observed stream: %w", err)
	}
	if len(wantEvents) != len(gotEvents) {
		return []Mismatch{{
			Path:   "events",
			Want:   strconv.Itoa(len(wantEvents)) + " events",
			Got:    strconv.Itoa(len(gotEvents)) + " events",
			Reason: "event count differs: " + eventNames(wantEvents) + " vs " + eventNames(gotEvents),
		}}, nil
	}

	for i := range wantEvents {
		coordinate := "event[" + strconv.Itoa(i) + "]"
		if wantEvents[i].Name != gotEvents[i].Name {
			mismatches = append(mismatches, Mismatch{
				Path:   coordinate + ".event",
				Want:   wantEvents[i].Name,
				Got:    gotEvents[i].Name,
				Reason: "event type differs",
			})
			continue
		}
		compareEvent(wantEvents[i], gotEvents[i], coordinate, &mismatches)
	}
	return mismatches, nil
}

// compareEventData compares one event's data payload. Structural identity is
// checked as JSON when both sides are JSON, so key order and insignificant
// whitespace do not fail an unchanged stream. exact-except also compares the
// event id; semantic does not, and passes it no volatile set.
func compareEventData(want, got SSEEvent, coordinate string, volatile map[string]struct{}, invariants []Invariant, out *[]Mismatch) {
	path := coordinate + ".data"
	if !want.IsJSON() || !got.IsJSON() {
		if want.Data != got.Data {
			*out = append(*out, Mismatch{Path: path, Want: want.Data, Got: got.Data, Reason: "event data differs"})
		}
		return
	}

	wantValue, wantErr := decodeJSON([]byte(want.Data))
	gotValue, gotErr := decodeJSON([]byte(got.Data))
	if wantErr != nil || gotErr != nil {
		if want.Data != got.Data {
			*out = append(*out, Mismatch{Path: path, Want: want.Data, Got: got.Data, Reason: "event data is not valid JSON and differs"})
		}
		return
	}

	applyInvariants(invariants, wantValue, gotValue)

	before := len(*out)
	diffJSON("", wantValue, gotValue, volatile, out)
	for i := before; i < len(*out); i++ {
		(*out)[i].Path = path + (*out)[i].Path
	}
}

func compareStatus(want, got int) []Mismatch {
	if want == 0 || want == got {
		return nil
	}
	return []Mismatch{{Path: "status", Want: strconv.Itoa(want), Got: strconv.Itoa(got), Reason: "status differs"}}
}

// compareHeaders checks every header the expectation names. Headers the expectation
// does not name are not constrained; excluded names are skipped entirely.
func compareHeaders(want, got map[string]string, exclude map[string]struct{}) []Mismatch {
	var mismatches []Mismatch
	for _, name := range sortedHeaderNames(want) {
		canonical := textproto.CanonicalMIMEHeaderKey(name)
		if _, skipped := exclude[canonical]; skipped {
			continue
		}

		observed, present := lookupHeader(got, canonical)
		switch {
		case !present:
			mismatches = append(mismatches, Mismatch{Path: canonical, Want: want[name], Got: "<absent>", Reason: "header missing"})
		case observed != want[name]:
			mismatches = append(mismatches, Mismatch{Path: canonical, Want: want[name], Got: observed, Reason: "header differs"})
		}
	}
	return mismatches
}

func lookupHeader(headers map[string]string, canonical string) (string, bool) {
	for name, value := range headers {
		if textproto.CanonicalMIMEHeaderKey(name) == canonical {
			return value, true
		}
	}
	return "", false
}

func sortedHeaderNames(headers map[string]string) []string {
	names := make([]string, 0, len(headers))
	for name := range headers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// splitExclusions separates JSON Pointers from header names. A leading "/" makes an
// entry a pointer, which is why cases.yaml can mix "/model" and "Authorization" in
// one allowed_patches list.
func splitExclusions(exclude []string) (pointers []string, headers map[string]struct{}) {
	headers = make(map[string]struct{}, len(exclude))
	for _, entry := range exclude {
		if strings.HasPrefix(entry, pointerPrefix) {
			pointers = append(pointers, entry)
			continue
		}
		headers[textproto.CanonicalMIMEHeaderKey(entry)] = struct{}{}
	}
	return pointers, headers
}

func volatileSet(pointers []string) map[string]struct{} {
	if len(pointers) == 0 {
		return nil
	}

	set := make(map[string]struct{}, len(pointers))
	for _, pointer := range pointers {
		set[pointer] = struct{}{}
	}
	return set
}

func remarshalJSON(value map[string]any) (any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return decodeJSON(raw)
}

func encodingName(stream bool) string {
	if stream {
		return "sse"
	}
	return "json"
}

func eventNames(events []SSEEvent) string {
	names := make([]string, 0, len(events))
	for _, event := range events {
		if event.Name == "" {
			names = append(names, "message")
			continue
		}
		names = append(names, event.Name)
	}
	return "[" + strings.Join(names, " ") + "]"
}

func firstByteDifference(want, got []byte) string {
	limit := len(want)
	if len(got) < limit {
		limit = len(got)
	}
	for i := range limit {
		if want[i] != got[i] {
			return "first difference at byte " + strconv.Itoa(i)
		}
	}
	return "one body is a prefix of the other"
}
