package extproc

import (
	"regexp"
	"strings"
)

// upstreamErrorFieldPattern matches a candidate request-member path a provider
// names in an invalid-request message, such as messages[3].content[0].type or
// input[12].name: a top-level member of the Chat, Responses or Messages
// request, then member names and integer indices. The Messages error shape a
// provider answers through carries no param member, so the path in the
// message is the only place the rejected field is stated.
var upstreamErrorFieldPattern = regexp.MustCompile(
	`\b(?:messages|input|tools|tool_choice|system|instructions|max_tokens|max_output_tokens|max_completion_tokens|` +
		`reasoning|reasoning_effort|thinking|text|response_format|parallel_tool_calls|temperature|top_p|top_k|stop|` +
		`stop_sequences|metadata|include|prompt_cache_key|output_config|functions|function_call)` +
		`(?:\[\d{1,6}\])*(?:\.[a-z_]{1,40}(?:\[\d{1,6}\])*){0,8}`)

// upstreamErrorFieldMembers are the nested member names a logged path may
// contain. A provider can echo a rejected value that looks like a path
// ('tools.payroll.corp.internal'), so a path with any segment outside this
// contract vocabulary is not logged at all.
var upstreamErrorFieldMembers = map[string]bool{}

func init() {
	for _, name := range strings.Fields(`
		content type text role name id function functions arguments parameters description strict
		tool_calls tool_call_id tool_use_id tool_result tool_use call_id output input input_schema
		image_url url detail source media_type data file file_id file_data filename input_audio audio
		cache_control ttl citations is_error signature thinking redacted_thinking budget_tokens effort
		summary format json_schema schema properties items required additionalproperties enum
		refusal annotations encrypted_content status disable_parallel_tool_use
	`) {
		upstreamErrorFieldMembers[name] = true
	}
}

// maxUpstreamErrorFields bounds the paths one log line carries.
const maxUpstreamErrorFields = 4

// upstreamErrorFields lists the distinct request-member paths named in a
// provider's message, in order, at most maxUpstreamErrorFields; nil when it
// names none. Every nested segment must be contract vocabulary
// (upstreamErrorFieldMembers), so what is logged is never a value the client
// wrote. A bare member name ("text", "stop") is an ordinary word in a
// message, so it counts only quoted ('max_tokens'); a path with an index or a
// segment counts anywhere.
func upstreamErrorFields(message string) []string {
	var fields []string
	seen := map[string]bool{}
	for _, at := range upstreamErrorFieldPattern.FindAllStringIndex(message, -1) {
		field := message[at[0]:at[1]]
		// The match stops at the first character outside the path grammar; a
		// path that continues into other text (a dotted hostname, a dash) is a
		// value, not a member path.
		if continuesPastPath(message, at[1]) {
			continue
		}
		if !contractPath(field) {
			continue
		}
		quoted := at[0] > 0 && at[1] < len(message) &&
			strings.ContainsRune("'\"`", rune(message[at[0]-1])) && message[at[1]] == message[at[0]-1]
		if !strings.ContainsAny(field, "[.") && !quoted {
			continue
		}
		if seen[field] {
			continue
		}
		seen[field] = true
		fields = append(fields, field)
		if len(fields) == maxUpstreamErrorFields {
			break
		}
	}
	return fields
}

// continuesPastPath reports whether the token goes on past the matched path
// at end: a letter, digit, '-', ':', '@' or '/', or a '.' followed by one of
// those (a sentence-ending period does not count).
func continuesPastPath(message string, end int) bool {
	tokenChar := func(c byte) bool {
		return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.IndexByte("_-:@/", c) >= 0
	}
	if end >= len(message) {
		return false
	}
	if message[end] == '.' {
		return end+1 < len(message) && tokenChar(message[end+1])
	}
	return tokenChar(message[end])
}

// contractPath reports whether every nested member segment of field is in the
// contract vocabulary; the first segment is already constrained by the pattern.
func contractPath(field string) bool {
	segments := strings.Split(field, ".")
	for _, segment := range segments[1:] {
		if i := strings.IndexByte(segment, '['); i >= 0 {
			segment = segment[:i]
		}
		if !upstreamErrorFieldMembers[segment] {
			return false
		}
	}
	return true
}
