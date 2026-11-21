package extproc

import (
	"regexp"
	"strings"
)

// upstreamErrorFieldPattern matches a request-member path a provider names in
// an invalid-request message, such as messages[3].content[0].type or
// input[12].name. Its first segment must be a top-level request member of the
// Chat, Responses or Messages contract, and every later segment a lowercase
// member name or an integer index, so a match is contract vocabulary and
// never a value the client wrote. The Messages error shape a provider answers
// through carries no param member, so the path in the message is the only
// place the rejected field is stated.
var upstreamErrorFieldPattern = regexp.MustCompile(
	`\b(?:messages|input|tools|tool_choice|system|instructions|max_tokens|max_output_tokens|max_completion_tokens|` +
		`reasoning|reasoning_effort|thinking|text|response_format|parallel_tool_calls|temperature|top_p|top_k|stop|` +
		`stop_sequences|metadata|include|prompt_cache_key|output_config|functions|function_call)` +
		`(?:\[\d{1,6}\])*(?:\.[a-z_]{1,40}(?:\[\d{1,6}\])*){0,8}`)

// maxUpstreamErrorFields bounds the paths one log line carries.
const maxUpstreamErrorFields = 4

// upstreamErrorFields lists the distinct request-member paths named in a
// provider's message, in order, at most maxUpstreamErrorFields; nil when it
// names none. A bare member name ("text", "stop") is an ordinary word in a
// message, so it counts only quoted ('max_tokens'); a path with an index or a
// segment counts anywhere.
func upstreamErrorFields(message string) []string {
	var fields []string
	seen := map[string]bool{}
	for _, at := range upstreamErrorFieldPattern.FindAllStringIndex(message, -1) {
		field := message[at[0]:at[1]]
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
