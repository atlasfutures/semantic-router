package llmprotocol

import (
	"encoding/json"
	"strings"
)

// An Anthropic-defined tool is declared by type alone:
//
//	{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
//
// Claude was trained on its schema, so the API fills the schema in and the
// declaration carries none. The caller runs it: the model emits an ordinary
// tool_use, the client answers with a tool_result, exactly as for a custom
// tool. That is what separates it from a server tool, which the source API
// runs itself and which no other provider can run at all.
//
// A target other than Messages has no type table to fill the schema from, so
// the declaration is materialized: the schema Anthropic documents for the type
// is written out as a callable tool under the caller's name. The model on the
// other end sees the tool it would have seen on Claude, and the tool_use it
// emits is one the client already handles. Dropping the declaration instead
// sent the arm a history full of calls to a tool it was never shown, and
// gating it demanded a capability no arm can hold, since the arm never runs
// the tool either way. Every Workshop agent turn of 2026-09-14 failed that
// gate (memex-desktop#6902).
//
// Only types whose schema is documented and stable are listed. Computer use
// and the 2026 toolsets carry declaration-level parameters, a display size or
// a member list, that no schema table can stand in for. They remain server
// tools for routing purposes and gate the turn to an arm that takes the
// declaration as written.

// AnthropicDefinedTool is the documented definition behind one tool type.
type AnthropicDefinedTool struct {
	// Name is the name Anthropic requires the declaration to carry. A caller
	// that states its own name keeps it.
	Name        string
	Description string
	InputSchema json.RawMessage
}

// AnthropicDefined reports the documented definition of a tool declared by one
// of Anthropic's tool types, and false for a custom tool and a server tool.
//
// A dated revision the table has not seen, text_editor_20260401 say, resolves
// to the newest tabled revision of its family. Anthropic ships one every few
// months, and the day it ships is otherwise the day every Workshop agent turn
// is refused again. The older schema is an approximation the diagnostics
// count as one; the client's handler accepts the union of the revisions, so a
// call built on it is still one the client can run.
func (tool Tool) AnthropicDefined() (AnthropicDefinedTool, bool) {
	if definition, known := anthropicDefinedTools[tool.Type]; known {
		return definition, true
	}
	for _, family := range anthropicDefinedToolFamilies {
		if strings.HasPrefix(tool.Type, family.prefix) && datedRevision(tool.Type, family.prefix) {
			return anthropicDefinedTools[family.newest], true
		}
	}
	return AnthropicDefinedTool{}, false
}

// datedRevision reports whether what follows the family prefix is a date
// stamp and nothing else, the shape every Anthropic tool type takes. A type
// that merely starts with the prefix, text_editor_toolset_2027 say, is a new
// kind and stays a server tool until someone reads what it is.
func datedRevision(toolType, prefix string) bool {
	stamp := strings.TrimPrefix(toolType, prefix)
	if len(stamp) != 8 {
		return false
	}
	for _, r := range stamp {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// anthropicDefinedToolFamilies maps a family prefix to its newest tabled
// revision. Adding a revision to the table means moving `newest` here, which
// TestAnthropicDefinedToolsAreCallableDeclarations checks.
var anthropicDefinedToolFamilies = []struct{ prefix, newest string }{
	{prefix: "text_editor_", newest: "text_editor_20250728"},
	{prefix: "bash_", newest: "bash_20250124"},
	{prefix: "memory_", newest: "memory_20250818"},
}

// Materialized returns the tool as a callable declaration a target without
// Anthropic's type table can express. A tool that is not Anthropic-defined
// comes back unchanged. What the caller stated wins over what the type
// documents: the type fills in only what the declaration left out.
//
// A strict flag is kept only beside the caller's own schema. The documented
// schema lists optional members and no additionalProperties, which is not a
// schema OpenAI's strict mode accepts, and a 400 at the provider is the
// failure this table exists to prevent.
func (tool Tool) Materialized() Tool {
	definition, known := tool.AnthropicDefined()
	if !known {
		return tool
	}
	materialized := tool
	materialized.Type = ""
	if materialized.Name == "" {
		materialized.Name = definition.Name
	}
	if materialized.Description == "" {
		materialized.Description = definition.Description
	}
	if len(materialized.InputSchema) == 0 {
		materialized.InputSchema = definition.InputSchema
		materialized.Strict = nil
	}
	return materialized
}

// AnthropicDefinedToolTypes lists the types this table defines, sorted, so a
// test can state the inventory it covers.
func AnthropicDefinedToolTypes() []string {
	return []string{
		"bash_20250124",
		"memory_20250818",
		"text_editor_20250124",
		"text_editor_20250429",
		"text_editor_20250728",
	}
}

var anthropicDefinedTools = map[string]AnthropicDefinedTool{
	"text_editor_20250124": {
		Name:        "str_replace_editor",
		Description: textEditorDescription + " The `undo_edit` command will revert the last edit made to the file at `path`.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{` +
			`"command":{"type":"string","enum":["view","create","str_replace","insert","undo_edit"],"description":"The command to run. Allowed options are: ` + "`view`, `create`, `str_replace`, `insert`, `undo_edit`" + `."},` +
			textEditorCommonProperties +
			`},"required":["command","path"]}`),
	},
	"text_editor_20250429": {
		Name:        "str_replace_based_edit_tool",
		Description: textEditorDescription,
		InputSchema: json.RawMessage(`{"type":"object","properties":{` +
			`"command":{"type":"string","enum":["view","create","str_replace","insert"],"description":"The command to run. Allowed options are: ` + "`view`, `create`, `str_replace`, `insert`" + `."},` +
			textEditorCommonProperties +
			`},"required":["command","path"]}`),
	},
	"text_editor_20250728": {
		Name:        "str_replace_based_edit_tool",
		Description: textEditorDescription,
		InputSchema: json.RawMessage(`{"type":"object","properties":{` +
			`"command":{"type":"string","enum":["view","create","str_replace","insert"],"description":"The command to run. Allowed options are: ` + "`view`, `create`, `str_replace`, `insert`" + `."},` +
			textEditorCommonProperties + `,` +
			`"max_characters":{"type":"integer","description":"Optional parameter of ` + "`view`" + ` command: the maximum number of characters to display when viewing a file."}` +
			`},"required":["command","path"]}`),
	},
	"bash_20250124": {
		Name: "bash",
		Description: "Run commands in a bash shell. When invoking this tool, the contents of the `command` parameter does NOT need to be XML-escaped. " +
			"State is persistent across command calls and discussions with the user. " +
			"To inspect a particular line range of a file, e.g. lines 10-20, try `sed -n 10,20p /path/to/the/file`. " +
			"Please avoid commands that may produce a very large amount of output. " +
			"Please run long lived commands in the background, e.g. `sleep 10 &` or start a server in the background.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{` +
			`"command":{"type":"string","description":"The bash command to run. Required unless the tool is being restarted."},` +
			`"restart":{"type":"boolean","description":"Specifying true will restart this tool. Otherwise, leave this unspecified."}` +
			`}}`),
	},
	"memory_20250818": {
		Name: "memory",
		Description: "A memory directory the caller persists across sessions, rooted at `/memories`. " +
			"`view` shows a directory listing or a file's contents, `create` writes a new file, " +
			"`str_replace` replaces one unique string in a file, `insert` adds text after a line, " +
			"`delete` removes a file or directory, and `rename` moves one.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{` +
			`"command":{"type":"string","enum":["view","create","str_replace","insert","delete","rename"],"description":"The command to run."},` +
			`"path":{"type":"string","description":"Absolute path under /memories, e.g. ` + "`/memories/notes.md`" + `. Used by every command but rename."},` +
			`"file_text":{"type":"string","description":"Required parameter of ` + "`create`" + ` command, with the content of the file to be created."},` +
			`"old_str":{"type":"string","description":"Required parameter of ` + "`str_replace`" + ` command containing the string in ` + "`path`" + ` to replace."},` +
			`"new_str":{"type":"string","description":"Required parameter of ` + "`str_replace`" + ` command containing the new string."},` +
			`"insert_line":{"type":"integer","description":"Required parameter of ` + "`insert`" + ` command. The ` + "`insert_text`" + ` will be inserted AFTER the line ` + "`insert_line`" + ` of ` + "`path`" + `."},` +
			`"insert_text":{"type":"string","description":"Required parameter of ` + "`insert`" + ` command containing the text to insert."},` +
			`"old_path":{"type":"string","description":"Required parameter of ` + "`rename`" + ` command: the current path."},` +
			`"new_path":{"type":"string","description":"Required parameter of ` + "`rename`" + ` command: the new path."},` +
			`"view_range":{"type":"array","items":{"type":"integer"},"description":"Optional parameter of ` + "`view`" + ` command when ` + "`path`" + ` points to a file: [start_line, end_line], indexed from 1."}` +
			`},"required":["command"]}`),
	},
}

// textEditorDescription is Anthropic's documented description of the text
// editor, without the undo_edit sentence the 20250429 revision removed.
const textEditorDescription = "Custom editing tool for viewing, creating and editing files. " +
	"State is persistent across command calls and discussions with the user. " +
	"If `path` is a file, `view` displays the file with line numbers. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep. " +
	"The `create` command cannot be used if the specified `path` already exists as a file. " +
	"If a `command` generates a long output, it will be truncated and marked with `<response clipped>`. " +
	"Notes for using the `str_replace` command: the `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespace. " +
	"If the `old_str` parameter is not unique in the file, the replacement will not be performed; include enough context in `old_str` to make it unique. " +
	"The `new_str` parameter should contain the edited lines that should replace the `old_str`."

// textEditorCommonProperties are the members every text editor revision
// shares, after the command member and before any revision-specific one.
const textEditorCommonProperties = `"path":{"type":"string","description":"Absolute path to file or directory, e.g. ` + "`/repo/file.py`" + ` or ` + "`/repo`" + `."},` +
	`"file_text":{"type":"string","description":"Required parameter of ` + "`create`" + ` command, with the content of the file to be created."},` +
	`"old_str":{"type":"string","description":"Required parameter of ` + "`str_replace`" + ` command containing the string in ` + "`path`" + ` to replace."},` +
	`"new_str":{"type":"string","description":"Optional parameter of ` + "`str_replace`" + ` command containing the new string (if not given, no string will be added). Required parameter of ` + "`insert`" + ` command containing the string to insert."},` +
	`"insert_line":{"type":"integer","description":"Required parameter of ` + "`insert`" + ` command. The ` + "`new_str`" + ` will be inserted AFTER the line ` + "`insert_line`" + ` of ` + "`path`" + `."},` +
	`"view_range":{"type":"array","items":{"type":"integer"},"description":"Optional parameter of ` + "`view`" + ` command when ` + "`path`" + ` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting ` + "`[start_line, -1]`" + ` shows all lines from ` + "`start_line`" + ` to the end of the file."}`
