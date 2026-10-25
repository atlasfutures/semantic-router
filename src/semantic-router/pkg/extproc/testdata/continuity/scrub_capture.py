#!/usr/bin/env python3
"""Turn one captured client request into a continuity fixture.

The fixture keeps the request's structure: every key, every array length,
every block type and role, the cache breakpoints, the thinking and thread
controls, and the pairing between tool calls and their results. It does not
keep content. It works from allowlists, so a protocol it has never seen
leaks nothing: a string is kept only under a STRUCTURAL key, a number only
under a NUMERIC key, an identifier under a LINKED_IDS key is replaced by a
random one (the same original id always gets the same replacement inside
one capture), a tool name is kept only if it is Claude Code's own, and every
other string and number becomes a placeholder. Only an allowlist of headers
is kept.

    scrub_capture.py <body.json> <meta.json | -> <out-dir>

<meta.json> is the capture proxy's record, whose `req_headers` hold the
request headers. Pass `-` for a body captured without its headers. The
script writes <out-dir>/client.json and <out-dir>/headers.json. case.json,
which says what the cell must do with the body, is written by hand.
"""

import json
import os
import re
import secrets
import sys

# Keys whose string values are protocol vocabulary, not content.
STRUCTURAL = {
    "type",
    "role",
    "media_type",
    "ttl",
    "scope",
    "effort",
    "display",
    "keep",
    "tool_choice",
}
# Keys whose number values are protocol limits, not content: token budgets,
# sampling controls, and the bounds of a tool's input schema.
NUMERIC = {
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "budget_tokens",
    "temperature",
    "top_p",
    "top_k",
    "maximum",
    "minimum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
}
# Keys whose string values are identifiers that link one part of the request
# to another, or to an earlier response. Each gets a stable random stand-in.
LINKED_IDS = {
    "id",
    "tool_use_id",
    "tool_call_id",
    "call_id",
    "approval_request_id",
    "previous_message_id",
}
# Block types whose `input` is what the model wrote for a tool call.
TOOL_CALLS = {"tool_use", "server_tool_use"}
ID_PREFIX = re.compile(r"^([a-z]+_)")
# Headers that shape how the router reads the body, kept verbatim: the client
# version, the beta set and the protocol version say nothing about a user.
# Everything else, credentials and per-session ids above all, is dropped.
KEPT_HEADERS = {
    "user-agent",
    "anthropic-beta",
    "anthropic-version",
    "content-type",
    "x-claude-code-request-class",
}
BILLING = re.compile(
    r"^x-anthropic-billing-header: cc_version=([0-9.]+?)(?:\.[0-9a-f]+)?; "
    r"cc_entrypoint=([a-z-]+);"
)
# Claude Code's own tools keep their names. Any other tool name, a
# connector's above all, says which tools an account has, so it is replaced.
BUILTIN_TOOLS = {
    "Agent",
    "Bash",
    "DeferredToolPlaceholder",
    "Edit",
    "Glob",
    "Grep",
    "ListAgents",
    "NotebookEdit",
    "Read",
    "ReportFindings",
    "ScheduleWakeup",
    "Skill",
    "TodoWrite",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Workflow",
    "Write",
}


class Scrubber:
    def __init__(self):
        self.ids = {}
        self.tool_names = {}
        self.texts = 0

    def linked_id(self, value):
        if value not in self.ids:
            match = ID_PREFIX.match(value)
            prefix = match.group(1) if match else ""
            self.ids[value] = prefix + secrets.token_hex(12)
        return self.ids[value]

    def tool_name(self, value):
        if value in BUILTIN_TOOLS:
            return value
        if value not in self.tool_names:
            self.tool_names[value] = "mcp__fixture__tool_%d" % (
                len(self.tool_names) + 1
            )
        return self.tool_names[value]

    def text(self, value):
        billing = BILLING.match(value)
        if billing:
            return (
                "x-anthropic-billing-header: cc_version=%s.000; "
                "cc_entrypoint=%s; cch=00000;" % billing.groups()
            )
        if not value.strip():
            return value
        self.texts += 1
        return "fixture text %d" % self.texts

    def free(self, value):
        # A tool call's input is whatever the model wrote: its keys, strings and
        # numbers are content, so nothing in it is kept but its shape.
        if isinstance(value, dict):
            return {
                "field_%d" % (index + 1): self.free(item)
                for index, item in enumerate(value.values())
            }
        if isinstance(value, list):
            return [self.free(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, bool) or value is None:
            return value
        # A number the model wrote can be an account or record id.
        return 0

    def value(self, value, key):
        if isinstance(value, dict):
            return {k: self.member(k, v, value) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, bool) or value is None:
            return value
        if not isinstance(value, str):
            return value if key in NUMERIC else 0
        if key in STRUCTURAL:
            return value
        if key in LINKED_IDS:
            return self.linked_id(value)
        if key == "model":
            # A fine-tuned model id names an organization and a project.
            return "fixture-model"
        if key in ("name", "tool_name"):
            return self.tool_name(value)
        if key == "user_id":
            return json.dumps(
                {
                    "device_id": secrets.token_hex(32),
                    "account_uuid": "",
                    "session_id": secrets.token_hex(16),
                },
                separators=(",", ":"),
            )
        if key == "signature":
            return "fixture-signature-" + secrets.token_hex(16)
        return self.text(value)

    def member(self, key, value, parent):
        if key == "input" and parent.get("type") in TOOL_CALLS:
            return self.free(value)
        # Metadata is the client's own map: its keys say nothing about the
        # protocol, so a structural-looking key in it keeps nothing.
        if key == "metadata" and isinstance(value, dict) and set(value) != {"user_id"}:
            return self.free(value)
        # A tool schema's property names, and the required list that repeats
        # them, are the tool's interface and are kept; its descriptions,
        # titles, defaults and enum values are content.
        if key == "properties" and isinstance(value, dict):
            return {k: self.value(v, k) for k, v in value.items()}
        if key == "required" and isinstance(value, list):
            return value
        # A schema's example values are free-form, whatever their keys say.
        if key in ("default", "examples", "example", "const", "enum"):
            return self.free(value)
        return self.value(value, key)


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    body_path, meta_path, out_dir = argv[1:]
    with open(body_path, encoding="utf-8") as handle:
        body = json.load(handle)
    headers = {}
    if meta_path != "-":
        with open(meta_path, encoding="utf-8") as handle:
            raw = json.load(handle).get("req_headers", {})
        headers = {k.lower(): v for k, v in raw.items() if k.lower() in KEPT_HEADERS}
    scrubbed = Scrubber().value(body, "")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "client.json"), "w", encoding="utf-8") as out:
        json.dump(scrubbed, out, separators=(",", ":"))
    with open(os.path.join(out_dir, "headers.json"), "w", encoding="utf-8") as out:
        json.dump(headers, out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main(sys.argv)
