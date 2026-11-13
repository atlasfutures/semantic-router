"""Regenerate conversations.json: OpenAI Chat bodies and the real prompt token counts of the Qwen3.8-27B chat
template (transformers apply_chat_template, add_generation_prompt=True, template defaults), from a local snapshot:
    python3 rayline-core/execution/testdata/capacity/make_conversations.py /path/to/Qwen3.8-27B/snapshot
The texts are slices of the fixtures beside this script."""
import copy, gzip, json, os, sys

from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
text = lambda name, limit: gzip.open(os.path.join(OUT, name + ".txt.gz")).read().decode()[:limit]

tools = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file — returns UTF-8 text (≤ 1 MB).",
                                      "parameters": {"type": "object", "properties": {
                                          "path": {"type": "string", "description": "Absolute path"},
                                          "limit": {"type": "integer", "minimum": 1, "default": 2000},
                                          "follow": {"type": "boolean", "default": False},
                                          "ratio": {"type": "number", "default": 0.5}},
                                          "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.\nQuote \"paths\" with spaces.",
                                      "parameters": {"type": "object", "properties": {
                                          "command": {"type": "string"}, "env": {"type": "object"},
                                          "args": {"type": "array", "items": {"type": "string"}}},
                                          "required": ["command"]}}},
]
call = lambda id_, name, args: {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
conversations = {
    "user_only": {"messages": [{"role": "user", "content": "  Summarise the README.  "}]},
    "system_user": {"messages": [{"role": "system", "content": text("prose", 4000)},
                                 {"role": "user", "content": "What changed?\n\n"}]},
    "tool_loop": {"tools": tools, "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": None, "tool_calls": [
            call("c1", "read_file", {"path": "/repo/a.go", "limit": 400, "follow": True, "ratio": 0.25}),
            call("c2", "bash", {"command": "go test ./...\ncd /tmp && ls", "env": {"A": "1", "Ü": "ß"}, "args": ["-v", "x y"]})]},
        {"role": "tool", "tool_call_id": "c1", "content": text("code_30k", 6000)},
        {"role": "tool", "tool_call_id": "c2", "content": "ok  \t./...\t0.4s\n"},
        {"role": "assistant", "content": "The test passes now."},
        {"role": "user", "content": "Thanks — also check the lock file."}]},
    "reasoning_and_content_with_call": {"tools": tools[:1], "messages": [
        {"role": "user", "content": "Look at it."},
        {"role": "assistant", "reasoning_content": "  I should read the file first.  ", "content": "Reading it now.",
         "tool_calls": [call("c1", "read_file", {"path": "/r/uv.lock"})]},
        {"role": "tool", "tool_call_id": "c1", "content": text("uvlock_120k", 20000)},
        {"role": "user", "content": "And?"}]},
    "long_mixed": {"messages": [
        {"role": "user", "content": text("probe_hebrew_cant", 20000)},
        {"role": "assistant", "content": text("multilingual", 8000)},
        {"role": "user", "content": text("edge_cases", 20000)},
        {"role": "assistant", "content": text("numeric_csv", 20000)},
        {"role": "user", "content": text("probe_short_rand_lower_vowelrich", 10000)}]},
    "edge_system_with_tools": {"tools": tools, "messages": [
        {"role": "system", "content": text("edge_cases", 6000)},
        {"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": text("emoji", 3000)}]}]},
}
# vLLM's Chat path, applied before the template: tools are model_dump()ed (declared fields first, absent ones
# null), arguments are json.loads()ed (so 1e15 is a float), and text parts are joined with "\n".
vllm_conversations = {
    "vllm_bare_tool": {"tools": [{"type": "function", "function": {"name": "x"}}],
                       "messages": [{"role": "user", "content": "call x"}]},
    "vllm_floats": {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "number", "default": 1e15, "maximum": 1.5e300, "minimum": -1e-7}}}}}],
        "messages": [{"role": "user", "content": "go"},
                     {"role": "assistant", "content": None, "tool_calls": [call("c1", "f", {"n": 1e15, "xs": [1e20, 2.5e-5, 3, 1e16]})]},
                     {"role": "tool", "tool_call_id": "c1", "content": "done"}, {"role": "user", "content": "again"}]},
    "vllm_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"},
                                                             {"type": "text", "text": " third "}]}]},
}


def vllm_shape(body):
    body = copy.deepcopy(body)
    for tool in body.get("tools") or []:
        function = tool["function"]
        tool["function"] = {"name": function["name"], "description": function.get("description"),
                            "parameters": function.get("parameters"),
                            **{k: v for k, v in function.items() if k not in ("name", "description", "parameters")}}
    for message in body["messages"]:
        if isinstance(message.get("content"), list):
            message["content"] = "\n".join(part["text"] for part in message["content"])
    return body


def bodies_with_float_literals(body):
    # The request carries the floats as JSON numbers; spell them as a client would (shortest repr).
    return json.loads(json.dumps(body))


out = {}
for name, body in list(conversations.items()) + [(n, None) for n in vllm_conversations]:
    vllm = body is None
    body = vllm_conversations[name] if vllm else body
    shaped = vllm_shape(body) if vllm else body
    messages = copy.deepcopy(shaped["messages"])
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
    ids = tokenizer.apply_chat_template(messages, tools=shaped.get("tools"), add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    out[name] = {"body": {"model": "qwen", "max_tokens": 4096, **body}, "qwen_prompt_tokens": len(ids), "vllm": vllm}
    print(f"{name:32s} {len(ids)}")

# A tool whose schema has a 50,000-entry enum, and the tokens of its tojson(indent=4) spelling: a template may
# indent, so the byte bound of any other model must cover it.
enum_tool = {"type": "function", "function": {"name": "pick", "parameters": {"type": "object", "properties": {
    "id": {"type": "string", "enum": ["v%05d" % i for i in range(50000)]}}}}}
enum_tokens = len(tokenizer(json.dumps(enum_tool, indent=4, ensure_ascii=False), add_special_tokens=False)["input_ids"])
out["enum_50k_indent4"] = {"body": {"model": "qwen", "max_tokens": 4096, "tools": [enum_tool],
                                    "messages": [{"role": "user", "content": "pick one"}]},
                           "qwen_prompt_tokens": enum_tokens, "vllm": False, "tool_json_only": True}
print("enum_50k_indent4", enum_tokens)

with open(os.path.join(OUT, "conversations.json.gz"), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
    gz.write(json.dumps(out, ensure_ascii=False, indent=1).encode())
