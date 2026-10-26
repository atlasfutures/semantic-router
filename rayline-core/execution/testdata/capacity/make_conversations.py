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
out = {}
for name, body in conversations.items():
    messages = copy.deepcopy(body["messages"])
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
    ids = tokenizer.apply_chat_template(messages, tools=body.get("tools"), add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    out[name] = {"body": {"model": "qwen", "max_tokens": 4096, **body}, "qwen_prompt_tokens": len(ids)}
    print(f"{name:32s} {len(ids)}")
with open(os.path.join(OUT, "conversations.json.gz"), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
    gz.write(json.dumps(out, ensure_ascii=False, indent=1).encode())
