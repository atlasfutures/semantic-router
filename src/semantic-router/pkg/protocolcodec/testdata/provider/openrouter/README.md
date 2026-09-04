# OpenRouter provider captures

Real upstream responses, recorded from the dev ARC cell against OpenRouter.
Generation ids are replaced with `gen-fixture-*` so a fixture names no live
route; everything else is byte-for-byte what the provider sent.

| file | arm | mode | captured | usage recorded |
|---|---|---|---|---|
| `chat-response.json` | `deepseek/deepseek-v4-pro` | non-streamed | 2026-09-03 | 11 / 8, cached 0, reasoning 0 |
| `chat-stream.sse` | `deepseek/deepseek-v4-pro` | streamed | 2026-09-03 | 11 / 68, cached 0, reasoning 58 |
| `chat-response-reasoning.json` | `xiaomi/mimo-v2.5-pro` | non-streamed | 2026-09-03 | 258 / 22, cached 192, reasoning 13 |
| `chat-stream-reasoning.sse` | `xiaomi/mimo-v2.5-pro` | streamed | 2026-09-03 | 258 / 22, cached 192, reasoning 13 |
| `chat-response-usage-split.json` | `qwen/qwen3.6-35b-a3b` | non-streamed | 2026-09-03 | 20 / 64, reasoning 79 — more reasoning than completion, so the split cannot be true |
| `chat-stream-length-reasoning.sse` | `qwen/qwen3.6-35b-a3b` | streamed | 2026-09-03T09:13:58Z, generation `gen-1788426838-JzW2NkUv3Nm1JinGevq1` | 17 / 128, cached 0, reasoning 102, `finish_reason` length |
| `transport-error.json` | n/a | n/a | 2026-09-03 | none |

Two basket arms have no capture yet. `provider_arm_usage_test.go` skips their
cases and names the file each one is waiting for:

- `z-ai/glm-5.2` — `chat-response-glm.json`, `chat-stream-glm.sse`
- `deepseek/deepseek-v4-flash` — `chat-response-flash.json`, `chat-stream-flash.sse`
