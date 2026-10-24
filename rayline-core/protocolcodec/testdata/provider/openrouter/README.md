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
| `chat-response-glm.json` | `z-ai/glm-5.2` | non-streamed | 2026-09-04T09:26:03Z | 13 / 9, cached 11, reasoning 0 |
| `chat-stream-glm.sse` | `z-ai/glm-5.2` | streamed | 2026-09-04T09:26Z | 13 / 3, cached 11, reasoning 0 |
| `chat-response-flash.json` | `deepseek/deepseek-v4-flash` | non-streamed | 2026-09-04T09:26Z | 11 / 2, every sub-count 0 |
| `chat-stream-flash.sse` | `deepseek/deepseek-v4-flash` | streamed | 2026-09-04T09:26Z | 11 / 3, every sub-count 0 |
| `chat-response-warm-prefix.json` | `deepseek/deepseek-v4-pro` | non-streamed | 2026-09-04T09:30Z | 1702 / 2, cached 1024 |
| `chat-stream-warm-prefix.sse` | `deepseek/deepseek-v4-pro` | streamed | 2026-09-04T09:30Z | 1702 / 2, cached 1024 |
| `chat-response-all-reasoning.json` | `z-ai/glm-5.2` | non-streamed | 2026-09-04T09:25Z | 19 / 64, reasoning 64, `finish_reason` length, empty content |
| `anthropic-response-warm-prefix.json` | `deepseek/deepseek-v4-pro` | non-streamed, **cell output** | 2026-09-04T09:29Z | 166 / 25, cache_read 1536, thinking 22 |
| `transport-error.json` | n/a | n/a | 2026-09-03 | none |

`anthropic-response-warm-prefix.json` is the only capture that is not a raw
OpenRouter body: it is what the cell itself emitted, so it exercises the
Anthropic-as-source decoder. It is a different turn from the raw warm-prefix
pair beside it -- 1536 cached, not 1024 -- so the two are never asserted as
one turn.

What these captures settle, all four confirmed on 2026-09-04:

- OpenRouter's `prompt_tokens` INCLUDES the cached part. The cell's
  `input_tokens` excludes it, which is why 166 + 1536 = 1702 on the cell
  capture.
- `cache_write_tokens` was 0 on every arm, so no capture exercises a
  non-zero `cache_creation_input_tokens`.
- Caching is per OpenRouter provider, not per model, so a warm prefix is not
  reproducible from the model id alone.
- `deepseek-v4-flash` serialises small numbers in exponent form
  (`...8e-7`). Both flash captures carry it, so the wire decoder is exercised
  against that spelling.
