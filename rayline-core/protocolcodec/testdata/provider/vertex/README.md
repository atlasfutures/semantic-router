# Vertex AI OpenAI-compatible provider captures

Real responses from partner models served through Vertex AI's
OpenAI-compatible Chat Completions endpoint, recorded 2026-11-24 (global
location, HTTP 200, streamed with `stream_options.include_usage`). Response
ids are replaced with `fixture-*`; everything else is byte-for-byte what the
provider sent.

| file | model | mode | usage recorded |
|---|---|---|---|
| `chat-response-grok.json` | `xai/grok-4.7` | non-streamed | 220 / 16, reasoning 224 counted beside completion: total 460 = 220 + 16 + 224, cached 0 |
| `chat-stream-grok.sse` | `xai/grok-4.7` | streamed | 220 / 16, reasoning 172 beside completion: total 408, cached 128 |
| `chat-response-deepseek-flash.json` | `deepseek-ai/deepseek-v4.1-flash-maas` | non-streamed | 18 / 17; `metadata.weight_versions` is an array of objects; `prompt_tokens_details` null |
| `chat-stream-deepseek-flash.sse` | `deepseek-ai/deepseek-v4.1-flash-maas` | streamed | 18 / 14 |

Both models also send `usage.extra_properties`; Grok sends
`usage.cost_in_usd_ticks` and `usage.num_sources_used`. These have no
protocol-neutral bucket and are dropped with a log line.
