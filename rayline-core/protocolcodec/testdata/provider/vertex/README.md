# Vertex AI OpenAI-compatible provider shapes

Chat Completions responses from partner models served through Vertex AI's
OpenAI-compatible endpoint. Each file reproduces the usage and metadata shape
the provider sent. The ids, timestamps, content and counts are synthetic.

| file | model family | mode | shape |
|---|---|---|---|
| `chat-response-grok.json` | xAI Grok | non-streamed | `reasoning_tokens` counted beside `completion_tokens`: total = prompt + completion + reasoning (182 + 8 + 320 = 510), cached 128; extra `usage` members `num_sources_used`, `cost_in_usd_ticks`, `extra_properties` |
| `chat-stream-grok.sse` | xAI Grok | streamed | the same accounting on the final usage chunk (182 + 3 + 216 = 401) |
| `chat-response-deepseek-flash.json` | DeepSeek | non-streamed | `metadata.weight_versions` is an array; `choices[].matched_stop`; `usage.extra_properties` |
