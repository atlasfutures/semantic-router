# Wire-event replay fixtures

Real harness request bodies used to test the core's wire event inference.

- `claude_code_*.json`, `codex_*.json`: copied unchanged from VSR
  `src/semantic-router/pkg/selection/raylinearc/testdata/wire_signals/`; each
  file's `provenance` states its capture and redaction. The 2.1.309 files are
  one session with deterministic placeholders, so equal strings stay equal.
- `pi_0.80.2_overflow_*.jsonl`: stock Pi 0.80.2 (no extension), captured
  2026-10-24 against a local logging stub that returned a tool call, then a
  provider context overflow ("prompt is too long"), forcing Pi's own
  compaction. One request per line, in order: `session_affinity` records
  whether Pi sent `x-session-affinity` (its summarization call does not);
  message strings over 160 characters are shortened deterministically to a prefix
  and a digest (equal strings stay equal); system blocks keep their first 400
  characters; tool definitions are dropped. Content is
  synthetic; no credentials or provider responses are included. The
  `retained_tail` run used `compaction.keepRecentTokens: 4000` and a second
  prompt; `no_tail` used `keepRecentTokens: 2000` with large tool results.
