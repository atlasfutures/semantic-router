# Conversation continuity corpus

Real client requests, scrubbed, for `conversation_continuity_test.go`. Each
directory is one captured request:

- `client.json`: the body. Its structure is the capture's: keys, array
  lengths, roles, block types, cache breakpoints, thinking and thread
  controls, and the pairing between tool calls and results. Its content is
  not: text is a placeholder, ids are random, connector tool names are
  replaced.
- `headers.json`: the request headers that shape how the router reads the
  body (user agent, `anthropic-beta`, `anthropic-version`). Credentials and
  session ids are not kept.
- `case.json`: what the cell must do with it. `expect` is `served` (the
  provider reads the whole conversation, changed only by the named
  `transforms`, and every other top-level member as sent) or `refused` (the
  orphan tool-result 400, before any policy decision, with no provider
  body; `refusal_body` names the exact bytes). `dropped_fields` lists
  top-level members the cell does not forward. `body` names the body file
  instead of `client.json` when another test already keeps the capture, as
  `../thread_continuation` does, so a capture is stored once.

Add a capture with:

```bash
python3 scrub_capture.py <body.json> <proxy-meta.json | -> <new-case-dir>
```

then write its `case.json` by hand. Read the scrubbed `client.json` before
committing it: no user text, no account or session ids.
