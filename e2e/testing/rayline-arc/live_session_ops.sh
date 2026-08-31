#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Cost-bearing resource control for a live, interactive Rayline ARC session.
#
# This script owns exactly the three things that cost money outside Docker:
#   - the OpenRouter ephemeral key (hard server-side spend cap)
#   - the Modal encoder autoscaler floor (min_containers, ~$4.838/hr while >0)
#   - the Modal proxy token that authenticates the router to that encoder
#
# It is a thin, auditable wrapper over the same calls the frozen E2E launchers
# make; see docs/agent/runbook-claude-cli-arc-live-session.md for the procedure
# and e2e/testing/rayline-arc/run_openrouter_fullstack.py for the automated
# equivalent. Nothing here starts inference; nothing here starts compose.
#
# Usage:  live_session_ops.sh <command> [args]
# Run with no arguments for the command list.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
script_dir="${repo_root}/e2e/testing/rayline-arc"

# The repo's own venvs do not carry the Modal SDK. The launchers are run with
# the Pathfinder venv, which pins modal 1.5.1 — the version run_openrouter_
# fullstack.py:97 requires. Override with LIVE_SESSION_PYTHON if it moves.
PYTHON="${LIVE_SESSION_PYTHON:-${HOME}/code/pathfinder-rayline-vsr-mvp/.venv/bin/python}"

# Frozen encoder identity, from run_openrouter_fullstack.py:75-81. The app_id
# there (ap-XtsWCBEWdw1ncu9Kv12Chj) is NOT stable: as of 2026-08-10 that app is
# absent from the dev environment and a redeploy mints a new id. Everything
# here therefore matches on app NAME, which the image itself pins via
# RAYLINE_ARC_SESSION_APP_NAME (modal_session_service.py:175).
ENCODER_APP_NAME="${RAYLINE_ARC_SESSION_APP_NAME:-rayline-arc-session-encoder}"
ENCODER_CLASS_NAME="SessionEncoder"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-dev}"

# Compose identity for the live agentic stack.
COMPOSE_PROJECT="${RAYLINE_ARC_LIVE_PROJECT:-rayline-arc-live-claude}"
COMPOSE_FILE="${repo_root}/deploy/compose/rayline-arc/compose.yaml"
COMPOSE_OVERRIDE="${repo_root}/deploy/compose/rayline-arc/compose-openrouter-agentic.yaml"

# Colour escapes corrupt `modal ... --json` parsing; every launcher strips them.
modal_env=(env -u FORCE_COLOR -u COLORTERM TERM=dumb "MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}")

die() { echo "live_session_ops: $*" >&2; exit 1; }

require_python() {
  [ -x "${PYTHON}" ] || die "python interpreter not found: ${PYTHON} (set LIVE_SESSION_PYTHON)"
}

require_management_key() {
  [ -n "${OPENROUTER_MANAGEMENT_KEY:-}" ] || die "OPENROUTER_MANAGEMENT_KEY is required"
}

# ---------------------------------------------------------------- OpenRouter

# key-mint <run-id> <limit-usd>
# Prints two lines: the ephemeral key, then its hash. The HASH is what every
# later call needs; the key itself goes into RAYLINE_ARC_E2E_PROVIDER_KEY.
# The limit is enforced server-side by OpenRouter and is the ONLY hard spend
# cap on the provider once the artifact's completion-token cap is lifted.
cmd_key_mint() {
  local run_id="${1:?usage: key-mint <run-id> <limit-usd>}"
  local limit="${2:?usage: key-mint <run-id> <limit-usd>}"
  require_python; require_management_key
  "${PYTHON}" - "$run_id" "$limit" <<'PY'
import os, sys
sys.path.insert(0, os.environ["LIVE_SESSION_SCRIPT_DIR"])
from openrouter_key_management import create_ephemeral_key
key, key_hash = create_ephemeral_key(
    os.environ["OPENROUTER_MANAGEMENT_KEY"], sys.argv[1], float(sys.argv[2])
)
print(key)
print(key_hash)
PY
}

# key-usage <key-hash>  -> settled spend in USD, as a float
cmd_key_usage() {
  local key_hash="${1:?usage: key-usage <key-hash>}"
  require_python; require_management_key
  "${PYTHON}" - "$key_hash" <<'PY'
import os, sys
sys.path.insert(0, os.environ["LIVE_SESSION_SCRIPT_DIR"])
from openrouter_key_management import ephemeral_key_usage
print(f"{ephemeral_key_usage(os.environ['OPENROUTER_MANAGEMENT_KEY'], sys.argv[1]):.8f}")
PY
}

# key-delete <key-hash>
# Hard delete. There is no disable/expire path in the OpenRouter management
# API as this repo uses it — delete is the stop.
cmd_key_delete() {
  local key_hash="${1:?usage: key-delete <key-hash>}"
  require_python; require_management_key
  "${PYTHON}" - "$key_hash" <<'PY'
import os, sys
sys.path.insert(0, os.environ["LIVE_SESSION_SCRIPT_DIR"])
from openrouter_key_management import delete_ephemeral_key
delete_ephemeral_key(os.environ["OPENROUTER_MANAGEMENT_KEY"], sys.argv[1])
print("deleted")
PY
}

# ---------------------------------------------------------- Modal proxy token

# token-mint -> two lines: token_id, token_secret
cmd_token_mint() {
  require_python
  "${modal_env[@]}" "${PYTHON}" - <<'PY'
import modal
token = modal.Workspace.from_context().proxy_tokens.create()
print(token.token_id)
print(token.token_secret)
PY
}

# token-delete <token-id>
cmd_token_delete() {
  local token_id="${1:?usage: token-delete <token-id>}"
  require_python
  "${modal_env[@]}" "${PYTHON}" - "$token_id" <<'PY'
import sys, modal
modal.Workspace.from_context().proxy_tokens.delete(sys.argv[1])
print("deleted")
PY
}

# ------------------------------------------------------------- Modal encoder

_autoscale() {
  local min_containers="$1"
  require_python
  "${modal_env[@]}" \
    RAYLINE_ARC_APP_NAME="${ENCODER_APP_NAME}" \
    RAYLINE_ARC_CLASS_NAME="${ENCODER_CLASS_NAME}" \
    "${PYTHON}" - "$min_containers" <<'PY'
import os, sys, modal
instance = modal.Cls.from_name(
    os.environ["RAYLINE_ARC_APP_NAME"],
    os.environ["RAYLINE_ARC_CLASS_NAME"],
    environment_name=os.environ.get("MODAL_ENVIRONMENT", "dev"),
)()
instance.update_autoscaler(
    min_containers=int(sys.argv[1]),
    max_containers=1,
    buffer_containers=0,
    scaledown_window=300,
)
print(f"min_containers={sys.argv[1]}")
PY
}

# encoder-pin: hold one warm H100 so a think-gap over the 300s scaledown window
# does not pay a 78.9-96.9s cold start. BILLS CONTINUOUSLY at ~$4.838/hr.
cmd_encoder_pin() { _autoscale 1; }

# encoder-unpin: release the floor. This is the single most important teardown
# step — it is what stops the hourly burn.
cmd_encoder_unpin() { _autoscale 0; }

# encoder-url: resolve the deployed web URL (should match the frozen host).
cmd_encoder_url() {
  require_python
  "${modal_env[@]}" \
    RAYLINE_ARC_APP_NAME="${ENCODER_APP_NAME}" \
    RAYLINE_ARC_CLASS_NAME="${ENCODER_CLASS_NAME}" \
    "${PYTHON}" - <<'PY'
import os, modal
instance = modal.Cls.from_name(
    os.environ["RAYLINE_ARC_APP_NAME"],
    os.environ["RAYLINE_ARC_CLASS_NAME"],
    environment_name=os.environ.get("MODAL_ENVIRONMENT", "dev"),
)()
print(instance.web.get_web_url().rstrip("/"))
PY
}

# encoder-containers: list running container ids for the encoder app, matched
# by app name (see the app_id note above).
cmd_encoder_containers() {
  require_python
  "${modal_env[@]}" RAYLINE_ARC_APP_NAME="${ENCODER_APP_NAME}" "${PYTHON}" - <<'PY'
import json, os, subprocess, sys
name = os.environ["RAYLINE_ARC_APP_NAME"]
apps = json.loads(subprocess.run(
    [sys.executable, "-m", "modal", "app", "list", "--json"],
    capture_output=True, text=True, check=True,
).stdout)
app_ids = {a["app_id"] for a in apps if a.get("description") == name}
containers = json.loads(subprocess.run(
    [sys.executable, "-m", "modal", "container", "list", "--json"],
    capture_output=True, text=True, check=True,
).stdout)
for container in containers:
    if container.get("app_id") in app_ids or container.get("app_name") == name:
        print(container["container_id"])
PY
}

# encoder-app: show the deployed app row (id, state, tasks) for the encoder.
cmd_encoder_app() {
  require_python
  "${modal_env[@]}" RAYLINE_ARC_APP_NAME="${ENCODER_APP_NAME}" "${PYTHON}" - <<'PY'
import json, os, subprocess, sys
name = os.environ["RAYLINE_ARC_APP_NAME"]
apps = json.loads(subprocess.run(
    [sys.executable, "-m", "modal", "app", "list", "--json"],
    capture_output=True, text=True, check=True,
).stdout)
rows = [a for a in apps if a.get("description") == name]
if not rows:
    print(f"NOT DEPLOYED: {name}")
    raise SystemExit(1)
for a in rows:
    print(f"{a.get('app_id')} state={a.get('state')} tasks={a.get('tasks')}")
PY
}

# encoder-stop: stop every running container for the encoder app. Does NOT
# undeploy the app — `modal app stop` would remove the shared deployment that
# the frozen benchmarks depend on, so it is deliberately not done here.
cmd_encoder_stop() {
  local ids
  ids="$(cmd_encoder_containers)"
  if [ -z "${ids}" ]; then
    echo "no encoder containers running"
    return 0
  fi
  local id
  for id in ${ids}; do
    "${modal_env[@]}" "${PYTHON}" -m modal container stop "${id}" --yes || true
  done
  echo "requested stop for: ${ids}"
}

# encoder-health: prove the proxy token actually opens the encoder.
# Requires RAYLINE_ARC_E2E_MODAL_KEY / _SECRET and RAYLINE_ARC_E2E_ENCODER_BASE_URL.
cmd_encoder_health() {
  : "${RAYLINE_ARC_E2E_ENCODER_BASE_URL:?set RAYLINE_ARC_E2E_ENCODER_BASE_URL}"
  : "${RAYLINE_ARC_E2E_MODAL_KEY:?set RAYLINE_ARC_E2E_MODAL_KEY}"
  : "${RAYLINE_ARC_E2E_MODAL_SECRET:?set RAYLINE_ARC_E2E_MODAL_SECRET}"
  curl --fail --silent --show-error --max-time 30 \
    -H "Modal-Key: ${RAYLINE_ARC_E2E_MODAL_KEY}" \
    -H "Modal-Secret: ${RAYLINE_ARC_E2E_MODAL_SECRET}" \
    "${RAYLINE_ARC_E2E_ENCODER_BASE_URL%/}/health"
  echo
}

# ------------------------------------------------------------------ compose

cmd_compose_down() {
  docker compose --project-name "${COMPOSE_PROJECT}" \
    --file "${COMPOSE_FILE}" --file "${COMPOSE_OVERRIDE}" \
    down --volumes --remove-orphans
}

# ---------------------------------------------------------------- panic stop

# panic-stop [key-hash] [token-id]
# Halts ALL spend as fast as possible, in cost order: the encoder floor first
# (it is the expensive one), then provider access, then local containers.
# Every step is best-effort and independent — one failure never blocks the next.
cmd_panic_stop() {
  local key_hash="${1:-${OPENROUTER_EPHEMERAL_KEY_HASH:-}}"
  local token_id="${2:-${RAYLINE_ARC_E2E_MODAL_KEY:-}}"
  echo "== panic stop: releasing encoder autoscaler floor =="
  cmd_encoder_unpin || echo "!! unpin FAILED — check https://modal.com/apps immediately" >&2
  echo "== panic stop: stopping encoder containers =="
  cmd_encoder_stop || echo "!! container stop failed" >&2
  if [ -n "${key_hash}" ]; then
    echo "== panic stop: deleting OpenRouter ephemeral key =="
    cmd_key_delete "${key_hash}" || echo "!! key delete failed" >&2
  else
    echo "-- no key hash given; skipping OpenRouter key delete" >&2
  fi
  if [ -n "${token_id}" ]; then
    echo "== panic stop: deleting Modal proxy token =="
    cmd_token_delete "${token_id}" || echo "!! token delete failed" >&2
  else
    echo "-- no proxy token id given; skipping token delete" >&2
  fi
  echo "== panic stop: composing down =="
  cmd_compose_down || echo "!! compose down failed" >&2
  echo "== panic stop complete; now run: $0 verify-stopped =="
}

# verify-stopped: prove each cost source is actually off.
cmd_verify_stopped() {
  local dirty=0
  echo "-- modal containers for ${ENCODER_APP_NAME}:"
  local ids
  ids="$(cmd_encoder_containers || true)"
  if [ -n "${ids}" ]; then echo "   STILL RUNNING: ${ids}"; dirty=1; else echo "   none"; fi
  echo "-- docker compose services in ${COMPOSE_PROJECT}:"
  local services
  services="$(docker compose --project-name "${COMPOSE_PROJECT}" \
    --file "${COMPOSE_FILE}" --file "${COMPOSE_OVERRIDE}" ps --quiet 2>/dev/null || true)"
  if [ -n "${services}" ]; then echo "   STILL RUNNING"; dirty=1; else echo "   none"; fi
  echo "-- modal app state:"
  "${modal_env[@]}" "${PYTHON}" -m modal app list 2>/dev/null | grep -F "${ENCODER_APP_NAME}" || echo "   (not listed)"
  if [ "${dirty}" -ne 0 ]; then
    echo "VERIFY: NOT CLEAN" >&2
    return 1
  fi
  echo "VERIFY: clean"
}

usage() {
  cat <<EOF
live_session_ops.sh <command> [args]

OpenRouter (spend cap):
  key-mint <run-id> <limit-usd>   mint an ephemeral key; prints key then hash
  key-usage <key-hash>            settled spend, USD
  key-delete <key-hash>           hard delete the key

Modal proxy auth:
  token-mint                      prints token_id then token_secret
  token-delete <token-id>

Modal encoder (~\$4.838/hr while pinned):
  encoder-pin                     min_containers=1  (warm floor, BILLING)
  encoder-unpin                   min_containers=0  (stops the burn)
  encoder-url                     resolve deployed web URL
  encoder-containers              list running container ids
  encoder-stop                    stop running containers (does not undeploy)
  encoder-health                  authenticated GET /health through the proxy

Stack:
  compose-down                    docker compose down -v for the live project
  panic-stop [key-hash] [token-id]  halt all spend now, best effort
  verify-stopped                  prove everything is actually off
EOF
}

command="${1:-}"
shift || true
export LIVE_SESSION_SCRIPT_DIR="${script_dir}"
case "${command}" in
  key-mint)            cmd_key_mint "$@" ;;
  key-usage)           cmd_key_usage "$@" ;;
  key-delete)          cmd_key_delete "$@" ;;
  token-mint)          cmd_token_mint "$@" ;;
  token-delete)        cmd_token_delete "$@" ;;
  encoder-pin)         cmd_encoder_pin ;;
  encoder-unpin)       cmd_encoder_unpin ;;
  encoder-url)         cmd_encoder_url ;;
  encoder-containers)  cmd_encoder_containers ;;
  encoder-stop)        cmd_encoder_stop ;;
  encoder-health)      cmd_encoder_health ;;
  compose-down)        cmd_compose_down ;;
  panic-stop)          cmd_panic_stop "$@" ;;
  verify-stopped)      cmd_verify_stopped ;;
  ""|-h|--help|help)   usage ;;
  *)                   usage; die "unknown command: ${command}" ;;
esac
