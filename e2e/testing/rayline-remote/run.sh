#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
compose_file="${repo_root}/deploy/compose/rayline-remote/compose.yaml"
project_name="rayline-remote-e2e"
run_dir="$(mktemp -d)"
logs="${run_dir}/compose.log"

compose() {
  docker compose \
    --project-name "${project_name}" \
    --file "${compose_file}" \
    "$@"
}

cleanup() {
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_http() {
  local target_url="$1"
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --show-error --max-time 2 "${target_url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

scan_logs() {
  compose logs --no-color >"${logs}" 2>&1
  local marker
  for marker in \
    "rayline-remote-private-prompt-canary" \
    "rayline-remote-private-tool-canary" \
    "rayline-remote-private-episode-canary" \
    "rayline-remote-private-client-key-canary" \
    "rayline-remote-private-receipt-canary" \
    "public-e2e-provider-key" \
    "public-e2e-rayline-key" \
    "public-e2e-hmac-key-32-bytes-long"; do
    if rg --fixed-strings --quiet "${marker}" "${logs}"; then
      echo "Rayline remote privacy scan failed: protected value reached service logs" >&2
      return 1
    fi
  done
}

compose down --volumes --remove-orphans >/dev/null 2>&1 || true
compose up --build --detach
wait_http "http://127.0.0.1:${RAYLINE_REMOTE_E2E_POLICY_PORT:-18090}/health"
wait_http "http://127.0.0.1:${RAYLINE_REMOTE_E2E_PROVIDER_A_PORT:-18091}/health"
wait_http "http://127.0.0.1:${RAYLINE_REMOTE_E2E_PROVIDER_B_PORT:-18092}/health"
wait_http "http://127.0.0.1:${RAYLINE_REMOTE_E2E_ROUTER_API_PORT:-18093}/health"

python3 "${repo_root}/e2e/testing/rayline-remote/test_stack.py"
scan_logs
