#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
compose_file="${repo_root}/deploy/compose/rayline-arc/compose.yaml"
project_name="rayline-arc-e2e"
run_dir="$(mktemp -d)"
receipt="${run_dir}/restart-receipt.json"
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
  local url="$1"
  local deadline=$((SECONDS + 60))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --show-error --max-time 2 "${url}" >/dev/null 2>&1; then
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
    "rayline-arc-private-prompt-canary" \
    "rayline-arc-private-tool-canary" \
    "rayline-arc-private-episode-canary" \
    "rayline-arc-private-key-canary" \
    "${RAYLINE_ARC_E2E_REDIS_PASSWORD:-public-e2e-redis-secret}" \
    "${RAYLINE_ARC_E2E_PROVIDER_KEY:-public-e2e-provider-key}" \
    "public-e2e-modal-key" \
    "public-e2e-modal-secret"; do
    if rg --fixed-strings --quiet "${marker}" "${logs}"; then
      echo "Rayline ARC privacy scan failed: protected value reached service logs" >&2
      return 1
    fi
  done
}

compose down --volumes --remove-orphans >/dev/null 2>&1 || true
compose up --build --detach
wait_http "http://127.0.0.1:${RAYLINE_ARC_E2E_ENCODER_PORT:-18080}/health"
wait_http "http://127.0.0.1:${RAYLINE_ARC_E2E_ENCODER_B_PORT:-18083}/health"
wait_http "http://127.0.0.1:${RAYLINE_ARC_E2E_PROVIDER_PORT:-18081}/health"
wait_http "http://127.0.0.1:${RAYLINE_ARC_E2E_ROUTER_API_PORT:-18082}/health"

python3 "${repo_root}/e2e/testing/rayline-arc/test_stack.py" \
  --phase initial \
  --receipt "${receipt}"
python3 "${repo_root}/e2e/testing/rayline-arc/replica_contract.py"

compose restart router
wait_http "http://127.0.0.1:${RAYLINE_ARC_E2E_ROUTER_API_PORT:-18082}/health"
python3 "${repo_root}/e2e/testing/rayline-arc/test_stack.py" \
  --phase resume \
  --receipt "${receipt}"

compose stop redis
python3 "${repo_root}/e2e/testing/rayline-arc/test_stack.py" \
  --phase redis-loss \
  --receipt "${receipt}"

scan_logs
echo "Rayline ARC hermetic full-stack acceptance: PASS"
