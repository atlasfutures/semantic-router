FROM python:3.12-slim

WORKDIR /app
COPY artifact_fixture.py openrouter_artifact_fixture.py openrouter_agentic_artifact_fixture.py openrouter_kv_cache_artifact_fixture.py openrouter_kv_cache_matched_contract.py rayline_three_arm_budget.py mock_servers.py modal_fullstack_inputs.py /app/

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin nonroot

USER 65532:65532
