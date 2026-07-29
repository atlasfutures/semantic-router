FROM python:3.12-slim

WORKDIR /app
COPY artifact_fixture.py mock_servers.py /app/

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin nonroot

USER 65532:65532
