FROM python:3.12-slim

WORKDIR /app
COPY mock_services.py /app/

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin nonroot

USER 65532:65532
