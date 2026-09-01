# Programmable provider fixture for the protocol-conformance profile.
#
# The image carries the fixture binary and the versioned fixture tree, because a
# replay script's file references resolve on the fixture's own filesystem. The
# build context is the e2e module root so both fit in one copy.
#
#   docker build -f e2e/testing/conformance-fixture/Dockerfile e2e

FROM golang:1.25-alpine AS build

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/conformance-fixture ./cmd/conformance-fixture

FROM gcr.io/distroless/static-debian12:nonroot

LABEL maintainer="vLLM Semantic Router Team"
LABEL description="Programmable provider fixture that records and replays protocol-conformance cases"
LABEL version="0.1.0"

COPY --from=build /out/conformance-fixture /usr/local/bin/conformance-fixture
COPY --from=build /src/testcases/testdata/protocol-conformance /fixtures/protocol-conformance

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/conformance-fixture", "-addr", "0.0.0.0:8080"]
