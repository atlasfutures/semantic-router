#!/usr/bin/env bash
# Diff every byte the router writes for the repo's captured request corpora
# under two Go toolchains (default: the old Go 1.25 encoding/json engine
# against the extproc image's Go 1.27, whose encoding/json runs on the v2
# engine). Covers the protocol codec goldens (request, response, stream,
# rejection, capability, error translation) and the full ext_proc request
# path: provider-bound bodies, refusals and policy decide requests.
#
# Usage: tools/scripts/json-v2-byte-diff.sh [old-toolchain] [new-toolchain]
# Needs the Rust binding libraries built (make rust) for the extproc tests.
set -euo pipefail
old=${1:-go1.25.3}
new=${2:-go1.27.2}
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$(mktemp -d)
for toolchain in "$old" "$new"; do
	(cd "$root/rayline-core" && CGO_ENABLED=0 GOTOOLCHAIN=$toolchain \
		PROTOCOL_GOLDEN_BYTES_OUT="$out/$toolchain" go test -count=1 ./protocolcodec/ >/dev/null)
	(cd "$root/src/semantic-router" && CGO_ENABLED=1 GOTOOLCHAIN=$toolchain \
		VSR_JSON_BYTE_DUMP_OUT="$out/$toolchain" go test -count=1 -run TestJSONEngineByteDump ./pkg/extproc/ >/dev/null)
done
files=$(find "$out/$old" -type f | wc -l | tr -d ' ')
if diff -r "$out/$old" "$out/$new"; then
	echo "identical: $files files under $old and $new ($out)"
else
	echo "DIFFERENT under $old and $new ($out)" >&2
	exit 1
fi
