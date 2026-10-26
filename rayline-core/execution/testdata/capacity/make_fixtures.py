"""Regenerate the capacity-estimator fixtures and their real Qwen3.8-27B token counts.

Run from the repository root with a locally cached Qwen3.8-27B tokenizer.json (no download):
    python3 rayline-core/execution/testdata/capacity/make_fixtures.py /path/to/tokenizer.json
Real-content fixtures are slices of files in this repository; dense fixtures are seeded and synthetic.
Counts are the tokenizer's encoding of the text alone (no chat template, no special tokens).
"""
import base64, glob, gzip, hashlib, json, os, random, sys

from tokenizers import Tokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
tok_path = sys.argv[1]
tok = Tokenizer.from_file(tok_path)
count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def cut_lines(text, limit):
    text = text[:limit]
    return text[: text.rfind("\n") + 1]


def cut_tokens(text, tokens):
    lines, out, lo, hi = text.splitlines(keepends=True), "", 0, None
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count("".join(lines[:mid])) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return "".join(lines[:lo])


rng = random.Random(20261026)
fixtures = {}
fixtures["prose"] = cut_lines("".join(open(p).read() for p in sorted(glob.glob("website/docs/**/*.md", recursive=True))), 120_000)
code = "".join(open(p).read() for p in sorted(glob.glob("rayline-core/execution/*.go")) if not p.endswith("_test.go"))
fixtures["code_30k"] = cut_tokens(code, 30_000)
fixtures["uvlock_120k"] = cut_lines(open("src/vllm-sr/uv.lock").read(), 120 * 1024)
fixtures["json_catalog"] = cut_lines(open("dashboard/backend/evaluationplane/metric_analysis_catalog.v1.json").read(), 60_000)
fixtures["numeric_csv"] = "".join(
    ",".join([str(rng.randint(0, 99999)), f"{rng.uniform(-1000, 1000):.4f}", str(rng.randint(0, 9)), f"{rng.random():.6f}"]) + "\n"
    for _ in range(2500))
fixtures["float_logs"] = "".join(
    f"2026-10-26T14:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}.{rng.randint(0,999999):06d}Z INFO step={i} "
    f"loss={rng.random()*4:.6f} lr={rng.random()*1e-3:.3e} grad_norm={rng.random()*10:.4f}\n" for i in range(1000))
fixtures["hex_sha256"] = "".join(
    f"{hashlib.sha256(str(rng.random()).encode()).hexdigest()}  shard-{i:05d}.safetensors\n" for i in range(800))
blob = base64.b64encode(bytes(rng.getrandbits(8) for _ in range(45_000))).decode()
fixtures["base64"] = "\n".join(blob[i:i + 76] for i in range(0, len(blob), 76)) + "\n"

fixtures["multilingual"] = "".join(rng.choice([
    "路由器根据上下文窗口的大小决定是否把请求交给云端模型。本地模型的上下文窗口较小，超过时必须在发送之前切换。\n",
    "ルーターはコンテキストウィンドウの大きさに基づいてリクエストをクラウドに渡します。\n",
    "Маршрутизатор передаёт запрос облачной модели, если локальное окно контекста слишком мало.\n",
    "Die Kontextfensterüberschreitungsprüfung verhindert stillschweigende Kürzungen durch lokale Laufzeitumgebungen.\n",
    "Build passed ✅ deploy 🚀 tests 🧪 flaky 🔥 package 📦 ok 👍🏽 → ✓ ★ ∑ ≤ ≥ ±\n"]) for _ in range(600))

counts = {}
for name, text in fixtures.items():
    with open(os.path.join(OUT, name + ".txt.gz"), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(text.encode())
    counts[name] = {"bytes": len(text.encode()), "qwen_tokens": count(text)}
manifest = {"tokenizer": "Qwen/Qwen3.8-27B tokenizer.json",
            "tokenizer_sha256": hashlib.sha256(open(tok_path, "rb").read()).hexdigest(), "fixtures": counts}
json.dump(manifest, open(os.path.join(OUT, "counts.json"), "w"), indent=1, sort_keys=True)
for name, c in counts.items():
    print(f"{name:12s} bytes={c['bytes']:7d} tokens={c['qwen_tokens']:6d} bytes/token={c['bytes']/c['qwen_tokens']:.2f}")
