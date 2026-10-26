"""Regenerate the capacity-estimator fixtures and their real Qwen3.8-27B token counts.

Run from the repository root with a locally cached Qwen3.8-27B tokenizer.json (no download):
    python3 rayline-core/execution/testdata/capacity/make_fixtures.py /path/to/tokenizer.json
CAPACITY_EXTRA_TEXTS=<dir> adds every <dir>/*.txt as probe_<name> (the #357 review's adversarial probe texts).
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

# Adversarial content (review of #357): long single-case letter runs, non-Latin
# alphabets, rare Han, random Hangul, full-width forms and emoji.
fixtures["fasta"] = ">chr7:55019017-55211628 synthetic\n" + "".join(
    "".join(rng.choice("ACGT") for _ in range(60)) + "\n" for _ in range(1000))
fixtures["slugs_lower"] = "".join(
    "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(12, 40))) + rng.choice(" \n-_/")
    for _ in range(1500))
fixtures["upper_codes"] = "".join(
    "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(4, 24))) + rng.choice(" \n_,.")
    for _ in range(2500))


def alphabet_words(letters, words):
    return "".join("".join(rng.choice(letters) for _ in range(rng.randint(2, 11))) + rng.choice("  ,.\n")
                   for _ in range(words))


fixtures["greek"] = alphabet_words([chr(c) for c in range(0x3B1, 0x3CA)], 4000)
fixtures["hebrew"] = alphabet_words([chr(c) for c in range(0x5D0, 0x5EB)], 4000)
fixtures["armenian"] = alphabet_words([chr(c) for c in range(0x561, 0x587)], 4000)
fixtures["cyrillic"] = alphabet_words([chr(c) for c in range(0x430, 0x450)], 4000)
fixtures["arabic"] = alphabet_words([chr(c) for c in range(0x621, 0x64B)], 4000)
fixtures["latin_ext"] = alphabet_words([chr(c) for c in range(0x100, 0x180)], 4000)
fixtures["rare_han"] = "".join(chr(rng.choice([rng.randint(0x3400, 0x4DBF), rng.randint(0x8000, 0x9FA5)])) +
                               ("\n" if rng.random() < 0.02 else "") for _ in range(8000))
fixtures["hangul_random"] = "".join(chr(rng.randint(0xAC00, 0xD7A3)) + (" " if rng.random() < 0.2 else "")
                                    for _ in range(8000))
fixtures["fullwidth"] = "".join(chr(rng.randint(0xFF01, 0xFF5E)) + (" " if rng.random() < 0.1 else "")
                                for _ in range(8000))
fixtures["emoji"] = "".join(rng.choice(["🚀", "🔥", "🧪", "📦", "👍🏽", "👨‍👩‍👧", "🇯🇵", "✅", "⚠️", "🫠", "🪿"]) +
                            (" " if rng.random() < 0.3 else "") for _ in range(4000))

# Tokenizer edge cases: added tokens in text, contractions in any case,
# whitespace and CRLF runs, text that NFC changes, combining marks, non-ASCII
# digits and line separators.
edge = [
    "Don't you'LL see it's <|im_start|>user\n<think>x</think><tool_call>{}</tool_call>",
    "  leading   spaces\t\ttabs \r\n\r\n  \n\n\n   trailing   ",
    "cafe\u0301 re\u0301sume\u0301 A\u030a e\u0308\u0301 \u1e9b\u0323 ",
    "\u0661\u0662\u0663 \u0967\u0968 \u2160\u2161 \u00bd \u2028line\u2029para\u0085next\u00a0nb",
    "...hello!!! (x) [y] {z} --flag=value a\u0300\u0301\u0302 \u05d0\u05b0\u0591 'S 'T 'Re 'VE 'M 'Ll 'D",
    "\n\n\t\n  x\n  \n", "    ", "\n", "x ", " ", "a\r\nb\r\n\r\nc",
]
fixtures["edge_cases"] = "".join(rng.choice(edge) for _ in range(3000))
extra = os.environ.get("CAPACITY_EXTRA_TEXTS")
if extra:
    for path in sorted(glob.glob(os.path.join(extra, "*.txt"))):
        fixtures["probe_" + os.path.basename(path)[:-4]] = open(path, encoding="utf-8").read()

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
