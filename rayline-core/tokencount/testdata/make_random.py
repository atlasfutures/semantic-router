"""Regenerate random.json.gz: seeded random strings over the character pools of the #357 review's comparison
(ASCII, whitespace, marks, contractions, CJK, Unicode 16 scripts, NFC-changing text, digits, symbols, Cyrillic,
Hebrew, Arabic, added tokens, Latin), long runs, and Unicode 16 composition pairs, with the reference
Qwen3.8-27B token count of each (cached tokenizer.json; no download):
    python3 rayline-core/tokencount/testdata/make_random.py /path/to/tokenizer.json"""
import gzip, json, os, random, sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
random.seed(20261026)
pools = {
    'ascii': [chr(c) for c in range(32, 127)] + ['\n', '\r', '\t', '\r\n'],
    'ws': [' ', '\t', '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\x1f', '\x85', '\xa0', ' ', ' ',
           ' ', ' ', ' ', ' ', '　', '᠎', '​'],
    'marks': ['́', '̈', 'ְ', 'ּ', 'َ', 'ّ', 'ा', 'ั', '⃝', '️', '‍', 'ͅ'],
    'contr': ["'s", "'S", "'t", "'re", "'RE", "'ve", "'m", "'ll", "'LL", "'d", "'ſ", "'K", "’s"],
    'cjk': [chr(c) for c in range(0x4e00, 0x4e00 + 300)] + [chr(c) for c in range(0x3040, 0x30ff)] + [chr(c) for c in range(0xac00, 0xac00 + 200)],
    'rare': [chr(c) for c in list(range(0x10D40, 0x10D90)) + list(range(0x11380, 0x113e0)) + list(range(0x1E5D0, 0x1E600)) +
             list(range(0x16D40, 0x16D7A)) + list(range(0x1CC00, 0x1CCF0)) + list(range(0x31350, 0x31360)) +
             list(range(0x2EBF0, 0x2EC00)) + list(range(0x16100, 0x1613A)) + list(range(0x105C0, 0x105F4)) + [0x10041, 0x10413, 0x07FD]],
    'nfc': ['é', 'Å', '가', '각', 'Å', 'Ω', '豈', 'क़', 'ȫ', 'क़'],
    'num': [str(i) for i in range(10)] + ['١', '१', '⅕', 'Ⅷ', '１', '²', '〇', '①'],
    'sym': list('!@#$%^&*()[]{}<>?/\\|`~-_=+;:,."') + ['—', '©', '\U0001F600', '\U0001F468‍\U0001F469', '€', '­', '﻿', '⁠', '\x00', '\x07', '\x7f'],
    'cyr': [chr(c) for c in range(0x400, 0x530)], 'heb': [chr(c) for c in range(0x591, 0x5f5)], 'ara': [chr(c) for c in range(0x600, 0x700)],
    'added': ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '<|endoftext|>', '<|im_start|', '<thin', '< think>',
              '<<|im_end|>', '<|vision_start|><|image_pad|>', '<tts_text_bos_single>', '<tts_text_bos>'],
    'latin': [chr(c) for c in range(0x41, 0x5b)] + [chr(c) for c in range(0x61, 0x7b)] + [chr(c) for c in range(0xc0, 0x250)],
}
pools['rare'] = [c if isinstance(c, str) else chr(c) for c in pools['rare']]
names = list(pools)
texts = []
for _ in range(1500):
    n = random.choice([1, 2, 3, 5, 10, 40, 200, 1000])
    chosen = random.sample(names, random.choice([1, 2, 3, 4]))
    texts.append(''.join(random.choice(pools[random.choice(chosen)]) for _ in range(n)))
texts += [' ' * n for n in (1, 2, 3, 17, 100)] + ['\n' * n for n in (1, 2, 3, 100)] + ['a' * n for n in (1, 64, 65, 3000)]
texts += ['ab ' * 1000, '\t \n \r\n  x' * 50, '0' * 3000, 'ſ' * 20, "it'ſ fine", "Don'T", "'ll'll'll"]
texts += ['\U00011390\U000113C9' * 2000, '\U0001611E\U0001611E\U0001611F' * 500, '\U00016D63\U00016D67\U00016D67' * 500,
          '\U000105D2̇' * 1000, '\U00011935\U00011930' * 1000, 'a\U00010041́' * 1000]
counts = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random.json.gz')
with open(out, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
    gz.write(json.dumps([{"text": t, "qwen_tokens": c} for t, c in zip(texts, counts)], ensure_ascii=False).encode())
print(len(texts), os.path.getsize(out))
