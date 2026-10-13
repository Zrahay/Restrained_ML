"""Export the CLIP BPE tokenizer's data files for the Kotlin port.

The Kotlin tokenizer reimplements the *algorithm* (byte-to-unicode mapping,
BPE merges, regex splitting); this script exports the *data* it needs:
the token->id vocabulary and the ordered merge list.

Run: python ml/export/export_tokenizer_assets.py
"""

import json
from pathlib import Path

import mobileclip

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "clip_tokenizer.json"

tok = mobileclip.get_tokenizer("mobileclip_s1")
inner = tok.tokenizer  # open_clip SimpleTokenizer

# bpe_ranks: {(a, b): rank} -> ordered list ["a b", ...] (index = rank)
merges = [None] * len(inner.bpe_ranks)
for (a, b), rank in inner.bpe_ranks.items():
    merges[rank] = f"{a} {b}"
assert all(m is not None for m in merges)

data = {
    "context_length": inner.context_length,
    "sot": inner.sot_token_id,
    "eot": inner.eot_token_id,
    "vocab": inner.encoder,
    "merges": merges,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data), encoding="utf-8")
print(f"Wrote vocab={len(inner.encoder)} merges={len(merges)} to {OUT}")
