"""Generate golden test data for the on-device parity check.

For a few queries, saves the tokenizer's token IDs and the PyTorch text
encoder's embedding to a JSON bundled into the Android app's assets. The
app runs the same token IDs through mobileclip_text.pte on the phone and
compares embeddings — proving the model runs correctly on the A34 before
any real tokenizer or UI exists on that side.

Run: python ml/export/make_golden.py
"""

import json
from pathlib import Path

import mobileclip
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"
OUT = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "golden_text.json"

QUERIES = [
    "a red dress",
    "man in black shirt",
    "a dog on a beach",
]

model, _, _ = mobileclip.create_model_and_transforms(
    "mobileclip_s1", pretrained=str(CHECKPOINT)
)
model.eval()
tokenizer = mobileclip.get_tokenizer("mobileclip_s1")

cases = []
with torch.inference_mode():
    for query in QUERIES:
        tokens = tokenizer([query])
        embedding = model.encode_text(tokens)
        cases.append(
            {
                "query": query,
                "tokens": tokens.squeeze(0).tolist(),
                "embedding": embedding.squeeze(0).tolist(),
            }
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"cases": cases}))
print(f"Wrote {len(cases)} golden cases to {OUT}")
