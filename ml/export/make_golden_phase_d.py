"""Golden data for Phase D: Kotlin tokenizer + image preprocessing parity.

Produces three things in the Android app's assets:
  1. tokenizer_golden.json — 50 queries -> exact token IDs (tests the
     Kotlin BPE tokenizer in isolation)
  2. golden_images.json + the image files — image -> desktop embedding
     (tests Kotlin preprocessing + image encoder end to end)
  3. (golden_text.json from make_golden.py already covers text end-to-end)

Run: python ml/export/make_golden_phase_d.py
"""

import json
import shutil
from pathlib import Path

import mobileclip
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"
IMAGES_DIR = REPO_ROOT / "ml" / "data" / "images"
ASSETS = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"

QUERIES = [
    "a red dress",
    "man in black shirt",
    "woman wearing a blue denim jacket",
    "a dog on a beach",
    "sunset over the mountains",
    "two people shaking hands",
    "a photo of a t-shirt",
    "person holding an umbrella in the rain",
    "close-up of a wristwatch",
    "kids playing football",
    "a group photo at a wedding",
    "screenshot of a chat",
    "food on a white plate",
    "a cat sleeping on a sofa",
    "man riding a motorcycle",
    "birthday cake with candles",
    "a car parked on the street",
    "selfie in front of a mirror",
    "white sneakers",
    "a bouquet of red roses",
    "it's a beautiful day",
    "don't stop believing",
    "I'll be there",
    "we're going home",
    "you've got mail",
    "she'd rather walk",
    "THE QUICK BROWN FOX",
    "MiXeD CaSe QuErY",
    "   leading and trailing spaces   ",
    "multiple    spaces    between",
    "query with numbers 123 and 456",
    "iphone 15 pro max",
    "50% off sale sign",
    "price tag $19.99",
    "hashtag #vacation photo",
    "email@example.com on screen",
    "question? answer!",
    "semi;colon and comma, test",
    "quotes \"around\" words",
    "parentheses (like this)",
    "hyphen-ated long-form words",
    "slash/separated/words",
    "snow-capped peak at dawn",
    "neon signs at night",
    "a a a repeated words",
    "x",
    "supercalifragilisticexpialidocious",
    "zebra crossing pedestrian",
    "aerial view of a city",
    "black and white portrait of an old man",
]

TEST_IMAGES = ["00000.jpg", "00001.jpg", "00002.jpg"]

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s1", pretrained=str(CHECKPOINT)
)
model.eval()
tokenizer = mobileclip.get_tokenizer("mobileclip_s1")

# 1. Tokenizer golden: query -> exact 77 token IDs
cases = []
for q in QUERIES:
    ids = tokenizer([q]).squeeze(0).tolist()
    cases.append({"query": q, "tokens": ids})
(ASSETS / "tokenizer_golden.json").write_text(
    json.dumps({"cases": cases}), encoding="utf-8"
)
print(f"Wrote {len(cases)} tokenizer cases")

# 2. Image golden: file -> desktop embedding (and bundle the jpgs)
img_cases = []
with torch.inference_mode():
    for name in TEST_IMAGES:
        src = IMAGES_DIR / name
        shutil.copy(src, ASSETS / name)
        x = preprocess(Image.open(src).convert("RGB")).unsqueeze(0)
        emb = model.encode_image(x).squeeze(0).tolist()
        img_cases.append({"file": name, "embedding": emb})
(ASSETS / "golden_images.json").write_text(
    json.dumps({"cases": img_cases}), encoding="utf-8"
)
print(f"Wrote {len(img_cases)} image cases (+ copied jpgs to assets)")
