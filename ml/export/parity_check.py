"""Parity check: exported .pte encoders vs the original PyTorch model.

Runs the same inputs through both and compares embeddings. The export is a
translation of the model into ExecuTorch's format, and translations can be
subtly lossy without ever crashing — this is the gate that catches that
before any Android work builds on the .pte files.

Pass = cosine similarity >= 0.999 for every tested image and query.

Run: python ml/export/parity_check.py
"""

from pathlib import Path

import mobileclip
import torch
from executorch.runtime import Runtime
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"
IMAGE_PTE = REPO_ROOT / "ml" / "export" / "mobileclip_image.pte"
TEXT_PTE = REPO_ROOT / "ml" / "export" / "mobileclip_text.pte"
IMAGES_DIR = REPO_ROOT / "ml" / "data" / "images"

NUM_TEST_IMAGES = 5
TEST_QUERIES = [
    "a red dress",
    "man in black shirt",
    "a photo of a jacket",
    "blue jeans",
    "woman wearing a hat",
]
COSINE_THRESHOLD = 0.999


def load_pte_method(path):
    program = Runtime.get().load_program(str(path))
    return program.load_method("forward")


def cosine(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).item()


def main():
    print(f"Loading PyTorch model: {CHECKPOINT}")
    model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s1", pretrained=str(CHECKPOINT)
    )
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s1")

    print(f"Loading exported programs: {IMAGE_PTE.name}, {TEXT_PTE.name}")
    image_method = load_pte_method(IMAGE_PTE)
    text_method = load_pte_method(TEXT_PTE)

    failures = 0

    print(f"\n--- Image encoder parity ({NUM_TEST_IMAGES} real images) ---")
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))[:NUM_TEST_IMAGES]
    with torch.inference_mode():
        for path in image_paths:
            x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            ref = model.encode_image(x)
            out = image_method.execute([x])[0]
            cos = cosine(ref, out)
            max_diff = (ref - out).abs().max().item()
            ok = cos >= COSINE_THRESHOLD
            failures += not ok
            print(f"{path.name}: cosine={cos:.6f}  max|diff|={max_diff:.2e}  {'OK' if ok else 'FAIL'}")

    print(f"\n--- Text encoder parity ({len(TEST_QUERIES)} queries) ---")
    with torch.inference_mode():
        for query in TEST_QUERIES:
            tokens = tokenizer([query])
            ref = model.encode_text(tokens)
            out = text_method.execute([tokens])[0]
            cos = cosine(ref, out)
            max_diff = (ref - out).abs().max().item()
            ok = cos >= COSINE_THRESHOLD
            failures += not ok
            print(f"{query!r}: cosine={cos:.6f}  max|diff|={max_diff:.2e}  {'OK' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"PARITY CHECK FAILED: {failures} case(s) below cosine {COSINE_THRESHOLD}")
        raise SystemExit(1)
    print("PARITY CHECK PASSED — exported encoders match PyTorch")


if __name__ == "__main__":
    main()
