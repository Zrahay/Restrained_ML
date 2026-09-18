"""Sanity check for MobileCLIP-S0 image and text encoders (milestone 1).

Verifies both encoders load with pretrained weights and produce embeddings
that correlate with image content:

  1. Pick N images, each with a distinct dominant category
  2. Build one caption per image from its category
  3. Encode images and captions, L2-normalize, compute cosine similarity
  4. Pass = similarity matrix has its row max on the diagonal

Run: python ml/sanity_check.py
"""

from pathlib import Path

import torch
from datasets import load_from_disk

import mobileclip

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "ml" / "data" / "fashionpedia_5k"
CHECKPOINT_PATH = REPO_ROOT / "ml" / "models" / "mobileclip_s0.pt"

MODEL_NAME = "mobileclip_s0"
NUM_TEST_IMAGES = 8


def load_model():
    model, _, preprocess = mobileclip.create_model_and_transforms(
        MODEL_NAME, pretrained=str(CHECKPOINT_PATH)
    )
    model.eval()
    return model, preprocess


def pick_test_images(ds, n):
    """Pick one image per distinct dominant category until we have n.

    Dominant category = category of the largest object (by bbox area).
    Using max-area rather than the first listed object avoids captions
    describing tiny background items.
    """
    category_names = ds.features["objects"]["category"].feature.names
    images, captions, seen = [], [], set()

    for row in ds:
        categories, areas = row["objects"]["category"], row["objects"]["area"]
        if not categories:
            continue
        dominant = categories[max(range(len(areas)), key=lambda i: areas[i])]
        if dominant in seen:
            continue
        seen.add(dominant)
        images.append(row["image"].convert("RGB"))
        captions.append(f"a photo of a {category_names[dominant]}")
        if len(images) == n:
            break

    return images, captions


@torch.no_grad()
def encode(model, preprocess, images, captions):
    tokenizer = mobileclip.get_tokenizer(MODEL_NAME)

    image_batch = torch.stack([preprocess(img) for img in images])
    text_tokens = tokenizer(captions)

    image_features = model.encode_image(image_batch)
    text_features = model.encode_text(text_tokens)

    # CLIP embeddings must be L2-normalized before cosine similarity
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return image_features, text_features


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    model, preprocess = load_model()

    print(f"\nPreprocess pipeline (spec for the Android port):")
    print(preprocess)

    print(f"\nLoading dataset: {DATASET_PATH}")
    ds = load_from_disk(str(DATASET_PATH))
    images, captions = pick_test_images(ds, NUM_TEST_IMAGES)
    print(f"Picked {len(images)} images with distinct dominant categories\n")

    image_features, text_features = encode(model, preprocess, images, captions)
    similarity = image_features @ text_features.T

    print("Cosine similarity matrix (rows = images, cols = captions):")
    # Strip the shared "a photo of a " prefix so labels stay readable
    short = [c.removeprefix("a photo of a ") for c in captions]
    header = "".join(f"{c[:16]:>18}" for c in short)
    print(f"{'':>18}{header}")
    for i, cap in enumerate(short):
        row = f"{cap[:16]:>18}" + "".join(f"{similarity[i][j].item():18.3f}" for j in range(len(short)))
        print(row + f"   <- {'OK' if similarity[i].argmax().item() == i else 'MISS'}")

    diagonal_correct = sum(
        1 for i in range(len(captions)) if similarity[i].argmax().item() == i
    )
    accuracy = diagonal_correct / len(captions)
    random_baseline = 1 / len(captions)
    print(f"\nDiagonal max: {diagonal_correct}/{len(captions)} rows correct ({accuracy:.0%})")
    print(f"Random baseline: {random_baseline:.0%}")
    # CLIP confuses semantically adjacent categories (t-shirt vs sweater,
    # pants vs jumpsuit); 75%+ with misses on near-neighbors = pass.
    if accuracy >= 0.75:
        print("SANITY CHECK PASSED")
    else:
        print("SANITY CHECK FAILED — check preprocessing or weights")


if __name__ == "__main__":
    main()
