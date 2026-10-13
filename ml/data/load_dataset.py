"""Download the Fashionpedia 5k subset and export its images for the indexer.

Run: python ml/data/load_dataset.py
"""

from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "ml" / "data" / "fashionpedia_5k"
IMAGES_DIR = REPO_ROOT / "ml" / "data" / "images"

ds = load_dataset("detection-datasets/fashionpedia", split="train")
subset = ds.shuffle(seed=42).select(range(5000))
subset.save_to_disk(str(DATASET_DIR))

# Export as plain JPGs: the indexer globs a folder, and retrieval results
# are easier to eyeball as files. Filename = row index in fashionpedia_5k,
# so embeddings can be joined back to the dataset's labels later.
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
for i, row in enumerate(tqdm(subset, desc="Exporting images")):
    row["image"].convert("RGB").save(IMAGES_DIR / f"{i:05d}.jpg", quality=95)

print(f"Saved dataset to {DATASET_DIR}")
print(f"Exported {len(subset)} images to {IMAGES_DIR}")
