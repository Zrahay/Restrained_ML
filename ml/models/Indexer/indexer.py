"""
This file is concerned with indexing the data that we are going to get the embeddings
and then storing the embeddings inside the database.

Run: python ml/models/Indexer/indexer.py
"""

import sys
from pathlib import Path

import mobileclip
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
dataset_dir = REPO_ROOT / "ml" / "data" / "images"
checkpoint = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"
output_file = REPO_ROOT / "ml" / "models" / "embeddings.pt"
batch = 32
model_name = "mobileclip_s1"

model, _, preprocess = mobileclip.create_model_and_transforms(
    model_name, pretrained=str(checkpoint)
)
model.eval()

image_paths = sorted(dataset_dir.glob("*.jpg"))
if not image_paths:
    sys.exit(f"No .jpg files found in {dataset_dir} — run ml/data/load_dataset.py first")

failed = 0

all_embeddings = []
all_filenames = []

with torch.inference_mode(): # This enables us to run forward passes without PyTorch needing to build the computational graphs and that saves us a lot of memory and gives speeds as well
    for i in tqdm(range(0, len(image_paths), batch), desc="Indexing"):
        batch_paths = image_paths[i : i + batch]
        batch_images = []
        batch_names = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
                batch_names.append(path.name)
            except Exception as e:
                failed += 1
                print(f"Skipped {path.name}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images) # All PyTorch vision models expect this because they can not deal with 32 different images in different memory locations, they take in a 4D Input -> [Batch Size, RGB Channel, H, W, B] so the matrix multiplication can happen at just one memory location
        features = model.encode_image(batch_tensor) # This API just encodes all the images
        features = torch.nn.functional.normalize(features, dim=-1) # We normalize the embeddings to unit length so the retriever's dot product equals cosine similarity

        all_embeddings.append(features)
        all_filenames.extend(batch_names)

all_embeddings = torch.cat(all_embeddings, dim=0)
torch.save(
    {"embeddings": all_embeddings, "filenames": all_filenames},
    output_file,
)
print(f"Indexed {len(all_filenames)} / {len(image_paths)} images ({failed} failed)")
print(f"Saved embeddings to {output_file}")
