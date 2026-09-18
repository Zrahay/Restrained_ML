"""
This file is concerned with indexing the data that we are going to get the embeddings
and then storing the embeddings inside the database.

Run: python ml/models/Indexer/indexer.py
"""

from pathlib import Path

import mobileclip
import torch
from PIL import Image
from tqdm import tqdm

dataset_dir = Path("Dataset/val_test2020/test")
checkpoint = "ml/models/mobileclip_s1.pt"
output_file = "ml/models/embeddings.pt"
batch = 32
model_name = "mobileclip_s1"

model, _, preprocess = mobileclip.create_model_and_transforms(
    model_name, pretrained=checkpoint
)
model.eval()

image_paths = sorted(dataset_dir.glob("*.jpg"))

all_embeddings = []
all_filenames = []

with torch.no_grad():
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
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images)
        features = model.encode_image(batch_tensor)
        features = torch.nn.functional.normalize(features, dim=-1)

        all_embeddings.append(features)
        all_filenames.extend(batch_names)

all_embeddings = torch.cat(all_embeddings, dim=0)
torch.save(
    {"embeddings": all_embeddings, "filenames": all_filenames},
    output_file,
)
