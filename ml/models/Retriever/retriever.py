"""
This file is concerned with retrieving the most similar images to a given text query.

Run: python ml/models/Retriever/retriever.py --query "a red dress"
"""

import argparse
from pathlib import Path

import mobileclip
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
embeddings_file = REPO_ROOT / "ml" / "models" / "embeddings.pt"
checkpoint = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"
model_name = "mobileclip_s1"
top_k = 5

model, _, preprocess = mobileclip.create_model_and_transforms(
    model_name, pretrained=str(checkpoint)
)
model.eval()
tokenizer = mobileclip.get_tokenizer(model_name)

data = torch.load(embeddings_file, weights_only=True)
image_embeddings = data["embeddings"]
filenames = data["filenames"]


def retrieve(query: str, k: int = top_k):
    k = min(k, len(filenames))  # topk crashes if k exceeds the index size
    with torch.inference_mode():
        tokens = tokenizer([query])
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        similarities = text_features @ image_embeddings.T
        scores, indices = similarities.squeeze(0).topk(k)

    results = [(filenames[idx], score.item()) for idx, score in zip(indices, scores)]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=top_k)
    args = parser.parse_args()

    results = retrieve(args.query, args.k)
    for filename, score in results:
        print(f"{score:.4f}  {filename}")
