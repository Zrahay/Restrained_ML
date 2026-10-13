from pathlib import Path

import torch
import mobileclip

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
model_checkpoint = REPO_ROOT / "ml" / "models" / "mobileclip_s1.pt"

clip_model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s1",
    pretrained=str(model_checkpoint),
)


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    
    def forward(self, image):
        return self.clip_model.encode_image(image)


image_encoder = ImageEncoder(clip_model).eval()

sample_image = torch.randn(1, 3, 256, 256)

exported_program = torch.export.export(
    image_encoder,
    (sample_image,),
)

edge_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
)

et_program = edge_program.to_executorch()

with open("mobileclip_image.pte", "wb") as f:
    f.write(et_program.buffer)

print("Exported the Image Encoder")



class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, tokens):
        return self.clip_model.encode_text(tokens)


text_encoder = TextEncoder(clip_model).eval()

# torch.export traces tensor ops, so the example input must be the
# tokenizer's output ([1, 77] int64), not a raw string
tokenizer = mobileclip.get_tokenizer("mobileclip_s1")
sample_text = tokenizer(["Boy in red dress"])

exported_program = torch.export.export(
    text_encoder,
    (sample_text,),
)

edge_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
)

et_program = edge_program.to_executorch()

with open("mobileclip_text.pte", "wb") as f:
    f.write(et_program.buffer)


print("Exported the Text Encoder")