"""Bereitstellung von Funktionen zur Erzeugung von Bild-Embeddings"""

from typing import List
from itertools import batched
import gc
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import default_collate
import torch
from transformers import AutoImageProcessor, BatchFeature, BaseImageProcessor
from transformers.image_utils import ImageInput


def check_cuda_memory() -> None:
    """Listet die Größe aller Tensoren auf cuda-device"""
    for obj in filter(lambda o: isinstance(o, torch.Tensor), gc.get_objects()):
        if obj.device.type == "cuda":
            print(type(obj), obj.size(), obj.device)


def create_image_batches(
    images: ImageInput, model_name: str, batch_size: int
) -> List[BatchFeature]:
    """Erzeuge batches der Bilder, die als Input für die Embedding-Erzeugung dienen können.
    Es wird AutoImageProcessor von HuggingFace verwendet,
    um einen zum Modell passenden Prozessor zu erstellen.
    Die Funktion gibt eine Liste von BatchFeature-Objekten zurück,
    die die Bilder in den Batches enthalten.
    """

    image_processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
        model_name, use_fast=True
    )
    inputs = image_processor(images)
    batches = batched(inputs["pixel_values"], batch_size)
    return [
        BatchFeature(data={"pixel_values": default_collate(list(batch))}) for batch in batches
    ]


@torch.no_grad()
def generate_image_embeddings(
    model, inp_batches: List[BatchFeature], device="cuda"
) -> np.ndarray:
    """Erzeuge Embeddings für Bilder, die in den Batches übergeben werden.
    Es wird das übergebene Modell verwendet, das auf das übergebene Device verschoben wird.
    Nach der Erzeugung der Embeddings wird das Modell wieder auf das ursprüngliche Device
    zurückverschoben.
    Die Funktion gibt ein numpy-Array zurück, das die Embeddings enthält."""

    olddev = model.device
    model.to(device)
    embeddings = []
    for batch in tqdm(inp_batches):
        batch.to(device)
        out = model(**batch, output_hidden_states=True, output_attentions=False)
        vector = out.pooler_output.detach().cpu().numpy().squeeze()
        batch.to("cpu", non_blocking=True)
        del out, batch
        torch.cuda.empty_cache()
        embeddings.append(vector)
    model.to(olddev)
    return np.vstack(embeddings)
