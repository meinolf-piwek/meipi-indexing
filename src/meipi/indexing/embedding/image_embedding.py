"""Helpers for batching images and generating image embeddings.

This module provides small, model-agnostic utilities used by indexing jobs:

- pre-processing image inputs into model-compatible batches
- running batched forward passes on a selected device
- collecting pooled vectors as ``numpy.ndarray``
"""

from typing import List
from itertools import batched
import gc
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import default_collate
import torch
from transformers import AutoImageProcessor, BatchFeature, BaseImageProcessor
from transformers.image_utils import ImageInput


def check_cuda_memory() -> None:
    """Print all currently alive CUDA tensors for debugging memory usage."""
    for obj in filter(lambda o: isinstance(o, torch.Tensor), gc.get_objects()):
        if obj.device.type == "cuda":
            print(type(obj), obj.size(), obj.device)


def create_image_batches(
    images: ImageInput, model_name: str, batch_size: int
) -> List[BatchFeature]:
    """Create model-ready image batches using the matching HuggingFace processor.

    Args:
        images: Raw image inputs accepted by ``transformers``.
        model_name: HuggingFace model id used to resolve ``AutoImageProcessor``.
        batch_size: Number of samples per returned batch.

    Returns:
        A list of ``BatchFeature`` objects containing ``pixel_values`` tensors.
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
    """Generate pooled embeddings for all batches and return one stacked array.
    The model is temporarily moved to ``device`` for inference and restored to its
    original device afterwards.
    """
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

