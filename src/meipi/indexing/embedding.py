from itertools import batched
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import default_collate
import torch
from transformers import AutoImageProcessor, BatchFeature
import gc

def check_cuda_memory():
    for obj in gc.get_objects():
        try:
            if obj.device.type == 'cuda' and type(obj) == torch.Tensor:
                print(type(obj), obj.size(), obj.device)
        except:
            pass

def create_image_batches(images, model_name, batch_size):    
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast = True)
    inputs = image_processor(images) 
    batches = batched(inputs["pixel_values"], batch_size)
    return [BatchFeature(data={"pixel_values": default_collate(batch)}) for batch in batches]


@torch.no_grad()
def generate_image_embeddings(model, inp_batches, device="cuda"):
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
