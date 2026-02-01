import gc
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def count_parameters(model: SentenceTransformer) -> float:
    return sum(p.numel() for p in model.parameters())


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
