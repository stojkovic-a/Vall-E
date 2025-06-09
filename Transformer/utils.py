import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import json
from tqdm import tqdm


def read_file(path: Path):
    extension = path.suffix
    if extension == ".pt":
        return torch.load(path)
    else:
        with open(path, "r") as f:
            content = f.read()
        return content
