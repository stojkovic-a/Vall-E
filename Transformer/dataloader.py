import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torch
import json
from tqdm import tqdm
from utils import read_file
from dataset import PQDataset
import torch.nn.functional as F


def get_dataloader(phonem_dir, qnt_dir, phonem_suffix, qnt_suffix, batch_size):
    dataset = PQDataset(phonem_dir, qnt_dir, qnt_suffix, phonem_suffix)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    return dataloader, dataset


def collate_fn(batch):
    """
    Returns a tuple with first element being a list (batch) of padded_phonemes
    lists and second element list (batch) of padded qnts tensors
    """
    phonemes = [item[0] for item in batch]
    qnts = [item[1] for item in batch]
    longest_phonem = max(len(p) for p in phonemes)
    padded_phonemes = [p + [0] * (longest_phonem - len(p)) for p in phonemes]
    longest_qnts = max(max(len(inner) for inner in outer) for outer in qnts)
    padded_qnts = [
        F.pad(q, [0, longest_qnts - q.shape[1],0,0], mode="constant", value=0)
        for q in qnts
    ]
    return padded_phonemes, padded_qnts


if __name__ == "__main__":
    dataloader, _ = get_dataloader(
        "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500",
        "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500",
        ".phn.txt",
        ".qnt.pt",
        16,
    )

    for d in dataloader:
        pass
