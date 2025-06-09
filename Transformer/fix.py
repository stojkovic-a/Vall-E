import argparse
import random
from functools import cache
from pathlib import Path

import soundfile
import torch
import torchaudio
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor
from tqdm import tqdm

import config as conf


def main():
    """
    graphemes_preparation.py saves files with names starting
    with \n. To fix run this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--suffix", default=".phn.txt")
    args = parser.parse_args()

    paths = [*args.folder.rglob(f"*{args.suffix}")]

    for path in tqdm(paths):
        filename = path.name
        if filename.startswith("\n"):
            new_name = filename.lstrip("\n")
            new_path = path.parent / new_name
            path.rename(new_path)


if __name__ == "__main__":
    main()
