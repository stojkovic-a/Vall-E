import argparse
import random
import string
from functools import cache
from pathlib import Path
import pathlib
import torch
from tqdm import tqdm
from g2p_en import G2p
import sys
import nltk
from g2p import _get_graphs, encode


def _save_files_dict(path: Path, d: dict):
    for key in d.keys():
        file_name = path.parent / f"{key}.phn.txt"
        with open(file_name, "w") as f:
            f.write(" ".join(d[key]))


def _parse_multigraphs(graphs):
    d = dict()
    i = 0
    j = 0
    for index in range(len(graphs)):
        if graphs[index] == " " and i == 0:
            i = index
            file_name = graphs[j:i]
        elif graphs[index] == "\n":
            graphem = graphs[i + 1 : index]
            phonem = encode(graphem)
            d[file_name] = phonem
            i = 0
            j = index
    return d


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--suffix", type=str, default=".trans.txt")
    args = parser.parse_args()

    paths = list(args.dir.rglob(f"*{args.suffix}"))
    for path in tqdm(paths):
        graphs = _get_graphs(path)
        d = _parse_multigraphs(graphs)
        _save_files_dict(path, d)


if __name__ == "__main__":
    # sys.argv = [
    #     "g2p.py",
    #     "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500/20/205",
    #     "--suffix",
    #     ".trans.txt",
    # ]
    main()
