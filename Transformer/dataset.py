import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import json
from tqdm import tqdm
from utils import read_file


class PQDataset(Dataset):

    def _create_vocab(self, files: list[Path], add_padd=True, add_eos=True):
        """
        Only for files containing strings
        """
        vocab = dict()
        i = 0
        if add_padd == True:
            vocab["<PAD>"] = i
            i += 1
        for path in tqdm(files):
            with open(path, "r") as f:
                text = f.read()
            symbols = text.split(" ")
            for symbol in symbols:
                if symbol not in vocab:
                    vocab[symbol] = i
                    i += 1

        if add_eos == True:
            vocab["<EOS>"] = i
            i += 1
        return vocab

    def _save_dict(self, d, name):
        with open(name, "w") as f:
            json.dump(d, f, indent=4)

    def _load_dict(self, name):
        with open(name, "r") as f:
            loaded_dict = json.load(f)
        return loaded_dict

    def __init__(
        self, phonem_dir, qnt_dir, qnt_suffix=".qnt.pt", phonem_suffix=".phn.txt"
    ):
        super().__init__()
        self.phonem_dir = phonem_dir
        self.qnt_dir = qnt_dir
        self.qnt_suffix = qnt_suffix
        self.phonem_suffix = phonem_suffix
        self.phonem_paths = sorted(list(Path(phonem_dir).rglob(f"*{phonem_suffix}")))
        self.qnt_paths = sorted(list(Path(qnt_dir).rglob(f"*{qnt_suffix}")))
        assert len(self.phonem_paths) == len(self.qnt_paths)
        for i in range(len(self.phonem_paths)):
            assert (
                self.phonem_paths[i].stem.split(".")[0]
                == self.qnt_paths[i].stem.split(".")[0]
            )
        self.phonem_vocab = self._create_vocab(self.phonem_paths)
        self._save_dict(self.phonem_vocab, "./phoneme.json")
        # self.qnt_vocab = self._create_vocab(self.qnt_paths)
        # self._save_dict(self.qnt_vocab, "./qnt.json")

    def __len__(self):
        return len(self.phonem_paths)

    def _get_phonem_vocab_size(self):
        return len(self.phonem_vocab)

    def _get_qnt_vocab_size(self):
        return 1026

    def __getitem__(self, idx):
        phonem_path = self.phonem_paths[idx]
        qnt_path = self.qnt_paths[idx]
        phonems = read_file(phonem_path)
        qnts = read_file(qnt_path).squeeze()
        phonemes_vocab = [self.phonem_vocab[x] for x in phonems.split(" ")]
        qnts_vocab = qnts + 1
        return phonemes_vocab, qnts_vocab


if __name__ == "__main__":
    ds = PQDataset(
        "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500",
        "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500",
        ".qnt.pt",
        ".phn.txt",
    )
    ds.__get_phonem_vocab_size()
    # test = torch.load(
    #     "/home/diffine/Aleksandar/Transformer/Dataset/LibriSpeech/train-other-500/1006/135212/1006-135212-0000.qnt.pt"
    # )
    # pass
