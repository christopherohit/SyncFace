"""Multi-identity dataset provider for Stage A pretraining.

Stage A in the architecture diagram trains a universal motion prior on
a multi-identity corpus while keeping per-identity *temporary* fields
(individual codes + a small canonical hash grid). This module is the
entry point the Stage A trainer will use.

It is a stub: `MultiIdentityNeRFDataset` lays out the contract by
wrapping per-identity `NeRFDataset` instances and tagging each yielded
sample with an integer `identity_id`. The actual sampling strategy
(round-robin vs. weighted, paired same-audio different-identity batches
for the disentangle loss, etc.) is implemented in Phase 3.
"""

import os
from typing import List

import torch
from torch.utils.data import Dataset

from .provider import NeRFDataset


class MultiIdentityNeRFDataset(Dataset):
    def __init__(self, opt, device, identity_paths: List[str], type: str = "train"):
        self.opt = opt
        self.device = device
        self.type = type
        self.identity_paths = identity_paths
        self._datasets: List[NeRFDataset] = []
        self._cum_lengths: List[int] = []
        running = 0
        for path in identity_paths:
            opt_i = _shallow_clone(opt)
            opt_i.path = path
            ds = NeRFDataset(opt_i, device=device, type=type)
            self._datasets.append(ds)
            running += len(ds)
            self._cum_lengths.append(running)

    def __len__(self):
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx):
        identity_id = 0
        local_idx = idx
        for i, end in enumerate(self._cum_lengths):
            if idx < end:
                identity_id = i
                local_idx = idx - (self._cum_lengths[i - 1] if i > 0 else 0)
                break
        sample = self._datasets[identity_id][local_idx]
        sample["identity_id"] = torch.tensor(identity_id, dtype=torch.long)
        return sample


def _shallow_clone(opt):
    """argparse.Namespace lacks a copy() method; build a fresh namespace
    with the same attributes so each dataset can have its own `path`
    without mutating the original opt."""
    import argparse
    new = argparse.Namespace(**vars(opt))
    return new


def discover_identities(corpus_root: str) -> List[str]:
    """Each immediate subdirectory of `corpus_root` is treated as one identity."""
    if not os.path.isdir(corpus_root):
        raise NotADirectoryError(corpus_root)
    return sorted(
        os.path.join(corpus_root, d)
        for d in os.listdir(corpus_root)
        if os.path.isdir(os.path.join(corpus_root, d))
    )
