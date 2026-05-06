"""Expected on-disk format and a windowed dataset over spike + patch pairs.

This release does not bundle data. The format below is what the user's
preprocessing pipeline produces from the public Allen Brain Observatory
Visual Coding — Neuropixels recordings.

Per-session NPZ (one file per recording session):

    X_train          : (n_train_bins, n_neurons) float32  — z-scored spike
                       counts at 120 Hz
    X_test           : (n_test_bins,  n_neurons) float32
    frame_idx_train  : (n_train_bins,) int64              — index of the
                       stimulus frame the bin's center is aligned to
    frame_idx_test   : (n_test_bins,)  int64
    zscore_mean      : (1, n_neurons) float32
    zscore_std       : (1, n_neurons) float32
    unit_areas       : (n_neurons,)   <U5                 — per-unit brain
                       region label

Per-stimulus CLIP patch table (one file per stimulus):

    clip_vitl14_patches_<stimulus>.npy
        shape (n_frames, 576, 1024) float32
        produced by `open_clip` ViT-L-14-336 on the original frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


WINDOW = 20  # 20 bins at 120 Hz = 167 ms


class PatchWindowDataset(Dataset):
    """(spike-window, true-CLIP-patch) pairs for Neurrator-Patches training.

    For each valid time index `c` in the recording, returns a window
    `X[c-WINDOW//2 : c+WINDOW//2]` and the CLIP patch grid of the frame
    that bin `c` is aligned to.

    Parameters
    ----------
    X : (n_bins, n_neurons) float array.
    frame_idx : (n_bins,) int array, frame the bin is aligned to.
    clip_patches : (n_frames, n_patches, patch_dim) float array.
    window : int, default WINDOW.
    augment : bool, default False
        If True, randomly jitter the window center by +/- 2 bins and apply
        independent neuron dropout (10% per neuron).
    """

    def __init__(
        self,
        X: np.ndarray,
        frame_idx: np.ndarray,
        clip_patches: np.ndarray,
        window: int = WINDOW,
        augment: bool = False,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.frame_idx = torch.tensor(frame_idx, dtype=torch.long)
        self.clip_patches = torch.tensor(clip_patches, dtype=torch.float32)
        self.half_w = window // 2
        self.start = self.half_w
        self.end = len(X) - self.half_w
        self.augment = augment

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        c = idx + self.start
        if self.augment:
            jitter = int(torch.randint(-2, 3, (1,)).item())
            c = max(self.half_w, min(c + jitter, len(self.X) - self.half_w - 1))
        w = self.X[c - self.half_w : c + self.half_w].clone()
        if self.augment:
            mask = (torch.rand(w.shape[1]) > 0.1).float()
            w = w * mask.unsqueeze(0)
        target = self.clip_patches[self.frame_idx[idx + self.start]]
        return w, target


class TrialClassDataset(Dataset):
    """(spike-window, integer-class-label) pairs for the retrieval encoder.

    Parameters
    ----------
    X : (n_bins, n_neurons) float array.
    frame_idx : (n_bins,) int array.
    frame_to_class : dict[int, int]
        Maps each training frame id to a contiguous class index 0..n_classes-1.
    window : int, default WINDOW.
    augment : bool, default False.
    """

    def __init__(
        self,
        X: np.ndarray,
        frame_idx: np.ndarray,
        frame_to_class: dict,
        window: int = WINDOW,
        augment: bool = False,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.frame_idx = frame_idx
        self.frame_to_class = frame_to_class
        self.half_w = window // 2
        self.start = self.half_w
        self.end = len(X) - self.half_w
        self.augment = augment

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        c = idx + self.start
        if self.augment:
            jitter = int(torch.randint(-2, 3, (1,)).item())
            c = max(self.half_w, min(c + jitter, len(self.X) - self.half_w - 1))
        w = self.X[c - self.half_w : c + self.half_w].clone()
        if self.augment:
            mask = (torch.rand(w.shape[1]) > 0.1).float()
            w = w * mask.unsqueeze(0)
        frame = int(self.frame_idx[idx + self.start])
        return w, self.frame_to_class[frame]
