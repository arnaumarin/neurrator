"""Neurrator — natural-language decoding from single-neuron spike trains."""

from .encoder_patches import NeurratorPatches
from .encoder_retrieval import RetrievalEncoder
from .losses import patch_loss, info_nce
from .llava_inject import PatchInjector

__all__ = [
    "NeurratorPatches",
    "RetrievalEncoder",
    "patch_loss",
    "info_nce",
    "PatchInjector",
]
