"""Losses used to train Neurrator-Patches.

`patch_loss`  : 0.5 * MSE + 0.5 * (1 - cosine) on flattened patch features.
`info_nce`    : optional contrastive term that prevents the encoder from
                collapsing to a mean prediction across frames; positives
                are matching frame pairs within a mini-batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_loss(pred: torch.Tensor, target: torch.Tensor,
               mse_w: float = 0.5, cos_w: float = 0.5) -> torch.Tensor:
    """Combined MSE + cosine loss on patch features.

    Parameters
    ----------
    pred, target : (B, n_patches, patch_dim) tensors.

    Returns
    -------
    loss : scalar tensor.
    """
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(
        F.normalize(pred, dim=-1), F.normalize(target, dim=-1), dim=-1
    ).mean()
    return mse_w * mse + cos_w * (1.0 - cos)


def info_nce(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE on pooled patch predictions.

    Pools each batch element across patches, L2-normalises, then asks the
    pooled prediction to be closer to its own pooled target than to any
    other target in the batch. Discourages mode collapse: predictions for
    different frames must remain distinguishable in the pooled space.

    Parameters
    ----------
    pred, target : (B, n_patches, d_patch) tensors.
    tau : float, temperature.

    Returns
    -------
    loss : scalar tensor.
    """
    p = F.normalize(pred.mean(dim=1), dim=-1)
    t = F.normalize(target.mean(dim=1), dim=-1)
    logits = p @ t.t() / tau
    labels = torch.arange(p.size(0), device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
