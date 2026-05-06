"""Retrieval-based variant for image-identity decoding experiments.

Instead of regressing CLIP patches directly, this encoder classifies
spike windows over a fixed vocabulary of training images. At inference
on a held-out trial we pick the argmax training image, and feed *its*
real CLIP patches to LLaVA — guaranteeing on-manifold input.

Architecture is a thin variant of NeurratorPatches: same multi-scale
conv + temporal Transformer body, but the head is a single classifier
producing logits over the training-image vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_patches import _MultiScaleConv


class RetrievalEncoder(nn.Module):
    """Spike window -> logits over a fixed training-image vocabulary.

    Parameters
    ----------
    n_neurons : int
    n_classes : int
        Size of the training-image vocabulary.
    d_model : int, default 384
    n_layers : int, default 2
    n_heads : int, default 8
    conv_ch : int, default 128
    dropout : float, default 0.2
    """

    def __init__(
        self,
        n_neurons: int,
        n_classes: int,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 8,
        conv_ch: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_drop = nn.Dropout(dropout)
        self.multi_conv = _MultiScaleConv(n_neurons, conv_ch)
        self.input_proj = nn.Sequential(
            nn.Linear(conv_ch * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention-weighted temporal pooling.
        self.pool_query = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, n_neurons) float tensor

        Returns
        -------
        logits : (B, n_classes) float tensor
        """
        x = self.input_drop(x)
        x = self.multi_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer(x)                                # (B, T, D)
        attn = self.pool_query(x).softmax(dim=1)               # (B, T, 1)
        pooled = (x * attn).sum(dim=1)                          # (B, D)
        return self.head(pooled)                                # (B, n_classes)
