"""Neurrator-Patches: spike windows -> CLIP ViT-L/14 patch grid.

Output shape (B, n_patches, patch_dim) = (B, 576, 1024) matches the
penultimate-layer activations CLIP ViT-L/14 produces for a 336x336 image
tiled into 14-pixel patches (24x24 grid). The predicted patches are then
fed to a frozen LLaVA-1.5-7B via a forward hook on its vision tower
(see ``llava_inject.py``).

Architecture
------------
    spikes (B, T, n_neurons)            ── 20 bins (167 ms) at 120 Hz
        └─> multi-scale 1D conv          (kernels 3, 7, 15)
            └─> linear + LayerNorm        d_model = 384
                └─> 2-layer Transformer
                    └─> cross-attention from learned patch queries
                        └─> 2-layer patch self-attention
                            └─> linear projection to patch_dim = 1024
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MultiScaleConv(nn.Module):
    """Three parallel 1-D convolutions with different temporal kernels.

    Captures spike features at multiple timescales from short windows.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                nn.GELU(),
                nn.BatchNorm1d(out_ch),
            )
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_neurons, T)  — already permuted by caller
        return torch.cat([b(x) for b in self.branches], dim=1)


class NeurratorPatches(nn.Module):
    """Predict CLIP ViT-L/14 patch features from a spike-count window.

    Parameters
    ----------
    n_neurons : int
        Number of recorded units in the input window.
    n_patches : int, default 576
        Number of patch tokens produced (24*24 for CLIP ViT-L/14 at 336 px).
    patch_dim : int, default 1024
        Per-patch embedding dimension (CLIP ViT-L/14 hidden size).
    d_model : int, default 384
        Internal transformer width.
    n_layers : int, default 2
        Number of temporal Transformer layers.
    n_heads : int, default 8
        Multi-head attention heads.
    conv_ch : int, default 128
        Channels per multi-scale conv branch (3 branches concat to 3*conv_ch).
    dropout : float, default 0.2
    """

    def __init__(
        self,
        n_neurons: int,
        n_patches: int = 576,
        patch_dim: int = 1024,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 8,
        conv_ch: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = patch_dim

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

        # Learned patch queries that cross-attend to the temporal embedding.
        self.patch_queries = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ca_norm = nn.LayerNorm(d_model)

        # Patch-level self-attention to learn spatial relations between tokens.
        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.patch_transformer = nn.TransformerEncoder(patch_layer, num_layers=2)

        # Project to CLIP-L/14 patch dimension (1024).
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, n_neurons) float tensor
            Spike-count window.

        Returns
        -------
        patches : (B, n_patches, patch_dim) float tensor
            Predicted CLIP ViT-L/14 patch grid.
        """
        B = x.size(0)

        x = self.input_drop(x)
        x = self.multi_conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, conv*3)
        x = self.input_proj(x)                                     # (B, T, D)
        x = self.transformer(x)                                    # (B, T, D)

        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, x, x)
        patches = self.ca_norm(queries + attn_out)                 # (B, P, D)
        patches = self.patch_transformer(patches)
        patches = self.out_proj(patches)                           # (B, P, patch_dim)
        return patches
