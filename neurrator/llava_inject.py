"""PatchInjector: bypass LLaVA's vision tower with a predicted patch grid.

LLaVA-1.5-7B's vision tower normally returns a CLIP ViT-L/14 hidden-state
sequence of shape (B, 577, 1024) (1 CLS token + 576 patches). To narrate
from neural activity we replace the tower's `forward` with a callable
that returns the predicted patch grid wrapped in the same dataclass.
The rest of the LLaVA stack (projector + LM) is untouched.

Usage
-----
    from transformers import LlavaForConditionalGeneration
    from neurrator import PatchInjector

    llava = LlavaForConditionalGeneration.from_pretrained(...)
    original_forward = llava.vision_tower.forward

    # `patches` is a (1, 576, 1024) float16 tensor predicted by Neurrator.
    llava.vision_tower.forward = PatchInjector(patches)
    out = llava.generate(**inputs, max_new_tokens=60, do_sample=False)

    # Restore for subsequent calls.
    llava.vision_tower.forward = original_forward
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PatchInjector:
    """Callable that returns a fixed patch grid from any vision-tower call.

    Parameters
    ----------
    patches : (B, 576, 1024) float tensor on the same device as LLaVA
        weights. A learnable CLS token is prepended (zeros) so the output
        matches the shape LLaVA's projector expects (B, 577, 1024).
    n_hidden_states : int, default 25
        LLaVA-1.5-7B's projector reads from a specific hidden-state index;
        we replicate the final hidden state across the depth so that any
        index lookup returns the predicted patches.
    """

    def __init__(self, patches: torch.Tensor, n_hidden_states: int = 25) -> None:
        self._patches = patches
        self._n_hidden_states = n_hidden_states

    def __call__(self, *args, **kwargs) -> BaseModelOutputWithPooling:
        p = self._patches
        B = p.size(0)
        cls = torch.zeros(B, 1, p.size(-1), dtype=p.dtype, device=p.device)
        full = torch.cat([cls, p], dim=1)                    # (B, 577, 1024)
        return BaseModelOutputWithPooling(
            last_hidden_state=full,
            pooler_output=full[:, 0],
            hidden_states=tuple([full] * self._n_hidden_states),
        )
