# Neurrator architecture

## Pipeline

```
        ┌────────────────┐    ┌────────────────────┐    ┌──────────────┐
spikes  │ Neurrator-     │ →  │  PatchInjector     │ →  │ frozen LLaVA │ → text
(B,T,N) │ Patches (12 M  │    │  hook on LLaVA's   │    │  -1.5-7B     │
        │ params)        │    │  vision tower      │    │              │
        └────────────────┘    └────────────────────┘    └──────────────┘
              ▲                       ▲
              │                       │
        targets:                output of NeurratorPatches:
        CLIP ViT-L/14           same shape that CLIP ViT-L/14
        patch grid              produces for a real image
        (576, 1024)             (576, 1024)
```

## Key dimensions

| symbol | value | meaning |
|---|---|---|
| `T`         | 20      | bins per input window (167 ms at 120 Hz) |
| `N`         | ~2 000  | recorded units per session (z-scored) |
| `n_patches` | 576     | image-patch tokens (24×24 grid for 336-px image, 14-px patch) |
| `patch_dim` | 1024    | CLIP ViT-L/14 hidden size |
| `d_model`   | 384     | internal Transformer width |

## Encoder (NeurratorPatches)

Implemented in `neurrator/encoder_patches.py`. Total ~12 M parameters.

1. **Multi-scale 1-D conv**, three parallel branches with kernels (3, 7, 15)
   on the temporal axis; outputs are concatenated and projected to
   `d_model`.
2. **2-layer Transformer encoder** (8 heads, FF = 2 × d_model) over the
   temporal axis.
3. **Cross-attention from learned patch queries**: `n_patches` learnable
   query tokens of size `d_model` attend over the temporal embedding.
4. **2-layer patch-level Transformer** to learn spatial relations between
   patch tokens.
5. **Linear projection** `d_model → patch_dim`.

The output `(B, n_patches, patch_dim)` matches CLIP ViT-L/14's penultimate
layer for a real image, so it can be fed directly to LLaVA's projector.

## Loss

`patch_loss(pred, target) = 0.5 * MSE + 0.5 * (1 − cosine)` on flattened
patch features (see `neurrator/losses.py`). An optional `info_nce` term
on pooled patches is provided to discourage mode collapse when the
training set is small.

## LLaVA injection

LLaVA-1.5-7B's vision tower normally returns a CLIP ViT-L/14 hidden state
of shape `(B, 577, 1024)` — 1 CLS token + 576 patches. To narrate from
neural activity we replace `model.vision_tower.forward` with a
`PatchInjector` callable that returns the predicted patch grid wrapped in
the same `BaseModelOutputWithPooling` dataclass. The projector and LM
are untouched and stay frozen.

See `neurrator/llava_inject.py`.

## Retrieval variant

For experiments that decode held-out *image identities* (rather than
held-out frames within a familiar movie), the encoder is reframed as a
classifier over the training-image vocabulary (see
`neurrator/encoder_retrieval.py`). At inference on a held-out trial we
take the argmax training image and feed *its* real CLIP patches to LLaVA,
guaranteeing on-manifold input to the language model.
