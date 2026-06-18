# Neurrator

`Neurrator` decodes natural-language narrations from spike trains recorded
with high-density extracellular electrophysiology in mouse visual cortex.
A learned encoder maps short windows of single-unit spike activity into the
patch-embedding space of a frozen vision–language model (CLIP ViT-L/14 with
LLaVA-1.5-7B as the language head); a forward hook replaces LLaVA's vision
tower at inference so the model produces text directly from neural-side
embeddings. Training and evaluation use the public Allen Brain Observatory
Visual Coding — Neuropixels release.


## Repository layout

```
neurrator/
├── README.md                  ← this file
├── LICENSE                    ← MIT (anonymous)
├── requirements.txt
├── neurrator/                 ← importable Python package
│   ├── encoder_patches.py     ← Neurrator-Patches: spikes → 576×1024 patch grid
│   ├── encoder_retrieval.py   ← image-identity retrieval variant
│   ├── losses.py              ← MSE + cosine and InfoNCE
│   ├── llava_inject.py        ← PatchInjector hook for frozen LLaVA
│   └── data.py                ← expected NPZ schema for spikes + CLIP patches
├── scripts/
│   ├── train_patches.py       ← single-session Neurrator-Patches training
│   ├── decode_with_llava.py   ← LLaVA decoding from predicted patches
│   └── evaluate_sbert.py      ← SBERT scoring vs reference captions
└── docs/
    └── architecture.md        ← pipeline + dimension reference
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with `torch>=2.1`, `transformers>=4.40`. A single CUDA-capable GPU is
sufficient for one-session training (~10 minutes); LLaVA-1.5-7B inference
in float16 fits in 16 GB of GPU memory.

## Data

This release does not redistribute neural data. Both stimuli and recordings
are publicly available from the Allen Brain Observatory Visual Coding —
Neuropixels release (`brain_observatory_1.1` protocol):
[brain-map.org/](https://portal.brain-map.org/).

The expected on-disk format produced by the user's preprocessing pipeline is
documented in `neurrator/data.py`:

- per-session NPZ with keys `X_train`, `X_test` (spike-count windows),
  `frame_idx_train`, `frame_idx_test` (which stimulus frame each window is
  centered on), and z-score statistics;
- a `(n_frames, 576, 1024)` array of CLIP ViT-L/14 patch features for the
  presented stimulus, computed once with the standard `open_clip` pipeline.

## Method in one paragraph

A 167 ms window of spike counts (20 bins at 120 Hz) is fed to a small
trainable encoder built from a multi-scale 1-D convolution and a 2-layer
Transformer with cross-attention patch queries. The output is a tensor with
the same shape that CLIP ViT-L/14 produces at its penultimate layer for a
real movie frame: 576 image-patch tokens (a 24×24 grid produced by tiling
a 336×336 image into 14-pixel patches), each a 1024-dimensional embedding.
This patch tensor is the only learned interface between brain and language:
it is handed verbatim to a frozen LLaVA-1.5-7B, whose vision tower is
bypassed at runtime by a forward hook (see `llava_inject.py`). LLaVA emits
free-form text describing the scene the animal was viewing.

## License

MIT. See `LICENSE`.
