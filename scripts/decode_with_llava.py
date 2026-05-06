#!/usr/bin/env python3
"""Decode held-out spike windows into natural-language narrations.

Pipeline:
    spike window  → NeurratorPatches  → 576x1024 CLIP-L/14 patch grid
                                       │
                                       ▼
                       PatchInjector hook on LLaVA's vision tower
                                       │
                                       ▼
                            LLaVA-1.5-7B (frozen)
                                       │
                                       ▼
                                    text

Outputs JSON: a list of records  {bin, frame, text}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neurrator import NeurratorPatches, PatchInjector
from neurrator.data import WINDOW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="trained NeurratorPatches .pt")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--session-id", required=True)
    p.add_argument("--stimulus", default="nm1")
    p.add_argument("--n-frames", type=int, default=20,
                   help="number of held-out test frames to decode (evenly spaced)")
    p.add_argument("--out", type=Path, default=Path("results/narrations.json"))
    p.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--prompt", default="USER: <image>\nDescribe this scene in one sentence.\nASSISTANT:")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    npz = np.load(args.processed_dir / f"neural_{args.stimulus}_{args.session_id}.npz")
    X_te, fi_te = npz["X_test"], npz["frame_idx_test"]
    n_neurons = X_te.shape[1]
    half = WINDOW // 2

    state = torch.load(args.ckpt, map_location=device)
    model = NeurratorPatches(n_neurons).to(device).eval()
    model.load_state_dict(state)

    # Pick `n_frames` evenly spaced bins inside the windowed range.
    valid_lo, valid_hi = half, len(X_te) - half - 1
    bins = np.linspace(valid_lo, valid_hi, args.n_frames).round().astype(int)

    # Predict the patch grid for each chosen bin.
    windows = torch.stack([
        torch.tensor(X_te[c - half : c + half], dtype=torch.float32) for c in bins
    ]).to(device)
    with torch.no_grad():
        preds = model(windows)                          # (n_frames, 576, 1024)
    preds = preds.to(torch.float16)

    # Set up frozen LLaVA.
    from transformers import LlavaProcessor, LlavaForConditionalGeneration  # type: ignore
    print(f"loading {args.llava_id} ...")
    processor = LlavaProcessor.from_pretrained(args.llava_id)
    llava = LlavaForConditionalGeneration.from_pretrained(
        args.llava_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    original_forward = llava.vision_tower.forward

    # Dummy image: vision tower is hijacked, but the processor still needs
    # an image to set up the input ids correctly.
    dummy = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))

    results = []
    for i, c in enumerate(bins):
        inputs = processor(images=dummy, text=args.prompt,
                           return_tensors="pt").to(device, torch.float16)
        llava.vision_tower.forward = PatchInjector(preds[i:i+1])
        with torch.no_grad():
            out = llava.generate(**inputs, max_new_tokens=60, do_sample=False)
        text = processor.decode(out[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        llava.vision_tower.forward = original_forward

        frame = int(fi_te[c])
        results.append({"bin": int(c), "frame": frame, "text": text})
        print(f"  bin {c:5d}  frame {frame:5d}  -> {text}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
