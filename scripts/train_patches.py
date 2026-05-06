#!/usr/bin/env python3
"""Train Neurrator-Patches on a single recording session.

Inputs (paths configurable via --processed-dir / --clip-dir / --session-id):
  <processed-dir>/neural_<stimulus>_<session-id>.npz
  <clip-dir>/clip_vitl14_patches_<stimulus>.npy

See ``neurrator/data.py`` for the expected NPZ schema. This script is a
thin reference implementation; full training of the paper used multi-GPU
orchestration not bundled here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurrator import NeurratorPatches, patch_loss
from neurrator.data import PatchWindowDataset, WINDOW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-id", required=True, help="recording session id")
    p.add_argument("--stimulus", default="nm1", help="nm1 / nm3 / scenes")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--clip-dir", type=Path, default=Path("data/clip_embeddings"))
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    npz = np.load(args.processed_dir / f"neural_{args.stimulus}_{args.session_id}.npz")
    X_tr, X_te = npz["X_train"], npz["X_test"]
    fi_tr, fi_te = npz["frame_idx_train"], npz["frame_idx_test"]
    n_neurons = X_tr.shape[1]

    clip_patches = np.load(args.clip_dir / f"clip_vitl14_patches_{args.stimulus}.npy")

    # Carve a small validation split off the train set (last 2 of 18 repeats
    # in the released preprocessing — adjust if your repeat count differs).
    bins_per_repeat = X_tr.shape[0] // 18
    n_train_repeats = 16
    Xtr, Xval = X_tr[: n_train_repeats * bins_per_repeat], X_tr[n_train_repeats * bins_per_repeat:]
    fitr, fival = fi_tr[: n_train_repeats * bins_per_repeat], fi_tr[n_train_repeats * bins_per_repeat:]

    tr_ds = PatchWindowDataset(Xtr, fitr, clip_patches, WINDOW, augment=True)
    val_ds = PatchWindowDataset(Xval, fival, clip_patches, WINDOW, augment=False)
    te_ds = PatchWindowDataset(X_te, fi_te, clip_patches, WINDOW, augment=False)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, num_workers=4)

    model = NeurratorPatches(n_neurons).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, patience = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = patch_loss(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            cos_sum, n = 0.0, 0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pn = torch.nn.functional.normalize(pred, dim=-1)
                tn = torch.nn.functional.normalize(yb,   dim=-1)
                cos_sum += (pn * tn).sum(-1).mean().item() * xb.size(0)
                n += xb.size(0)
        val_cos = cos_sum / max(n, 1)

        flag = ""
        if val_cos > best:
            best = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            flag = " *"
        else:
            patience += 1
        print(f"[{epoch+1:3d}/{args.epochs}]  val_cos={val_cos:.4f}{flag}  ({time.time()-t0:.1f}s)")
        if patience >= args.patience:
            print("early stop"); break

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"neurrator_patches_{args.stimulus}_{args.session_id}.pt"
    torch.save(best_state, out_path)
    print(f"saved -> {out_path}    best val cos: {best:.4f}")


if __name__ == "__main__":
    main()
