#!/usr/bin/env python3
"""Score decoded narrations against reference captions with SBERT cosine.

For each decoded narration, computes:
  - matched   : SBERT cosine vs the true frame's reference caption
  - shuffled  : SBERT cosine vs a random other reference (within-batch null)
  - floor     : SBERT cosine vs a fixed pool of word-salad sentences

Inputs:
  narrations_json — list of records {frame, text} produced by
                    decode_with_llava.py
  captions_json   — dict mapping str(frame_id) -> [str, ...] of reference
                    captions (e.g. BLIP-2 captions of the original stimulus)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Pure word-salad sentences for the random-sentence floor (30 sentences,
# matching the appendix specification). Domain-disjoint from natural-scene
# and natural-movie content, so they give the cleanest possible
# "matched >> random" demonstration.
WORD_SALAD = [
    "Purple elephant denominator sandwich quintuple.",
    "Whisper gargle xylophone barometer flutter.",
    "Octopus ketchup spinning thunder pickle.",
    "Marbled grumble forty-seven hyperbolic potato.",
    "Velvet algorithm sneezes asymmetric piano.",
    "Triangle banana crackles under sleeping moonlight.",
    "Quibbling froth perpendicular tomato eclipse.",
    "Spoon helicopter zenith gargantuan zebra.",
    "Crimson syllogism juggles tangerine vortex.",
    "Inverted pickle flux capacitor lemon waltz.",
    "Snickering geometry hibernates beneath staircase.",
    "Onomatopoeia quintessence wobbles parabolically.",
    "Saxophone weasel orbits the centrifugal raisin.",
    "Mahogany whispers detonate the linear mustache.",
    "Velveteen cyclones harvest oblique pumpkins.",
    "Unicycle paradox toothpick symphony decoupled.",
    "Trapezoidal gerbils ferment polychromatic gelatin.",
    "Galloping silverware oxidizes the thoughtful nebula.",
    "Pomegranate vector marbleizes the recursive tuba.",
    "Limpid hexagon snorkels through algorithmic syrup.",
    "Gargoyle sneeze quadrilateral lavender persimmon.",
    "Nasal trapezoid ricochets viscous trapezium teapot.",
    "Decagonal oboe pickles the meandering thunderclap.",
    "Filibustering noodle calibrates angular crustacean.",
    "Polyhedral seagull marinates the inverted clavichord.",
    "Lopsided chrysanthemum juggles encrypted jellybeans.",
    "Phosphorescent narwhal yodels at parabolic mittens.",
    "Tessellated cucumber denounces the hyperbolic kazoo.",
    "Quivering meringue circumscribes asymptotic flugelhorn.",
    "Pneumatic walnut transverses the lemniscate xylophone.",
]
assert len(WORD_SALAD) == 30, "appendix specifies 30 random-sentence floor entries"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--narrations", type=Path, required=True)
    p.add_argument("--captions",   type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/sbert_scores.json"))
    p.add_argument("--sbert-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    narrations = json.loads(args.narrations.read_text())
    captions   = json.loads(args.captions.read_text())

    decoded = [r["text"] for r in narrations]
    refs    = [captions[str(r["frame"])][0] for r in narrations]

    from sentence_transformers import SentenceTransformer  # type: ignore
    sbert = SentenceTransformer(args.sbert_model, device=args.device)

    e_dec  = sbert.encode(decoded,    normalize_embeddings=True, show_progress_bar=False)
    e_ref  = sbert.encode(refs,       normalize_embeddings=True, show_progress_bar=False)
    e_floor = sbert.encode(WORD_SALAD, normalize_embeddings=True, show_progress_bar=False)

    matched = (e_dec * e_ref).sum(-1)
    floor   = (e_dec @ e_floor.T).mean(axis=1)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(narrations))
    shuffled = (e_dec * e_ref[perm]).sum(-1)

    summary = {
        "n_decoded":   int(len(narrations)),
        "n_unique":    int(len(set(decoded))),
        "matched":  {"mean": float(matched.mean()),  "std": float(matched.std())},
        "shuffled": {"mean": float(shuffled.mean()), "std": float(shuffled.std())},
        "floor":    {"mean": float(floor.mean()),    "std": float(floor.std())},
        "delta_matched_minus_shuffled": float(matched.mean() - shuffled.mean()),
        "delta_matched_minus_floor":    float(matched.mean() - floor.mean()),
        "per_trial": [
            {"frame": r["frame"], "decoded": r["text"], "reference": refs[i],
             "matched": float(matched[i]),
             "shuffled": float(shuffled[i]),
             "floor":    float(floor[i])}
            for i, r in enumerate(narrations)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"  matched  {summary['matched']['mean']:.3f} ± {summary['matched']['std']:.3f}")
    print(f"  shuffled {summary['shuffled']['mean']:.3f} ± {summary['shuffled']['std']:.3f}")
    print(f"  floor    {summary['floor']['mean']:.3f} ± {summary['floor']['std']:.3f}")
    print(f"  Δ matched−shuffled  {summary['delta_matched_minus_shuffled']:+.3f}")
    print(f"  unique sentences    {summary['n_unique']}/{summary['n_decoded']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
