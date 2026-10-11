"""cutoff_test.py - choose the display cutoff using OUTSIDE images (not in any category).

1. Put 15-20 photos that belong to none of our categories in data/outside/ (jpg/png).
2. Run:  python cutoff_test.py
The cutoff balances two things on a grid of thresholds t:
  - on VALIDATION images: F1 of the suggestions that stay visible (score >= t),
  - on OUTSIDE images:    share that get NO suggestion at all (max score < t).
It picks the t with the best harmonic mean of the two and writes data/cutoff.json.
Honest note: fusion was fitted on the validation split, so the validation side is a little
optimistic, and 15-20 outside images give a rough cutoff, not a precise one."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import DATA, load_clean

K = 5


def val_curves(S, Y, thresholds):
    """Precision / recall / F1 of the top-K suggestions that also pass each threshold."""
    top = np.argsort(-S, axis=1)[:, :K]
    sc = np.take_along_axis(S, top, axis=1)
    hit = np.take_along_axis(Y, top, axis=1)
    n_true = max(Y.sum(), 1)
    rows = []
    for t in thresholds:
        keep = sc >= t
        shown = keep.sum()
        good = (hit * keep).sum()
        p = good / shown if shown else 1.0
        r = good / n_true
        rows.append((p, r, 2 * p * r / max(p + r, 1e-9)))
    return np.array(rows)


def choose_cutoff(S_val, Y_val, max_out, thresholds=None):
    thresholds = np.arange(0.01, 0.96, 0.01) if thresholds is None else thresholds
    curves = val_curves(S_val, Y_val, thresholds)
    reject = np.array([(max_out < t).mean() for t in thresholds])
    f1 = curves[:, 2]
    score = 2 * f1 * reject / np.maximum(f1 + reject, 1e-9)
    b = int(np.argmax(score))
    return float(thresholds[b]), thresholds, curves, reject


if __name__ == "__main__":
    from clip_utils import embed_paths

    folder = DATA / "outside"
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) < 10:
        raise SystemExit(f"Found only {len(files)} images in {folder}. Put 15-20 outside images there.")
    df, cats, Z, Y, idx, texts = load_clean()
    sug = joblib.load(DATA / "suggester.joblib")
    va = idx["val"]
    S_val = sug.fuse(sug.components(Z[va], [texts[i] for i in va]))

    Zo = embed_paths([str(p) for p in files])
    S_out = sug.fuse(sug.components(Zo, [""] * len(files)))     # no title: image only
    max_out = S_out.max(axis=1)
    print("\nOutside images (top suggestion and score):")
    for p, s, row in zip(files, max_out, S_out):
        print(f"  {s:.3f}  {cats[int(np.argmax(row))]:<35} {p.name}")

    t, grid, curves, reject = choose_cutoff(S_val, Y[va], max_out)
    show = pd.DataFrame({"cutoff": grid, "val P (kept)": curves[:, 0].round(3),
                         "val R (kept)": curves[:, 1].round(3), "val F1": curves[:, 2].round(3),
                         "outside rejected": reject.round(2)})
    print("\n", show.iloc[[0, 4, 9, 14, 19, 29, 39, 49, 59, 74]].to_string(index=False))
    i = int(round((t - 0.01) / 0.01))
    print(f"\nChosen cutoff {t:.2f}: validation F1 {curves[i, 2]:.3f}, "
          f"{reject[i] * 100:.0f}% of {len(files)} outside images get no suggestion "
          f"(old cutoff 0.05 rejected {(max_out < 0.05).mean() * 100:.0f}%).")
    json.dump({"cutoff": t, "n_outside": len(files), "val_f1": float(curves[i, 2]),
               "outside_rejected": float(reject[i])}, open(DATA / "cutoff.json", "w"), indent=1)
    print("Saved data/cutoff.json (demo_app.py reads it through cutoff.load_cutoff())")
