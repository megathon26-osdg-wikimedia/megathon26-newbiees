"""evaluate.py - all results tables on the TEST split.
python evaluate.py            main tables
python evaluate.py --robust   also the blur / tiny / jpeg table (needs CLIP, ~5 min)"""
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA, SEEDS, load_clean
from metrics import (cleaned_precision, lenient_precision, mean_ap, prf, report,
                     subset, tier_recall)
from pipeline import COMP

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")          # index = TRAIN images only
te, tr = idx["test"], idx["train"]
Yte, te_texts = Y[te], [texts[i] for i in te]
comp = sug.components(Z[te], te_texts)
comp["FUSED"] = sug.fuse(comp)
pd.set_option("display.width", 160)

# 1. main table: all labels
main = pd.concat([report(n, *subset(S, Yte)) for n, S in comp.items()])
print("\n=== 1. Strict results, all labels ===\n", main.to_string(index=False))

# 2. hard mode: the seed category itself is hidden (otherwise the task is too easy)
ci = {c: i for i, c in enumerate(cats)}
seeds = set(SEEDS)
non_seed = np.array([i for c, i in ci.items() if c not in seeds])
hard = pd.concat([report(n, *subset(S, Yte, non_seed)) for n, S in comp.items()])
print("\n=== 2. Hard mode: seed categories removed ===\n", hard.to_string(index=False))

# 3. head / torso / tail (by train frequency)
order = np.argsort(-Y[tr].sum(axis=0))
third = len(order) // 3
tiers = {"head": order[:third], "torso": order[third:2 * third], "tail": order[2 * third:]}
tier_rows = [{"scorer": n, **{t: round(tier_recall(S, Yte, 5, c), 3) for t, c in tiers.items()}}
             for n, S in comp.items()]
print("\n=== 3. Recall@5 by category frequency ===\n", pd.DataFrame(tier_rows).to_string(index=False))

# 4. hierarchy: lenient precision and cleaned lists (FUSED, k=5)
Sf = comp["FUSED"]
strict = prf(*subset(Sf, Yte), 5)[0]
cp, avg_len = cleaned_precision(*subset(Sf, Yte), sug.A, 5)
print("\n=== 4. Category graph (FUSED, k=5) ===")
print(f"strict P@5 {strict:.3f} | lenient P@5 (ancestors accepted) "
      f"{lenient_precision(*subset(Sf, Yte), sug.A, 5):.3f}")
print(f"after ancestor suppression: precision {cp:.3f}, average list length {avg_len:.2f}")

# 5. robustness
if "--robust" in sys.argv:
    from clip_utils import degrade, embed_paths
    paths = df["image_path"].values[te].tolist()
    rows = []
    for kind in ["none", "blur", "tiny", "jpeg"]:
        Zd = embed_paths(paths, degrade_fn=lambda im, k=kind: degrade(im, k))
        c = sug.components(Zd, te_texts)
        p, r, f = prf(*subset(sug.fuse(c), Yte), 5)
        rows.append({"image quality": kind, "P@5": round(p, 3), "R@5": round(r, 3),
                     "F1@5": round(f, 3)})
    print("\n=== 5. Robustness (FUSED, k=5) ===\n", pd.DataFrame(rows).to_string(index=False))

main.to_csv(DATA / "results_strict.csv", index=False)
hard.to_csv(DATA / "results_hard.csv", index=False)
k5 = main[main["k"] == 5].set_index("scorer")[["P@k", "R@k", "F1@k"]]
ax = k5.plot(kind="bar", figsize=(7, 4))
ax.set_ylabel("score"); ax.set_title("Category suggestion quality at k = 5 (time-split test set)")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(DATA / "results.png", dpi=200)
print("\nSaved data/results_strict.csv, results_hard.csv, results.png")
