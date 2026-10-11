import numpy as np

from common import SEEDS, load_clean

df, cats, Z, Y, idx, texts = load_clean()
tr, te = idx["train"], idx["test"]
order = np.argsort(-Y[tr].sum(axis=0))
third = len(order) // 3
tiers = {"head": order[:third], "torso": order[third:2 * third], "tail": order[2 * third:]}
for name, cols in tiers.items():
    n_img = int((Y[te][:, cols].sum(axis=1) > 0).sum())
    n_tr = Y[tr][:, cols].sum(axis=0)
    print(f"{name:6s}: {len(cols):3d} categories, {n_img:3d} test images, "
          f"train images per category {int(n_tr.min())}-{int(n_tr.max())}")
ci = {c: i for i, c in enumerate(cats)}
non_seed = [i for c, i in ci.items() if c not in set(SEEDS)]
print(f"hard mode: {int((Y[te][:, non_seed].sum(axis=1) > 0).sum())} test images have a non-seed label")
