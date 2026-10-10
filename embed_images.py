"""embed_images.py - CLIP embedding of every downloaded image.
Quick test (32 images, saves nothing):  python embed_images.py test
Full run (10-15 min first time):        python embed_images.py
Output: data/Z.npy (n x 512) and data/Z_ids.npy (pageid of each row)."""
import sys

import numpy as np
import pandas as pd

from clip_utils import DEVICE, embed_paths
from common import DATA

test = len(sys.argv) > 1 and sys.argv[1] == "test"
meta = pd.read_csv(DATA / "metadata.csv", keep_default_na=False)
if test:
    meta = meta.sample(32, random_state=0).reset_index(drop=True)
print("Running on:", DEVICE, "| images:", len(meta))

Z = embed_paths(meta["image_path"].tolist())
print("Embeddings:", Z.shape, "| row length ~", float(np.linalg.norm(Z[0])))
if not test:
    np.save(DATA / "Z.npy", Z)
    np.save(DATA / "Z_ids.npy", meta["pageid"].values)
    print("Saved data/Z.npy and data/Z_ids.npy")

# Sanity check: neighbours of 3 random images should look alike.
rng = np.random.default_rng(1)
for i in rng.choice(len(meta), 3, replace=False):
    sims = Z @ Z[i]
    print("\nQUERY:", meta.title[i][:60], "|", meta.seeds[i])
    for j in np.argsort(-sims)[1:4]:
        print(f"   {sims[j]:.2f}  {meta.title[j][:55]} | {meta.seeds[j]}")
