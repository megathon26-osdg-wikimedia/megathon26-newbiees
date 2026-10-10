"""prepare_labels.py - clean labels, split by time, remove near-duplicates.
Run after embed_images.py.  Output: data/clean.csv, Y.npy, Zc.npy, cats.json"""
import json

import numpy as np
import pandas as pd

from common import DATA, SEEDS, is_junk

N_MIN = 8        # a category needs this many TRAIN images to be kept
DUP = 0.97       # val/test image this similar to a train image = near-duplicate

meta = pd.read_csv(DATA / "metadata.csv", keep_default_na=False)
Z = np.load(DATA / "Z.npy")
ids = np.load(DATA / "Z_ids.npy")

# 1. align by pageid (never by row order) and drop unreadable images
row = pd.Series(np.arange(len(ids)), index=ids)
meta = meta[meta["pageid"].isin(row.index)].copy()
meta["zrow"] = row[meta["pageid"]].values
meta = meta[np.linalg.norm(Z[meta["zrow"].values], axis=1) > 0.5]

# 2. drop process/maintenance categories
meta["labels_all"] = meta["categories"].apply(
    lambda s: [c for c in s.split("|") if c and not is_junk(c)])
meta = meta[meta["labels_all"].str.len() > 0]

# 3. split by upload time: oldest 70% train, next 15% val, newest 15% test
ts = pd.to_datetime(meta["timestamp"], errors="coerce", utc=True)
meta = meta.assign(ts=ts.fillna(ts.min())).sort_values("ts", kind="stable")
meta = meta.reset_index(drop=True)
n = len(meta)
pos = np.arange(n)
meta["split"] = np.where(pos < 0.70 * n, "train", np.where(pos < 0.85 * n, "val", "test"))

# 4. keep categories that have enough TRAIN images (counting on train only)
counts = pd.Series([c for cs in meta.loc[meta["split"] == "train", "labels_all"]
                    for c in cs]).value_counts()
keep = set(counts[counts >= N_MIN].index)
meta["labels"] = meta["labels_all"].apply(lambda cs: [c for c in cs if c in keep])
meta = meta[meta["labels"].str.len() > 0].reset_index(drop=True)
cats = sorted({c for cs in meta["labels"] for c in cs})
ci = {c: i for i, c in enumerate(cats)}
Y = np.zeros((len(meta), len(cats)), dtype="float32")
for i, cs in enumerate(meta["labels"]):
    for c in cs:
        Y[i, ci[c]] = 1
Zc = Z[meta["zrow"].values]

# 5. near-duplicates: drop val/test images that almost copy a train image
tr = np.where(meta["split"] == "train")[0]
for name in ("val", "test"):
    idx = np.where(meta["split"] == name)[0]
    dup = (Zc[idx] @ Zc[tr].T).max(axis=1) >= DUP
    meta.loc[idx[dup], "split"] = "dropped"
    print(f"{name}: removed {dup.sum()} near-duplicates")

# 6. save
meta["labels"] = meta["labels"].apply("|".join)
meta[["pageid", "title", "timestamp", "seeds", "description", "image_path",
      "split", "labels"]].to_csv(DATA / "clean.csv", index=False)
np.save(DATA / "Y.npy", Y)
np.save(DATA / "Zc.npy", Zc)
json.dump(cats, open(DATA / "cats.json", "w", encoding="utf-8"), indent=1)

print("\nimages per split:\n", meta["split"].value_counts().to_string())
print(f"categories kept: {len(cats)}  (labels per image: {Y.sum(1).mean():.1f})")
print("seed categories missing from vocabulary:", [s for s in SEEDS if s not in ci])
print("\nTOP 30 categories - read them! add junk words to common.py if needed:")
print(pd.Series(Y.sum(0), index=cats).sort_values(ascending=False).head(30).to_string())
