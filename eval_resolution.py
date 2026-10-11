"""eval_resolution.py - results by image RESOLUTION band (simulated), TEST split.
Each test image is shrunk by a factor (1 = original 256 px, 8 = about 32 px) and stretched
back, then embedded again. Needs CLIP; about 1 minute per band on CPU.
Run:  python eval_resolution.py      Output: data/results_resolution.csv"""
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from clip_utils import embed_paths
from common import DATA, load_clean
from metrics import prf, subset

FACTORS = [1, 2, 4, 6, 8, 12]


def shrink(im, f):
    if f == 1:
        return im
    w, h = im.size
    return im.resize((max(8, w // f), max(8, h // f)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
te = idx["test"]
Yte, te_texts = Y[te], [texts[i] for i in te]
paths = df["image_path"].values[te].tolist()

rows = []
for f in FACTORS:
    Zd = embed_paths(paths, degrade_fn=lambda im, f=f: shrink(im, f))
    comp = sug.components(Zd, te_texts)
    comp["FUSED"] = sug.fuse(comp)
    row = {"shrink factor": f, "approx. size (px)": max(8, 256 // f),
           "mean cosine to clean": round(float((Zd * Z[te]).sum(axis=1).mean()), 3)}
    for name, S in comp.items():
        row[f"P@1 {name}"] = round(prf(*subset(S, Yte), 1)[0], 3)
    row["R@5 FUSED"] = round(prf(*subset(comp["FUSED"], Yte), 5)[1], 3)
    rows.append(row)
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv(DATA / "results_resolution.csv", index=False)
print("\nSaved data/results_resolution.csv")
