"""throughput.py - images per second when suggestions are made in batches (CPU or GPU).
Run:  python throughput.py       (latency.py still gives the one-image median / p95)"""
import time

import joblib
import numpy as np
from PIL import Image

from clip_utils import DEVICE, embed_pil
from common import DATA, load_clean
from pipeline import COMP

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
sug.set_index(Z, Y, texts)
n = min(64, len(df))
imgs = [Image.open(p).convert("RGB") for p in df["image_path"].values[:n]]
tx = texts[:n]
embed_pil(imgs[:2])                                    # warm-up
print("device:", DEVICE, "| index:", len(Z), "images")
for bs in (1, 8, 32):
    t0 = time.time()
    done = 0
    for i in range(0, n, bs):
        z = embed_pil(imgs[i:i + bs])
        S = sug.fuse(sug.components(z, tx[i:i + bs]))
        np.argsort(-S, axis=1)[:, :10]
        done += len(z)
    dt = time.time() - t0
    print(f"batch {bs:2d}: {done / dt:6.1f} images/s  ({dt / done * 1000:.0f} ms per image)")
