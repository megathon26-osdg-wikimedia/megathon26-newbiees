"""latency.py - how long does ONE suggestion take, and how big is the index?"""
import time

import joblib
import numpy as np
from PIL import Image

from clip_utils import embed_pil
from common import DATA, load_clean

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
sug.set_index(Z, Y, texts)                              # full index, like the demo
img = Image.open(df["image_path"].iloc[0]).convert("RGB")
embed_pil([img])                                        # warm-up
times = []
for _ in range(30):
    t0 = time.time()
    sug.suggest(embed_pil([img]), texts[0], k=10)
    times.append((time.time() - t0) * 1000)
print(f"median {np.percentile(times, 50):.1f} ms | p95 {np.percentile(times, 95):.1f} ms")
print(f"index: {len(Z)} images, {Z.nbytes / 1e6:.1f} MB of embeddings")
