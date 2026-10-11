"""feedback_store.py - persist reviewer feedback so the next retrain can use it.

In demo_app.py, inside save_review(), call log_review(...) BEFORE the early return:

    from feedback_store import log_review
    ...
    path = ""                      # set to the saved image path when something was accepted
    (after you save the image:  path = f"data/feedback/{len(paths)}.jpg")
    log_review(path, LAST["title"], selected, LAST["shown"])

Files written:
  data/feedback_log.csv       every review: time, suggestions shown, suggestions accepted
  data/feedback_accepted.csv  image_path, title, labels   (only reviews with >= 1 accepted)"""
import csv
import time

from common import DATA


def log_review(image_path, title, accepted, shown):
    new_log = not (DATA / "feedback_log.csv").exists()
    with open(DATA / "feedback_log.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "n_shown", "n_accepted"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), len(shown), len(accepted)])
    if accepted and image_path:
        new = not (DATA / "feedback_accepted.csv").exists()
        with open(DATA / "feedback_accepted.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["image_path", "title", "labels"])
            w.writerow([image_path, title or "", "|".join(accepted)])


def load_accepted(cats):
    """Accepted reviews from earlier sessions -> (image paths, embeddings, Y rows, titles, label lists).
    Called once when the demo starts, so nothing accepted is ever forgotten."""
    import numpy as np
    import pandas as pd

    path = DATA / "feedback_accepted.csv"
    if not path.exists():
        return [], np.zeros((0, 512), dtype="float32"), np.zeros((0, len(cats)), dtype="float32"), [], []
    from pathlib import Path

    from clip_utils import embed_paths
    ci = {c: i for i, c in enumerate(cats)}
    fb = pd.read_csv(path, keep_default_na=False)
    keep = [i for i, r in fb.iterrows() if Path(r["image_path"]).exists()]
    fb = fb.loc[keep].reset_index(drop=True)
    labs = [[c for c in str(s).split("|") if c in ci] for s in fb["labels"]]
    ok = [i for i, l in enumerate(labs) if l]
    fb, labs = fb.loc[ok].reset_index(drop=True), [labs[i] for i in ok]
    if len(fb) == 0:
        return [], np.zeros((0, 512), dtype="float32"), np.zeros((0, len(cats)), dtype="float32"), [], []
    Z = embed_paths(fb["image_path"].tolist())
    Y = np.zeros((len(fb), len(cats)), dtype="float32")
    for i, l in enumerate(labs):
        for c in l:
            Y[i, ci[c]] = 1
    return fb["image_path"].tolist(), Z, Y, fb["title"].tolist(), labs
