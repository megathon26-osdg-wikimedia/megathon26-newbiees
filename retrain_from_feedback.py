"""retrain_from_feedback.py - fold accepted reviews into training and retrain everything.

Run:  python retrain_from_feedback.py
What it does
  1. reads data/feedback_accepted.csv (written by feedback_store.log_review in the demo),
  2. embeds those images and adds them to the TRAIN part only (val and test stay untouched),
  3. retrains the head, refits TF-IDF and the fusion exactly like train.py,
  4. prints the acceptance rate of the logged reviews and test P@1 before vs after,
  5. keeps the old model as data/suggester_before_retrain.joblib.
Acceptance rate = accepted suggestions / suggestions shown. Falling rate = retrain/drift signal."""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from clip_utils import embed_paths, embed_texts
from common import DATA, load_clean, text_of
from metrics import prf, subset
from pipeline import COMP, Suggester
from scorers import Graph, fit_fusion, train_head

df, cats, Z, Y, idx, texts = load_clean()
tr, va, te = idx["train"], idx["val"], idx["test"]
ci = {c: i for i, c in enumerate(cats)}

# --- acceptance rate from the log
if (DATA / "feedback_log.csv").exists():
    log = pd.read_csv(DATA / "feedback_log.csv")
    rate = log["n_accepted"].sum() / max(log["n_shown"].sum(), 1)
    print(f"reviews logged: {len(log)} | acceptance rate: {rate:.2f} "
          f"({int(log['n_accepted'].sum())} of {int(log['n_shown'].sum())} suggestions)")
    if len(log) >= 20:
        half = len(log) // 2
        a = log.iloc[:half]; b = log.iloc[half:]
        print(f"  first half {a['n_accepted'].sum() / max(a['n_shown'].sum(), 1):.2f} -> "
              f"second half {b['n_accepted'].sum() / max(b['n_shown'].sum(), 1):.2f}")
else:
    print("no data/feedback_log.csv yet (use the demo and Save my review first)")

# --- accepted feedback images
fb_path = DATA / "feedback_accepted.csv"
if not fb_path.exists():
    raise SystemExit("no data/feedback_accepted.csv - nothing to retrain with")
fb = pd.read_csv(fb_path, keep_default_na=False)
fb["labels"] = fb["labels"].apply(lambda s: [c for c in s.split("|") if c in ci])
fb = fb[fb["labels"].str.len() > 0].reset_index(drop=True)
print("accepted feedback images usable:", len(fb))
if len(fb) == 0:
    raise SystemExit("no usable feedback rows")
Zfb = embed_paths(fb["image_path"].tolist())
Yfb = np.zeros((len(fb), len(cats)), dtype="float32")
for i, cs in enumerate(fb["labels"]):
    for c in cs:
        Yfb[i, ci[c]] = 1
tfb = [text_of(t, "") for t in fb["title"]]

old = joblib.load(DATA / "suggester.joblib")
old.set_index(Z[tr], Y[tr], [texts[i] for i in tr])
Yte, te_texts = Y[te], [texts[i] for i in te]
p_old = prf(*subset(old.fuse(old.components(Z[te], te_texts)), Yte), 1)[0]

# --- retrain like train.py, with the extra TRAIN rows
Ztr = np.vstack([Z[tr], Zfb]); Ytr = np.vstack([Y[tr], Yfb])
ttr = [texts[i] for i in tr] + tfb
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                        stop_words="english").fit(ttr)
T = embed_texts([f"a photo of {c}" for c in cats])
W, b = train_head(Ztr, Ytr)
edges = []
for name in ("graph_edges_full.json", "graph_edges.json"):
    if (DATA / name).exists():
        edges += json.load(open(DATA / name, encoding="utf-8"))
A = Graph(edges).ancestor_matrix(cats)
sug = Suggester(cats, T, W, b, tfidf, A)
sug.set_index(Ztr, Ytr, ttr)
comp_va = sug.components(Z[va], [texts[i] for i in va])
sug.clf = fit_fusion([comp_va[k] for k in COMP], Y[va])
p_new = prf(*subset(sug.fuse(sug.components(Z[te], te_texts)), Yte), 1)[0]
print(f"test P@1 before {p_old:.3f} -> after {p_new:.3f} (test images were not changed)")

shutil.copy(DATA / "suggester.joblib", DATA / "suggester_before_retrain.joblib")
joblib.dump(sug, DATA / "suggester.joblib")
print("saved data/suggester.joblib (old model kept as suggester_before_retrain.joblib)")
