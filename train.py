"""train.py - train the head, fit TF-IDF, fit fusion on VALIDATION, save everything.
Output: data/suggester.joblib"""
import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from clip_utils import embed_texts
from common import DATA, load_clean
from pipeline import COMP, Suggester
from scorers import Graph, fit_fusion, train_head

df, cats, Z, Y, idx, texts = load_clean()
tr, va = idx["train"], idx["val"]
print(f"train {len(tr)} | val {len(va)} | test {len(idx['test'])} | categories {len(cats)}")

tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                        stop_words="english").fit([texts[i] for i in tr])
T = embed_texts([f"a photo of {c}" for c in cats])
print("training the head...")
W, b = train_head(Z[tr], Y[tr])

edges = []
for name in ("graph_edges_full.json", "graph_edges.json"):
    if (DATA / name).exists():
        edges += json.load(open(DATA / name, encoding="utf-8"))
A = Graph(edges).ancestor_matrix(cats)
print("graph edges:", len(edges), "| ancestor links inside vocabulary:", int(A.sum()))

sug = Suggester(cats, T, W, b, tfidf, A)
sug.set_index(Z[tr], Y[tr], [texts[i] for i in tr])
comp_va = sug.components(Z[va], [texts[i] for i in va])
sug.clf = fit_fusion([comp_va[k] for k in COMP], Y[va])
print("fusion trust weights", dict(zip(COMP, sug.clf.coef_[0].round(2))))
joblib.dump(sug, DATA / "suggester.joblib")
print("saved data/suggester.joblib")
