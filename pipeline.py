"""pipeline.py - the Suggester object: four scorers + fusion + category graph."""
import numpy as np
import scipy.sparse as sp

from scorers import (apply_fusion, head_scores, knn_scores, suppress_ancestors,
                     text_knn_scores, zeroshot_scores)

COMP = ["knn", "head", "zeroshot", "text"]     # fixed order of the fusion features


class Suggester:
    def __init__(self, cats, T, W, b, tfidf, A):
        self.cats, self.T, self.W, self.b, self.tfidf, self.A = cats, T, W, b, tfidf, A
        self.clf = None

    def set_index(self, Z, Y, texts):
        self.Zref, self.Yref = Z, Y
        self.Xref = self.tfidf.transform(texts)

    def add_to_index(self, z, y, text):
        self.Zref = np.vstack([self.Zref, z])
        self.Yref = np.vstack([self.Yref, y])
        self.Xref = sp.vstack([self.Xref, self.tfidf.transform([text])]).tocsr()

    def components(self, Zq, texts):
        return {"knn": knn_scores(Zq, self.Zref, self.Yref),
                "head": head_scores(Zq, self.W, self.b),
                "zeroshot": zeroshot_scores(Zq, self.T),
                "text": text_knn_scores(self.tfidf.transform(texts), self.Xref, self.Yref)}

    def fuse(self, comp):
        return apply_fusion(self.clf, [comp[k] for k in COMP])

    def suggest(self, z, text="", k=10):
        """z: (1, 512) embedding. Returns [(category, score), ...] most specific first."""
        S = self.fuse(self.components(z, [text]))[0]
        order = suppress_ancestors(np.argsort(-S)[:k], self.A)
        return [(self.cats[i], float(S[i])) for i in order]
