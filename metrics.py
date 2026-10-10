"""metrics.py - P@k, R@k, F1@k, mAP, head/torso/tail recall, hierarchy-aware scores."""
import numpy as np
import pandas as pd

from scorers import suppress_ancestors


def subset(S, Y, cols=None):
    """Optionally keep only some category columns; keep images with >= 1 true label."""
    if cols is not None:
        S, Y = S[:, cols], Y[:, cols]
    m = Y.sum(axis=1) > 0
    return S[m], Y[m]


def prf(S, Y, k):
    top = np.argsort(-S, axis=1)[:, :k]
    hits = np.take_along_axis(Y, top, axis=1).sum(axis=1)
    P, R = hits / k, hits / np.maximum(Y.sum(axis=1), 1)
    F = 2 * P * R / np.maximum(P + R, 1e-9)
    return float(P.mean()), float(R.mean()), float(F.mean())


def mean_ap(S, Y):
    rel = np.take_along_axis(Y, np.argsort(-S, axis=1), axis=1)
    prec = np.cumsum(rel, axis=1) / np.arange(1, S.shape[1] + 1)
    return float(((prec * rel).sum(axis=1) / np.maximum(rel.sum(axis=1), 1)).mean())


def report(name, S, Y, ks=(1, 3, 5, 10)):
    ap = round(mean_ap(S, Y), 3)
    rows = []
    for k in ks:
        p, r, f = prf(S, Y, k)
        rows.append({"scorer": name, "k": k, "P@k": round(p, 3), "R@k": round(r, 3),
                     "F1@k": round(f, 3), "mAP": ap})
    return pd.DataFrame(rows)


def tier_recall(S, Y, k, cols):
    top = np.argsort(-S, axis=1)[:, :k]
    picked = np.zeros_like(Y)
    np.put_along_axis(picked, top, 1, axis=1)
    hit = (picked[:, cols] * Y[:, cols]).sum(axis=1)
    pos = Y[:, cols].sum(axis=1)
    m = pos > 0
    return float((hit[m] / pos[m]).mean()) if m.any() else float("nan")


def lenient_precision(S, Y, A, k=5):
    """A suggestion also counts as fine if it is an ANCESTOR of a true label
    (editors leave parents out on purpose)."""
    L = np.clip(Y + Y @ A.T, 0, 1)
    top = np.argsort(-S, axis=1)[:, :k]
    return float(np.take_along_axis(L, top, axis=1).sum(axis=1).mean() / k)


def cleaned_precision(S, Y, A, k=5):
    """Precision and mean length of the lists AFTER ancestor suppression."""
    hits = shown = 0
    for i in range(len(S)):
        kept = suppress_ancestors(np.argsort(-S[i])[:k], A)
        hits += Y[i, kept].sum()
        shown += len(kept)
    return hits / max(shown, 1), shown / len(S)
