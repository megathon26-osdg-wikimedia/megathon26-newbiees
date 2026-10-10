"""scorers.py - the four scorers, fusion and the category graph (numpy + scikit-learn only)."""
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------- scorer 1: zero-shot (image vs category NAME) ----------
def softmax(a, axis=1):
    a = a - a.max(axis=axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=axis, keepdims=True)


def zeroshot_scores(Zq, T, scale=100.0):
    return softmax(scale * (Zq @ T.T))


# ---------- scorer 2: kNN on image embeddings ----------
def knn_scores(Zq, Zref, Yref, K=20, tau=0.05, bs=256):
    K = min(K, len(Zref))
    out = np.zeros((len(Zq), Yref.shape[1]), dtype="float32")
    for s in range(0, len(Zq), bs):
        sims = Zq[s:s + bs] @ Zref.T
        idx = np.argpartition(-sims, K - 1, axis=1)[:, :K]
        top = np.take_along_axis(sims, idx, axis=1)
        w = np.exp((top - top.max(axis=1, keepdims=True)) / tau)
        w /= w.sum(axis=1, keepdims=True)
        out[s:s + bs] = (w[:, :, None] * Yref[idx]).sum(axis=1)
    return out


# ---------- scorer 3: trained head with asymmetric loss ----------
FEAT_SCALE = 10.0            # embeddings have length 1; scaling helps training


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def asl_loss_grad(p, Y, clip=0.05, gamma_neg=4.0, eps=1e-8):
    """Asymmetric loss and its gradient w.r.t. the logits (gamma_pos = 0)."""
    q = np.maximum(p - clip, 0.0)                          # margin: forgives tiny mistakes
    loss = -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - q + eps) * q ** gamma_neg).sum()
    g_pos = -Y * (1 - p)
    d_q = (1 - Y) * (q ** gamma_neg / (1 - q + eps)
                     - gamma_neg * q ** (gamma_neg - 1) * np.log(1 - q + eps))
    g_neg = d_q * np.where(p > clip, p * (1 - p), 0.0)
    return loss / len(Y), (g_pos + g_neg) / len(Y)


def train_head(Z, Y, epochs=300, lr=0.01, wd=1e-4, seed=0, verbose=True):
    """Linear layer + sigmoid, trained full-batch with Adam. Returns (W, b)."""
    X = Z * FEAT_SCALE
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (Y.shape[1], X.shape[1])).astype("float32")
    b = np.full(Y.shape[1], -4.0, dtype="float32")        # start with 'probably not'
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    for t in range(1, epochs + 1):
        p = _sigmoid(X @ W.T + b)
        loss, gz = asl_loss_grad(p, Y)
        gW, gb = gz.T @ X + wd * W, gz.sum(axis=0)
        for g, m, v, par in ((gW, mW, vW, W), (gb, mb, vb, b)):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            par -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if verbose and (t == 1 or t % 50 == 0):
            print(f"  head epoch {t:3d}  loss {loss:.3f}")
    return W, b


def head_scores(Zq, W, b):
    return _sigmoid((Zq * FEAT_SCALE) @ W.T + b)


# ---------- scorer 4: title/description text (TF-IDF neighbours) ----------
def text_knn_scores(Xq, Xref, Yref, K=20):
    sims = (Xq @ Xref.T).toarray()
    K = min(K, sims.shape[1])
    idx = np.argpartition(-sims, K - 1, axis=1)[:, :K]
    top = np.take_along_axis(sims, idx, axis=1)
    w = top / np.maximum(top.sum(axis=1, keepdims=True), 1e-9)
    return (w[:, :, None] * Yref[idx]).sum(axis=1).astype("float32")


# ---------- fusion: logistic regression over the scorers ----------
def candidate_mask(score_list, topn=30):
    """True where a category is in the top-n of ANY scorer for that image."""
    mask = np.zeros(score_list[0].shape, dtype=bool)
    for S in score_list:
        n = min(topn, S.shape[1])
        np.put_along_axis(mask, np.argpartition(-S, n - 1, axis=1)[:, :n], True, axis=1)
    return mask


def fit_fusion(score_list, Y):
    """Fit on VALIDATION scores only - never on test."""
    m = candidate_mask(score_list)
    X = np.stack([S[m] for S in score_list], axis=1)
    return LogisticRegression(max_iter=2000, C=10.0).fit(X, Y[m].astype(int))


def apply_fusion(clf, score_list):
    m = candidate_mask(score_list)
    out = np.zeros(score_list[0].shape, dtype="float32")
    out[m] = clf.predict_proba(np.stack([S[m] for S in score_list], axis=1))[:, 1]
    return out


# ---------- category graph: prefer the most specific category ----------
class Graph:
    def __init__(self, edges):
        self.parents = {}
        for child, parent in edges:
            c = child.replace("Category:", "", 1)
            p = parent.replace("Category:", "", 1)
            self.parents.setdefault(c, set()).add(p)

    def ancestors(self, c, depth=4):
        seen, frontier = set(), {c}
        for _ in range(depth):                      # 'seen' also protects against cycles
            frontier = set().union(*[self.parents.get(x, set()) for x in frontier]) - seen - {c}
            if not frontier:
                break
            seen |= frontier
        return seen

    def ancestor_matrix(self, cats):
        """A[a, c] = 1 when category a is an ancestor of category c."""
        ix = {c: i for i, c in enumerate(cats)}
        A = np.zeros((len(cats), len(cats)), dtype="float32")
        for c in cats:
            for a in self.ancestors(c):
                if a in ix:
                    A[ix[a], ix[c]] = 1
        return A


def suppress_ancestors(order, A):
    """Drop a category if a more specific (descendant) one is also in the list."""
    sub = A[np.ix_(order, order)]                   # sub[i, j] = 1: order[i] is ancestor of order[j]
    drop = (sub * (1 - sub.T)).any(axis=1)          # (1 - sub.T) ignores 2-cycles
    return order[~drop]
