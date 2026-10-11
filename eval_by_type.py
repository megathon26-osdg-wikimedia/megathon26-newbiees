"""eval_by_type.py - results per category TYPE (object / scene / event / abstract), TEST split.
Run:  python eval_by_type.py        Output: data/results_by_type.csv

hit@1  = share of test images (that have a true label of this type) whose top suggestion
         is one of their true labels of this type.
R@5    = recall of that type's true labels inside the top 5 suggestions.
Cells with fewer than 20 test images are marked '*' - do not over-read them."""
import joblib
import numpy as np
import pandas as pd

from category_types import category_types
from common import DATA, load_clean
from metrics import tier_recall

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
te, tr = idx["test"], idx["train"]
Yte, te_texts = Y[te], [texts[i] for i in te]
comp = sug.components(Z[te], te_texts)
comp["FUSED"] = sug.fuse(comp)

ctype = category_types(df, Y, tr)
rows = []
for t in sorted(set(ctype)):
    cols = np.where(ctype == t)[0]
    has = Yte[:, cols].sum(axis=1) > 0
    n_img = int(has.sum())
    for name, S in comp.items():
        if n_img == 0:
            continue
        top1 = np.argmax(S[has], axis=1)
        hit1 = float((Yte[has][np.arange(n_img), top1] * np.isin(top1, cols)).mean())
        rows.append({"type": t, "scorer": name, "categories": len(cols),
                     "test images": n_img,
                     "hit@1": round(hit1, 3),
                     "R@5": round(tier_recall(S, Yte, 5, cols), 3),
                     "note": "*" if n_img < 20 else ""})
res = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(res[res["scorer"] == "FUSED"].to_string(index=False))
print("\nAll scorers:\n", res.to_string(index=False))
res.to_csv(DATA / "results_by_type.csv", index=False)
print("\nSaved data/results_by_type.csv")
