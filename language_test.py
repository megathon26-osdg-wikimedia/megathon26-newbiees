"""language_test.py - how does the title/description scorer behave on NON-English titles?

Make data/lang_test.csv with columns:  image_path,title,language,labels
  image_path  path to an image of a category we know
  title       a file title written in that language (e.g. a Hindi or French title)
  language    e.g. en, hi, fr   (include some 'en' rows for comparison)
  labels      the true category name(s) from data/cats.json, joined by |
Run:  python language_test.py        Output: data/results_language.csv

Per language it reports
  title words known  = share of titles with at least one word the TF-IDF vocabulary knows
  P@1 text           = top-1 of the text scorer alone is a true label
  P@1 fused, title   = fused system with the title
  P@1 fused, no title= fused system with an empty title (image only)
If 'title words known' is near 0 for a language, the text scorer simply cannot help there."""
import joblib
import numpy as np
import pandas as pd

from clip_utils import embed_paths
from common import DATA, load_clean, text_of

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
sug.set_index(Z[idx["train"]], Y[idx["train"]], [texts[i] for i in idx["train"]])
ci = {c: i for i, c in enumerate(cats)}

t = pd.read_csv(DATA / "lang_test.csv", keep_default_na=False)
t["labels"] = t["labels"].apply(lambda s: [c for c in s.split("|") if c in ci])
t = t[t["labels"].str.len() > 0].reset_index(drop=True)
if len(t) == 0:
    raise SystemExit("no rows with labels that exist in data/cats.json")
Yq = np.zeros((len(t), len(cats)), dtype="float32")
for i, cs in enumerate(t["labels"]):
    for c in cs:
        Yq[i, ci[c]] = 1
Zq = embed_paths(t["image_path"].tolist())
tx = [text_of(x, "") for x in t["title"]]

comp_t = sug.components(Zq, tx)
comp_n = sug.components(Zq, [""] * len(t))
S_fused_t, S_fused_n, S_text = sug.fuse(comp_t), sug.fuse(comp_n), comp_t["text"]
known = sug.tfidf.transform(tx).getnnz(axis=1) > 0


def p1(S):
    return (Yq[np.arange(len(t)), np.argmax(S, axis=1)])


rows = []
for lang in sorted(t["language"].unique()):
    m = (t["language"] == lang).values
    rows.append({"language": lang, "images": int(m.sum()),
                 "title words known": round(float(known[m].mean()), 2),
                 "P@1 text": round(float(p1(S_text)[m].mean()), 2),
                 "P@1 fused, title": round(float(p1(S_fused_t)[m].mean()), 2),
                 "P@1 fused, no title": round(float(p1(S_fused_n)[m].mean()), 2)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(res.to_string(index=False))
res.to_csv(DATA / "results_language.csv", index=False)
print("\nSaved data/results_language.csv  (few images per language = rough numbers, say so)")
