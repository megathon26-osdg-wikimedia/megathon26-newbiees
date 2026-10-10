"""common.py - shared paths, constants and small helpers used by every script."""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")

try:                                   # the seed list lives in download_data.py
    from download_data import SEEDS
except Exception:
    SEEDS = []

# Categories that describe the upload PROCESS, not the picture. Extend this list
# after reading the "top 30 categories" printout of prepare_labels.py.
JUNK_WORDS = [
    "uploaded", "self-published", "files ", "media ", "pages ", "license",
    "gfdl", "photographs taken", "taken with", "taken on", "quality images",
    "featured", "valued image", "images from", "images by", "photographs by",
    "pictures by", "by camera", "by year", "by country", "by photographer",
    "by date", "wikimedia", "wikipedia", "commons", "needing", "unidentified",
    "watermark", "uploads", "scans", "flickr", "pictures of the day",
    "wiki loves", "cc-", "pd-",
]
DATE_RE = re.compile(r"^(\d{4}|\w+ \d{4}|\d{1,2} \w+ \d{4})$")


def is_junk(name):
    low = name.lower()
    return bool(DATE_RE.match(name)) or any(w in low for w in JUNK_WORDS)


def clean_title(title):
    t = title.replace("File:", "")
    t = re.sub(r"\.(jpe?g|png)$", "", t, flags=re.I)
    return re.sub(r"[_\-]+", " ", t).strip()


def text_of(title, description):
    """Text the title/description scorer sees for one image."""
    return clean_title(title) + " " + html.unescape(description or "")


def load_clean():
    """Everything prepare_labels.py produced, with rows aligned."""
    df = pd.read_csv(DATA / "clean.csv", keep_default_na=False)
    df["labels"] = df["labels"].apply(lambda s: s.split("|"))
    cats = json.load(open(DATA / "cats.json", encoding="utf-8"))
    Z = np.load(DATA / "Zc.npy")
    Y = np.load(DATA / "Y.npy")
    idx = {s: np.where(df["split"].values == s)[0] for s in ("train", "val", "test")}
    texts = [text_of(t, d) for t, d in zip(df["title"], df["description"])]
    return df, cats, Z, Y, idx, texts
