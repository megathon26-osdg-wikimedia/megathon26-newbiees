"""fetch_graph.py - real parents (2 levels up) of every category in metadata.csv.
Run after download_data.py. Output: data/graph_edges_full.json"""
import json
import time

import pandas as pd
import requests

from common import DATA, is_junk

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CategorySuggestionStudentProject/0.1 (college project)"}


def get_json(params):
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(min(15 * (attempt + 1), 90))
            continue
        if r.status_code == 200:
            return r.json()
        time.sleep(min(15 * (attempt + 1), 90) if r.status_code == 429 else 2 * (attempt + 1))
    raise RuntimeError("Commons API kept failing")


def parents_of(titles):
    """[(child, parent), ...] for up to 50 'Category:...' titles."""
    edges, extra = [], {}
    while True:
        params = {"action": "query", "format": "json", "formatversion": "2",
                  "titles": "|".join(titles), "prop": "categories",
                  "cllimit": "max", "clshow": "!hidden"}
        params.update(extra)
        data = get_json(params)
        for pg in data.get("query", {}).get("pages", []):
            for p in pg.get("categories", []):
                edges.append((pg["title"], p["title"]))
        if "continue" not in data:
            return edges
        extra = data["continue"]
        time.sleep(1)


meta = pd.read_csv(DATA / "metadata.csv", keep_default_na=False)
names = {c for s in meta["categories"] for c in s.split("|") if c and not is_junk(c)}
frontier = ["Category:" + c for c in sorted(names)]
seen, edges = set(frontier), []
for level in range(2):                       # 2 levels of ancestors
    nxt = set()
    for i in range(0, len(frontier), 50):
        for child, parent in parents_of(frontier[i:i + 50]):
            edges.append((child, parent))
            if parent not in seen:
                nxt.add(parent)
        time.sleep(1)
    seen |= nxt
    frontier = sorted(nxt)
    print(f"level {level + 1}: {len(edges)} edges so far")
json.dump(edges, open(DATA / "graph_edges_full.json", "w", encoding="utf-8"), indent=1)
print("Saved", len(edges), "edges")
