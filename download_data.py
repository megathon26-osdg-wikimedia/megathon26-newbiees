"""
download_data.py  (Phase 1)
 
Downloads small (256 px) open-licensed images from Wikimedia Commons for a list
of seed categories, together with each image's REAL Commons categories, title,
description, upload date and licence.
 
Run a quick test first:   python download_data.py test
Then the full download:   python download_data.py
 
Output (inside the folder "data"):
    data/images/<pageid>.jpg|png   the images
    data/metadata.csv              one row per image
    data/graph_edges.json          child -> parent category pairs found
"""
 
import csv
import json
import re
import sys
import time
from pathlib import Path
 
import requests
 
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CategorySuggestionStudentProject/0.1 (college project)"}
 
OUT = Path("data")
IMG_DIR = OUT / "images"
 
N_PER_CAT = 100  # images per seed category (full run)
 
# Mix of objects, scenes, events and animals. Names must match Commons exactly.
SEEDS = [
    # monuments
    "Taj Mahal", "Eiffel Tower", "Charminar", "Red Fort",
    "Statue of Liberty", "Colosseum",
    # animals
    "Panthera tigris", "Elephas maximus", "Pavo cristatus",
    "Panthera leo", "Giraffa camelopardalis",
    # scenes
    "Sunsets", "Waterfalls", "Beaches", "Glaciers", "Deserts",
    # events
    "Diwali", "Holi", "Ganesh Chaturthi", "Christmas markets",
    # objects
    "Airbus A380", "Boeing 747", "Bicycles", "Steam locomotives",
]
 
if len(sys.argv) > 1 and sys.argv[1] == "test":
    SEEDS = SEEDS[:2]
    N_PER_CAT = 10
    print("TEST MODE: 2 categories, 10 images each\n")
 
 
def api(params):
    """Call the Commons API with retries."""
    p = {"action": "query", "format": "json", "formatversion": "2"}
    p.update(params)
    time.sleep(1.0)  # pause before every call so the server is not overloaded
    r = None
    for attempt in range(8):
        try:
            r = requests.get(API, params=p, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            wait = min(15 * (attempt + 1), 90)
            print(f"  no internet connection? waiting {wait}s and trying again...")
            time.sleep(wait)
            r = None
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:  # "too many requests": wait longer and retry
            wait = min(15 * (attempt + 1), 90)
            print(f"  server says slow down, waiting {wait}s...")
            time.sleep(wait)
        else:
            time.sleep(2 * (attempt + 1))
    if r is None:
        raise RuntimeError("could not connect to the internet")
    r.raise_for_status()
 
 
def fetch_files(cat_title, limit):
    """Files directly inside one category, with their categories and info."""
    pages = {}
    cont = {}
    while len(pages) < limit:
        params = {
            "generator": "categorymembers",
            "gcmtitle": cat_title,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo|categories",
            "iiprop": "url|timestamp|extmetadata|mime|size",
            "iiurlwidth": "256",
            "iiextmetadatafilter": "LicenseShortName|ImageDescription",
            "cllimit": "max",
            "clshow": "!hidden",  # skip hidden maintenance categories
        }
        params.update(cont)
        data = api(params)
        for pg in data.get("query", {}).get("pages", []):
            e = pages.setdefault(
                pg["pageid"], {"title": pg["title"], "cats": set(), "info": None}
            )
            if "imageinfo" in pg:
                e["info"] = pg["imageinfo"][0]
            for c in pg.get("categories", []):
                e["cats"].add(c["title"])
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages
 
 
def collect(seed):
    """Files of a seed category; if too few, also use its subcategories."""
    cat = "Category:" + seed
    pages = fetch_files(cat, N_PER_CAT)
    edges = []
    if len(pages) < N_PER_CAT:
        data = api({"list": "categorymembers", "cmtitle": cat,
                    "cmtype": "subcat", "cmlimit": "10"})
        subs = data.get("query", {}).get("categorymembers", [])
        for s in subs:
            if len(pages) >= N_PER_CAT:
                break
            edges.append((s["title"], cat))
            for k, v in fetch_files(s["title"], N_PER_CAT - len(pages)).items():
                pages.setdefault(k, v)
    return pages, edges
 
 
def licence_ok(lic):
    low = lic.lower()
    return "cc" in low or "public domain" in low or low.startswith("pd")
 
 
def clean(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()[:300]
 
 
def download(url, path):
    if path.exists():
        return True
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            path.write_bytes(r.content)
            return True
        time.sleep(3 * (attempt + 1))
    return False
 
 
def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    all_edges = []
 
    for seed in SEEDS:
        try:
            pages, edges = collect(seed)
        except Exception as err:
            print(f"{seed}: FAILED ({err}) - skipped, run the script again later")
            continue
        all_edges += edges
        kept = 0
        for pid, e in list(pages.items())[:N_PER_CAT]:
            info = e["info"]
            if not info or info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = info.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not licence_ok(lic):
                continue
            if pid in rows:                      # image already seen in another seed
                rows[pid]["seeds"].add(seed)
                continue
            ext = ".png" if info["mime"] == "image/png" else ".jpg"
            path = IMG_DIR / f"{pid}{ext}"
            url = info.get("thumburl") or info["url"]
            if not download(url, path):
                continue
            time.sleep(0.2)                      # be polite to the server
            rows[pid] = {
                "pageid": pid,
                "title": e["title"],
                "timestamp": info.get("timestamp", ""),
                "license": lic,
                "seeds": {seed},
                "categories": "|".join(
                    sorted(c.replace("Category:", "") for c in e["cats"])),
                "description": clean(meta.get("ImageDescription", {}).get("value", "")),
                "image_path": str(path),
            }
            kept += 1
        print(f"{seed}: found {len(pages)}, kept {kept}")
 
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["pageid", "title", "timestamp", "license", "seeds",
                "categories", "description", "image_path"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows.values():
            r["seeds"] = "|".join(sorted(r["seeds"]))
            w.writerow(r)
 
    with open(OUT / "graph_edges.json", "w", encoding="utf-8") as f:
        json.dump(all_edges, f, indent=2)
 
    print(f"\nDone. {len(rows)} images saved in {IMG_DIR}")
    print(f"Metadata: {OUT / 'metadata.csv'}")
 
 
if __name__ == "__main__":
    main()
    