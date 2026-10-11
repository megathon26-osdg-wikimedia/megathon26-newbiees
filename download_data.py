"""
download_data.py  (Phase 1, version 3: events, abstract concepts, small categories,
                   and the seed category is added to images taken from subcategories)

Check that every seed name exists (fast, downloads nothing):  python download_data.py check
Quick test (2 categories, 10 images each):                    python download_data.py test
Full download:                                                python download_data.py

Images already saved are skipped, so re-running after the old download is safe.
metadata.csv is written even if the run is interrupted (Ctrl+C).
Output (inside the folder "data"): images/, metadata.csv, graph_edges.json
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

N_PER_CAT = 100   # images per normal seed category
N_SMALL = 40      # images per small seed category (gives a real head/torso/tail)

# Names must match Commons exactly. Run "python download_data.py check" to verify.
SEEDS = [
    # monuments
    "Taj Mahal", "Eiffel Tower", "Charminar", "Red Fort",
    "Statue of Liberty", "Colosseum",
    # animals
    "Panthera tigris", "Elephas maximus", "Pavo cristatus",
    "Panthera leo", "Giraffa camelopardalis",
    # scenes
    "Sunsets", "Waterfalls", "Beaches", "Glaciers", "Deserts",
    # events (Diwali name is a guess: check will tell)
    "Holi", "Ganesh Chaturthi", "Christmas markets",
    "Durga Puja", "Oktoberfest", "Fireworks", "Carnivals", "Marathons",
    # objects
    "Airbus A380", "Boeing 747", "Bicycles", "Diesel locomotives",
    # abstract concepts and patterns
    "Rangoli", "Mandalas", "Fractals", "Graffiti", "Silhouettes", "Calligraphy",
    # small categories (N_SMALL images each)
    "Hot air balloons", "Windmills", "Rickshaws", "Lighthouses",
    "Double-decker buses", "Chess", "Tea plantations",
]
SMALL = {"Hot air balloons", "Windmills", "Rickshaws", "Lighthouses",
         "Double-decker buses", "Chess", "Tea plantations",
         "Silhouettes", "Calligraphy"}

# Alternative names to try with "check" if a seed above prints MISSING or 0 files.
DIWALI_ALTERNATIVES = ["Diwali festival", "Diwali celebrations", "Deepavali",
                       "Diwali decorations", "Diwali lamps"]


def api(params):
    """Call the Commons API with retries."""
    p = {"action": "query", "format": "json", "formatversion": "2"}
    p.update(params)
    time.sleep(1.0)
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
        if r.status_code == 429:
            wait = min(15 * (attempt + 1), 90)
            print(f"  server says slow down, waiting {wait}s...")
            time.sleep(wait)
        else:
            time.sleep(2 * (attempt + 1))
    if r is None:
        raise RuntimeError("could not connect to the internet")
    r.raise_for_status()


def check(names):
    """Print how many files and subcategories each category has."""
    for name in names:
        data = api({"titles": "Category:" + name, "prop": "categoryinfo"})
        pg = data["query"]["pages"][0]
        if pg.get("missing"):
            print(f"MISSING  {name}")
            continue
        ci = pg.get("categoryinfo", {})
        print(f"ok       {name}: {ci.get('files', 0)} files, "
              f"{ci.get('subcats', 0)} subcategories")


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
            "clshow": "!hidden",
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


def collect(seed, n):
    """Files of a seed category; if too few, also use its subcategories."""
    cat = "Category:" + seed
    pages = fetch_files(cat, n)
    edges = []
    if len(pages) < n:
        data = api({"list": "categorymembers", "cmtitle": cat,
                    "cmtype": "subcat", "cmlimit": "30"})
        subs = data.get("query", {}).get("categorymembers", [])
        # take a few images from MANY subcategories, so one photographer's batch
        # or one sub-topic cannot dominate
        per_sub = max(5, n // max(len(subs), 1))
        for s in subs:
            if len(pages) >= n:
                break
            edges.append((s["title"], cat))
            for k, v in fetch_files(s["title"], min(per_sub, n - len(pages))).items():
                v["cats"].add(cat)   # the seed is the (indirect) parent of this image
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


def write_outputs(rows, all_edges):
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["pageid", "title", "timestamp", "license", "seeds",
                "categories", "description", "image_path"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows.values():
            row = dict(r)
            row["seeds"] = "|".join(sorted(r["seeds"]))
            w.writerow(row)
    with open(OUT / "graph_edges.json", "w", encoding="utf-8") as f:
        json.dump(all_edges, f, indent=2)


def main():
    global SEEDS, N_PER_CAT, N_SMALL
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    OUT.mkdir(exist_ok=True)
    if mode == "check":
        check(SEEDS)
        print("\nDiwali alternatives:")
        check(DIWALI_ALTERNATIVES)
        return
    if mode == "test":
        SEEDS, N_PER_CAT, N_SMALL = SEEDS[:2], 10, 10
        print("TEST MODE: 2 categories, 10 images each\n")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows, all_edges = {}, []
    try:
        for seed in SEEDS:
            n = N_SMALL if seed in SMALL else N_PER_CAT
            try:
                pages, edges = collect(seed, n)
            except Exception as err:
                print(f"{seed}: FAILED ({err}) - skipped, run the script again later")
                continue
            all_edges += edges
            kept = 0
            for pid, e in list(pages.items())[:n]:
                info = e["info"]
                if not info or info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                meta = info.get("extmetadata", {})
                lic = meta.get("LicenseShortName", {}).get("value", "")
                if not licence_ok(lic):
                    continue
                if pid in rows:
                    rows[pid]["seeds"].add(seed)
                    continue
                ext = ".png" if info["mime"] == "image/png" else ".jpg"
                path = IMG_DIR / f"{pid}{ext}"
                url = info.get("thumburl") or info["url"]
                if not download(url, path):
                    continue
                time.sleep(0.2)
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
    finally:
        write_outputs(rows, all_edges)   # always written, even after Ctrl+C

    print(f"\nDone. {len(rows)} images saved in {IMG_DIR}")
    print(f"Metadata: {OUT / 'metadata.csv'}")


if __name__ == "__main__":
    main()
