"""cutoff.py - the score below which a suggestion is hidden in the demo.
cutoff_test.py writes data/cutoff.json; demo_app.py reads it with load_cutoff()."""
import json

from common import DATA


def load_cutoff(default=0.05):
    try:
        return float(json.load(open(DATA / "cutoff.json"))["cutoff"])
    except Exception:
        return default
