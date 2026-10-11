"""category_types.py - which of the four types (object / scene / event / abstract) is each category?

The problem statement asks for objects, scenes, events and abstract concepts, so we report
results per type. Every image knows which SEED it was downloaded from; a category gets the
type of the seeds that most of its training images came from (so sub-categories inherit it).
Edit SEED_TYPE if you change the seed list in download_data.py."""
from collections import Counter

import numpy as np

SEED_TYPE = {
    # scene (places, landmarks, landscapes)
    "Taj Mahal": "scene", "Eiffel Tower": "scene", "Charminar": "scene", "Red Fort": "scene",
    "Statue of Liberty": "scene", "Colosseum": "scene", "Sunsets": "scene",
    "Waterfalls": "scene", "Beaches": "scene", "Glaciers": "scene", "Deserts": "scene",
    "Lighthouses": "scene", "Windmills": "scene", "Tea plantations": "scene",
    # object (animals, vehicles, things)
    "Panthera tigris": "object", "Elephas maximus": "object", "Pavo cristatus": "object",
    "Panthera leo": "object", "Giraffa camelopardalis": "object", "Airbus A380": "object",
    "Boeing 747": "object", "Bicycles": "object", "Diesel locomotives": "object",
    "Steam locomotives": "object", "Hot air balloons": "object", "Rickshaws": "object",
    "Double-decker buses": "object", "Kites (toy)": "object", "Chess": "object",
    # event
    "Holi": "event", "Ganesh Chaturthi": "event", "Christmas markets": "event",
    "Durga Puja": "event", "Oktoberfest": "event", "Fireworks": "event",
    "Carnivals": "event", "Marathons": "event",
    # abstract concepts, patterns, art
    "Rangoli": "abstract", "Mandalas": "abstract", "Fractals": "abstract",
    "Graffiti": "abstract", "Silhouettes": "abstract", "Calligraphy": "abstract",
}


def category_types(df, Y, train_idx):
    """Array with one type name per category column (majority vote over TRAIN images)."""
    seeds = [s.split("|") for s in df["seeds"].values]
    out = []
    unknown = set()
    for j in range(Y.shape[1]):
        votes = Counter()
        for i in train_idx[Y[train_idx, j] > 0]:
            for s in seeds[i]:
                if s in SEED_TYPE:
                    votes[SEED_TYPE[s]] += 1
                elif s:
                    unknown.add(s)
        out.append(votes.most_common(1)[0][0] if votes else "other")
    if unknown:
        print("WARNING: seeds without a type (add them to SEED_TYPE):", sorted(unknown))
    return np.array(out)
