# Commons Category Suggester

Automatic category suggestion for Wikimedia Commons (Megathon 2026).

Given a new image (and optionally its title), the system proposes a ranked list of
Commons categories and shows the similar known images behind each suggestion.
Suggestions are for human review; nothing is edited automatically.

## Approach

1. Embed every image with CLIP (open_clip, ViT-B-32).
2. Four scorers: zero-shot (category name vs image), kNN over similar images,
   a trained multi-label head, and a title/description text scorer.
3. A logistic-regression fusion combines the scorers (fitted on a validation split).
4. The Commons category graph is used to prefer the most specific category.
5. Evaluation on a time-based split with near-duplicates removed.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run order

```
python download_data.py          # wait for the line starting with "Done."
python fetch_graph.py
python embed_images.py test      # quick sanity check
python embed_images.py
python prepare_labels.py
python train.py
python evaluate.py
python evaluate.py --robust
python latency.py
python demo_app.py               # open http://127.0.0.1:7860
```

The `data/` folder is not stored in git. Download it again with `download_data.py`,
or get the zipped copy from: (add shared link here).

## Results

(add the tables from evaluate.py here)

## Limitations

- Small closed-world sample chosen by us; real Commons accuracy will be lower.
- Stored labels are incomplete, so strict scores undercount true quality.
- Seed categories carry their label by construction; hard mode hides them.
- Not grouped by uploader; English-leaning names and topics.
- Accepted images update the kNN index at once; the head and text scorer update at the next retrain.

## Team

(add names)
