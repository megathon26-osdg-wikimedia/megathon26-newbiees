# Commons Category Suggester

This README only covers setup and how to run the code.

## Requirements

- Python 3.13 (the pinned versions in `requirements.txt` were installed on 3.13)
- Internet access for the first run (Commons API, and the CLIP weights from the Hugging Face Hub)
- No GPU needed; the pinned PyTorch build is CPU-only (Linux and Windows)

## Setup

Run all commands from the repository root (the scripts use the relative folder `data/`).

```
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

Run the steps in this order. Each step writes files into `data/`.

| Step | Command | Output |
|------|---------|--------|
| 1. Download images and metadata | `python download_data.py` | `data/images/`, `data/metadata.csv`, `data/graph_edges.json` |
| 2. Embed images with CLIP | `python embed_images.py` | `data/Z.npy`, `data/Z_ids.npy` |
| 3. Clean labels, split by time, remove near-duplicates | `python prepare_labels.py` | `data/clean.csv`, `Y.npy`, `Zc.npy`, `cats.json` |
| 4. Fetch category parents (2 levels) | `python fetch_graph.py` | `data/graph_edges_full.json` |
| 5. Train the head, text index and fusion | `python train.py` | `data/suggester.joblib` |
| 6. Evaluate on the test split | `python evaluate.py` | printed tables, `data/results_*.csv`, `data/results.png` |

Quick checks before the full download:

```
python download_data.py check    # verifies the seed category names, downloads nothing
python download_data.py test     # 2 categories, 10 images each
python embed_images.py test      # embeds 32 images, saves nothing
```

The download can be interrupted with Ctrl+C and re-run; images already saved are skipped.

## Run the demo

Steps 1 to 5 must have been run first.

```
python demo_app.py
```

Then open http://127.0.0.1:7860 in a browser. Upload an image (a title is optional),
click "Suggest categories", tick the correct suggestions and click "Save my review".
Accepted reviews are stored in `data/feedback/` and reloaded the next time the demo starts.
Stop the demo with Ctrl+C.

## Optional scripts

| Command | Purpose |
|---------|---------|
| `python evaluate.py --robust` | Blur / low-resolution / JPEG robustness table (needs CLIP, a few minutes) |
| `python eval_by_type.py` | Results per category type |
| `python eval_resolution.py` | Results by simulated image resolution |
| `python latency.py` | Time for one suggestion and index size |
| `python throughput.py` | Images per second for batch sizes 1, 8 and 32 |
| `python tier_counts.py` | Size of the head / torso / tail tiers |
| `python cutoff_test.py` | Optional. Chooses the demo's display cutoff; without `data/cutoff.json` the demo uses 0.05. Needs 15-20 photos outside all categories in `data/outside/` |
| `python language_test.py` | Title scorer on non-English titles; needs `data/lang_test.csv` (columns `image_path,title,language,labels`) |
| `python retrain_from_feedback.py` | Adds accepted demo reviews to the training split and retrains |

## Notes

- The seed categories are listed in `SEEDS` in `download_data.py`. If you add a seed,
  add its type to `SEED_TYPE` in `category_types.py` as well.
- The dataset is not included in the repository. Step 1 downloads it from the Commons API.
- Images used are openly licensed Commons files; each file's licence is saved in `data/metadata.csv`.

## License

MIT. See the `LICENSE` file.
