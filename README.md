# Commons Category Suggester

## Requirements

- Python 3.10 or newer
- Internet access (for the Commons download and the first-time model download of about 600 MB)
- A GPU is optional. On CPU, embedding about 2,000 images takes roughly 10 to 15 minutes.

## Setup

```
git clone https://github.com/IIIT-ECell/megathon26-newbiees.git
cd megathon26-newbiees
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check the install:

```
python -c "import torch, open_clip; print('ok')"
```

## Run

Run every command from the project folder, with the virtual environment active. Run the steps in this order, because each step uses the files saved by the previous one. All data is downloaded and generated into a `data/` folder; nothing needs to be downloaded separately.

1. Download the data (about 30 to 60 minutes):
   ```
   python download_data.py test     # quick check: 2 categories, 10 images each
   python download_data.py
   ```
2. Fetch the category graph (optional, about 5 to 10 minutes):
   ```
   python fetch_graph.py
   ```
3. Compute image embeddings:
   ```
   python embed_images.py test      # quick check on 32 images, saves nothing
   python embed_images.py
   ```
4. Clean labels and split the data:
   ```
   python prepare_labels.py
   ```
5. Train the model:
   ```
   python train.py
   ```
6. Evaluate:
   ```
   python evaluate.py
   python evaluate.py --robust      # also runs the image-quality tests
   python latency.py                # optional: speed measurements
   ```
7. Start the demo:
   ```
   python demo_app.py
   ```
   Then open http://127.0.0.1:7860 in a browser.

If you change `prepare_labels.py` or `common.py`, re-run steps 4 to 6. Embeddings (step 3) only need to be recomputed when the images change.

## Troubleshooting

- `FileNotFoundError: data/metadata.csv`: run the scripts from the project folder, and make sure step 1 finished (it prints a line starting with `Done.`).
- `ModuleNotFoundError`: the virtual environment is not active. Run `source venv/bin/activate`.
- Out of memory during embedding: lower `bs=32` to `bs=8` in `embed_paths` in `clip_utils.py`.
- Download errors or `429`: the scripts wait and retry. Do not remove the pauses. Turn off any VPN if connections time out.

## License

MIT. See the `LICENSE` file.
