"""demo_app.py - upload an image, see suggestions + the similar images behind them,
tick the correct ones. Accepted reviews are SAVED to disk and reloaded at every start,
so nothing a reviewer accepted is ever forgotten.
Run:  python demo_app.py   then open http://127.0.0.1:7860   (stop with Ctrl+C)"""
import time

import joblib
import numpy as np
import gradio as gr
from PIL import Image

from clip_utils import embed_pil
from common import DATA, load_clean
from cutoff import load_cutoff
from feedback_store import load_accepted, log_review

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
sug.set_index(Z, Y, texts)                       # demo index = every known image
paths, labels = df["image_path"].tolist(), df["labels"].tolist()
ci = {c: i for i, c in enumerate(cats)}
CUTOFF = load_cutoff(0.05)                        # data/cutoff.json if cutoff_test.py was run

# bring back everything accepted in earlier sessions
fb_paths, fb_Z, fb_Y, fb_titles, fb_labels = load_accepted(cats)
for k in range(len(fb_paths)):
    sug.add_to_index(fb_Z[k:k + 1], fb_Y[k:k + 1], fb_titles[k])
    paths.append(fb_paths[k])
    labels.append(fb_labels[k])
print(f"cutoff {CUTOFF:.2f} | index holds {len(paths)} images "
      f"({len(fb_paths)} restored from earlier reviews)")

LAST = {"z": None, "img": None, "title": "", "shown": []}


def suggest(img, title):
    if img is None:
        return [], [], gr.update(choices=[], value=[])
    z = embed_pil([img])
    LAST.update(z=z, img=img, title=title or "")
    picks = [(c, s) for c, s in sug.suggest(z, LAST["title"], k=10, cutoff=CUTOFF) if s >= CUTOFF]
    names = [c for c, _ in picks]
    LAST["shown"] = names
    gallery = []
    if picks:                                     # no gallery when nothing passes the cutoff
        near = np.argsort(-(z @ sug.Zref.T)[0])[:6]
        gallery = [(Image.open(paths[j]).convert("RGB"), "; ".join(labels[j][:2])) for j in near]
    return [[c, round(s, 3)] for c, s in picks], gallery, gr.update(choices=names, value=[])


def save_review(selected):
    if LAST["z"] is None:
        return "Nothing to save. Upload an image and click Suggest categories first."
    selected = list(selected or [])
    rejected = [n for n in LAST["shown"] if n not in selected]
    with open(DATA / "feedback_rejects.csv", "a", encoding="utf-8") as f:
        f.writelines(n.replace(",", " ") + "\n" for n in rejected)    # reliable negatives
    if not selected:
        log_review("", LAST["title"], [], LAST["shown"])
        LAST.update(z=None, img=None, shown=[])   # this result is now used up
        return f"Logged {len(rejected)} rejected suggestions."
    y = np.zeros((1, len(cats)), dtype="float32")
    for n in selected:
        y[0, ci[n]] = 1
    (DATA / "feedback").mkdir(exist_ok=True)
    path = f"data/feedback/{int(time.time() * 1000)}.jpg"        # unique name, never reused
    LAST["img"].convert("RGB").save(path)
    log_review(path, LAST["title"], selected, LAST["shown"])      # written to disk
    sug.add_to_index(LAST["z"], y, LAST["title"])
    paths.append(path)
    labels.append(selected)
    LAST.update(z=None, img=None, shown=[])       # a second click on Save cannot add it again
    return f"Saved. The index now holds {len(paths)} images."


with gr.Blocks(title="Commons category suggester") as demo:
    gr.Markdown("## Upload an image to get category suggestions\n"
                "Suggestions only - a human accepts or rejects them.")
    inp = gr.Image(type="pil", label="Upload an image")
    title = gr.Textbox(label="File name / title (optional, helps the text scorer)")
    go = gr.Button("Suggest categories")
    table = gr.Dataframe(headers=["Category", "Score"], label="Suggestions")
    gal = gr.Gallery(label="Why? The most similar known images", columns=6)
    pick = gr.CheckboxGroup(label="Tick the correct suggestions")
    save = gr.Button("Save my review (accepted images join the index)")
    msg = gr.Textbox(label="Status")
    go.click(suggest, [inp, title], [table, gal, pick])
    save.click(save_review, pick, msg)

demo.launch()
