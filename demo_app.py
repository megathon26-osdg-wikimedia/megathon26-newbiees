"""demo_app.py - upload an image, see suggestions + the similar images behind them,
tick the correct ones. Accepted images join the index at once (no retraining).
Run:  python demo_app.py   then open http://127.0.0.1:7860"""
import joblib
import numpy as np
import gradio as gr
from PIL import Image

from clip_utils import embed_pil
from common import DATA, load_clean

df, cats, Z, Y, idx, texts = load_clean()
sug = joblib.load(DATA / "suggester.joblib")
sug.set_index(Z, Y, texts)                       # demo index = every known image
paths, labels = df["image_path"].tolist(), df["labels"].tolist()
ci = {c: i for i, c in enumerate(cats)}
LAST = {"z": None, "img": None, "title": "", "shown": []}


def suggest(img, title):
    if img is None:
        return [], [], gr.update(choices=[], value=[])
    z = embed_pil([img])
    LAST.update(z=z, img=img, title=title or "")
    picks = sug.suggest(z, LAST["title"], k=10)
    names = [c for c, _ in picks]
    LAST["shown"] = names
    near = np.argsort(-(z @ sug.Zref.T)[0])[:6]
    gallery = [(Image.open(paths[j]).convert("RGB"), "; ".join(labels[j][:2])) for j in near]
    return [[c, round(s, 3)] for c, s in picks], gallery, gr.update(choices=names, value=[])


def save_review(selected):
    if LAST["z"] is None:
        return "Upload an image first."
    rejected = [n for n in LAST["shown"] if n not in selected]
    with open(DATA / "feedback_rejects.csv", "a", encoding="utf-8") as f:
        f.writelines(n.replace(",", " ") + "\n" for n in rejected)    # reliable negatives
    if not selected:
        return f"Logged {len(rejected)} rejected suggestions."
    y = np.zeros((1, len(cats)), dtype="float32")
    for n in selected:
        y[0, ci[n]] = 1
    (DATA / "feedback").mkdir(exist_ok=True)
    path = f"data/feedback/{len(paths)}.jpg"
    LAST["img"].convert("RGB").save(path)
    sug.add_to_index(LAST["z"], y, LAST["title"])
    paths.append(path)
    labels.append(list(selected))
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
