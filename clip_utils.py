"""clip_utils.py - CLIP loading and embedding helpers (the only file that needs torch)."""
import io

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_state = {}


def load_clip():
    if not _state:
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        _state.update(model=model.to(DEVICE).eval(), pre=pre,
                      tok=open_clip.get_tokenizer("ViT-B-32"))
    return _state["model"], _state["pre"], _state["tok"]


def embed_pil(images):
    """list of PIL images -> (n, 512) float32, every row has length 1."""
    model, pre, _ = load_clip()
    x = torch.stack([pre(im.convert("RGB")) for im in images]).to(DEVICE)
    with torch.no_grad():
        z = model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype("float32")


def embed_texts(texts, bs=256):
    model, _, tok = load_clip()
    out = []
    for i in range(0, len(texts), bs):
        with torch.no_grad():
            t = model.encode_text(tok(texts[i:i + bs]).to(DEVICE))
            t = t / t.norm(dim=-1, keepdim=True)
        out.append(t.cpu().numpy().astype("float32"))
    return np.concatenate(out)


def embed_paths(paths, bs=32, degrade_fn=None):
    """Unreadable files get an all-zero row (prepare_labels.py drops them)."""
    out = []
    for i in tqdm(range(0, len(paths), bs)):
        ims, ok = [], []
        for p in paths[i:i + bs]:
            try:
                im = Image.open(p).convert("RGB")
                ok.append(True)
            except Exception:
                im = Image.new("RGB", (224, 224))
                ok.append(False)
            ims.append(degrade_fn(im) if degrade_fn else im)
        z = embed_pil(ims)
        z[~np.array(ok)] = 0
        out.append(z)
    return np.concatenate(out)


def degrade(im, kind):
    """Simulate bad uploads: blur, tiny (low resolution) or jpeg (heavy compression)."""
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(3))
    if kind == "tiny":
        w, h = im.size
        return im.resize((max(16, w // 6), max(16, h // 6))).resize((w, h))
    if kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=10)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return im
