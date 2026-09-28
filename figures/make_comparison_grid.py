"""Build model-vs-gold bounding-box comparison grids.

Two PNGs, each 3 sample images (rows) x 3 anatomical regions (columns):
    comparison_grid_high_agreement.png
    comparison_grid_low_agreement.png
plus a legend//summary strip baked into each figure.

Only models with verified real results are drawn: Grounding DINO, BioViL-T,
MAIRA-2, CheXagent. BiomedParse and RadVLM are excluded (no results).
"""
import ast
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed"
sys.path.insert(0, ROOT)
os.environ.setdefault("CXR_CONFIG", os.path.join(ROOT, "config.yaml"))
from cxr_common import find_image_path, load_config, load_image  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs", "comparison_grid")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "Grounding DINO": "grounding_dino",
    "BioViL-T": "biovilt",
    "MAIRA-2": "maira2",
    "CheXagent": "chexagent",
}
# Okabe-Ito derived, high contrast on greyscale; linestyle varies too so the
# figure survives greyscale printing / colour-vision differences.
STYLE = {
    "gold":           dict(color="#00FF3C", lw=3.4, ls="-"),
    "Grounding DINO": dict(color="#E69F00", lw=2.0, ls="--"),
    "BioViL-T":       dict(color="#56B4E9", lw=2.0, ls="-."),
    "MAIRA-2":        dict(color="#CC79A7", lw=2.2, ls="-"),
    "CheXagent":      dict(color="#F0E442", lw=2.0, ls=":"),
}
REGIONS = ["cardiac silhouette", "mediastinum", "right mid lung zone"]


def pb(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        b = ast.literal_eval(v)
        return [float(x) for x in b] if isinstance(b, (list, tuple)) and len(b) == 4 else None
    except Exception:
        return None


def iou(a, b):
    if a is None or b is None:
        return None
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


# ── load ─────────────────────────────────────────────────────────────────────
frames = {}
for label, d in MODELS.items():
    df = pd.read_csv(f"{ROOT}/outputs/{d}/results_summary.csv")
    df["pred"] = df["pred_box"].map(pb)
    df["gold"] = df["gold_box"].map(pb)
    frames[label] = df.set_index(["image_id", "region"])

cfg = load_config(required_keys=["IMAGE_DIR"])
IMAGE_DIR = cfg["IMAGE_DIR"]

# ── candidate images: all 4 models present on all 3 chosen regions ───────────
per_region = {}
for r in REGIONS:
    d = {}
    for label, f in frames.items():
        sub = f.xs(r, level="region")
        d[label] = sub[sub["pred"].notna()]["pred"].to_dict()
    per_region[r] = d

common = None
for r in REGIONS:
    s = set.intersection(*[set(v) for v in per_region[r].values()])
    common = s if common is None else (common & s)
common = sorted(common)
print("images with all 4 models on all %d regions: %d" % (len(REGIONS), len(common)))

scored = []
for img in common:
    pw, gi = [], []
    for r in REGIONS:
        boxes = [per_region[r][l][img] for l in MODELS]
        pw += [iou(a, b) for a, b in itertools.combinations(boxes, 2)]
        gold = frames["Grounding DINO"].loc[(img, r), "gold"]
        gi += [iou(bx, gold) for bx in boxes]
    p = find_image_path(IMAGE_DIR, img)
    scored.append({
        "image_id": img,
        "agree": float(np.mean(pw)),
        "gold_iou": float(np.mean([x for x in gi if x is not None])),
        "path": p,
    })
sc = pd.DataFrame(scored).sort_values("agree", ascending=False).reset_index(drop=True)
print(sc.head(3)[["image_id", "agree", "gold_iou"]].to_string())
print(sc.tail(3)[["image_id", "agree", "gold_iou"]].to_string())

groups = [
    ("high_agreement", "Models agree most", sc.head(3).to_dict("records")),
    ("low_agreement", "Models disagree most", sc.tail(3).iloc[::-1].to_dict("records")),
]

legend_handles = [Line2D([0], [0], label="Gold annotation", **STYLE["gold"])] + [
    Line2D([0], [0], label=m, **STYLE[m]) for m in MODELS
]

PANEL = 5.6
for slug, title, recs in groups:
    nrow, ncol = len(recs), len(REGIONS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * PANEL, nrow * PANEL + 1.5))
    axes = np.atleast_2d(axes)

    for i, rec in enumerate(recs):
        img_id = rec["image_id"]
        pil = load_image(rec["path"])
        if pil.mode != "L":
            pil = pil.convert("L")
        W, H = pil.size          # ORIGINAL pixel size -- box coords live here
        # Downscale purely for rendering speed/file size. Drawing a 3000px image
        # into a ~600px panel is wasted work. extent= maps the shrunken array
        # back onto the original coordinate space, so box coords stay valid.
        disp = pil.copy()
        disp.thumbnail((1200, 1200))
        arr = np.asarray(disp)

        for j, region in enumerate(REGIONS):
            ax = axes[i, j]
            ax.imshow(arr, cmap="gray", interpolation="bilinear",
                      extent=(0, W, H, 0))
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_xticks([]); ax.set_yticks([])

            gold = frames["Grounding DINO"].loc[(img_id, region), "gold"]
            if gold is not None:
                x1, y1, x2, y2 = gold
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                fill=False, **STYLE["gold"]))
            lines = []
            for label in MODELS:
                bx = frames[label].loc[(img_id, region), "pred"]
                if bx is None:
                    lines.append(f"{label}: no box")
                    continue
                x1, y1, x2, y2 = bx
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                fill=False, **STYLE[label]))
                v = iou(bx, gold)
                lines.append(f"{label}: IoU {v:.3f}" if v is not None else f"{label}: n/a")

            ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                    va="top", ha="left", fontsize=9.5, family="monospace",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.62, pad=4.5, edgecolor="none"))
            if i == 0:
                ax.set_title(region, fontsize=15, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(f"{img_id[:13]}…\n{W}x{H}px", fontsize=10.5)

    fig.suptitle(
        f"CXR anatomical grounding — model predictions vs gold  |  {title}\n"
        f"gold set: 959 images x 15 regions   |   models shown: "
        f"Grounding DINO, BioViL-T, MAIRA-2, CheXagent",
        fontsize=16, fontweight="bold", y=0.995)
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=13, frameon=True, bbox_to_anchor=(0.5, 0.001))
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])

    out = os.path.join(OUT_DIR, f"comparison_grid_{slug}.png")
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    px = fig.get_size_inches() * 110
    print("saved %s  (%dx%d px, %.1f MB)"
          % (out, px[0], px[1], os.path.getsize(out) / 1e6))
