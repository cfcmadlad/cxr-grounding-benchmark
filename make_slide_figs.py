"""Two compact, slide-shaped figures from ONE chest X-ray, so the contrast
between the models is apples-to-apples.

  slide_working_models.png       gold vs MAIRA-2 and BioViL-T
  slide_underperforming.png      gold vs Grounding DINO and CheXagent
"""
import ast
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

OUT = os.path.join(ROOT, "outputs", "comparison_grid")
os.makedirs(OUT, exist_ok=True)

IMG = "70146be7-493e5cef-63ce50ad-28d1b378-e735d4f4"
REGIONS = ["cardiac silhouette", "mediastinum", "right mid lung zone"]
DIRS = {"Grounding DINO": "grounding_dino", "BioViL-T": "biovilt",
        "MAIRA-2": "maira2", "CheXagent": "chexagent"}
STYLE = {
    "gold":           dict(color="#00E63C", lw=4.0, ls="-"),
    "MAIRA-2":        dict(color="#CC79A7", lw=3.0, ls="-"),
    "BioViL-T":       dict(color="#56B4E9", lw=3.0, ls="-."),
    "Grounding DINO": dict(color="#E69F00", lw=3.0, ls="--"),
    "CheXagent":      dict(color="#F0E442", lw=3.0, ls=":"),
}


def pb(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        b = ast.literal_eval(v)
        return [float(x) for x in b] if len(b) == 4 else None
    except Exception:
        return None


def iou(a, b):
    if a is None or b is None:
        return None
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


frames = {}
for label, d in DIRS.items():
    df = pd.read_csv(f"{ROOT}/outputs/{d}/results_summary.csv")
    df["pred"] = df["pred_box"].map(pb)
    df["gold"] = df["gold_box"].map(pb)
    frames[label] = df.set_index(["image_id", "region"])

cfg = load_config(required_keys=["IMAGE_DIR"])
pil = load_image(find_image_path(cfg["IMAGE_DIR"], IMG)).convert("L")
W, H = pil.size
disp = pil.copy()
disp.thumbnail((1400, 1400))
arr = np.asarray(disp)
print("image", IMG[:16], W, "x", H)


def build(fname, models, title, subtitle):
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(len(REGIONS) * 5.2, 6.0))
    for j, region in enumerate(REGIONS):
        ax = axes[j]
        ax.imshow(arr, cmap="gray", interpolation="bilinear", extent=(0, W, H, 0))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])

        gold = frames[models[0]].loc[(IMG, region), "gold"]
        if gold is not None:
            x1, y1, x2, y2 = gold
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            fill=False, **STYLE["gold"]))
        lines = []
        for m in models:
            bx = frames[m].loc[(IMG, region), "pred"]
            if bx is None:
                lines.append(f"{m}: no prediction")
                continue
            x1, y1, x2, y2 = bx
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            fill=False, **STYLE[m]))
            lines.append(f"{m}: IoU {iou(bx, gold):.2f}")
        ax.set_title(region, fontsize=17, fontweight="bold", pad=9)
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=13, family="monospace", color="white",
                bbox=dict(facecolor="black", alpha=0.66, pad=5, edgecolor="none"))

    handles = [Line2D([0], [0], label="Gold (Chest ImaGenome)", **STYLE["gold"])] + \
              [Line2D([0], [0], label=m, **STYLE[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)
    fig.text(0.5, 0.915, subtitle, ha="center", fontsize=13.5, color="#333333")
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=125, facecolor="white")
    plt.close(fig)
    print("saved", p, os.path.getsize(p) // 1024, "KB")


build("slide_working_models.png", ["MAIRA-2", "BioViL-T"],
      "MAIRA-2 localises reliably; BioViL-T is inconsistent",
      "Same chest X-ray, three anatomical regions — predicted box vs gold annotation")

build("slide_underperforming.png", ["Grounding DINO", "CheXagent"],
      "Grounding DINO and CheXagent do not localise correctly",
      "Grounding DINO returns near whole-image boxes; CheXagent sits systematically high")
