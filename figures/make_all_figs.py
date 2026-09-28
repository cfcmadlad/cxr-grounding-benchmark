"""Generate the full figure set for the progress deck.

Charts are recomputed from the CURRENT results_summary.csv files -- the plots in
outputs/evaluation/ are stale (7 Aug, before the MAIRA-2 and CheXagent fixes).

Writes into outputs/deck_figs/.
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

OUT = os.path.join(ROOT, "outputs", "deck_figs")
os.makedirs(OUT, exist_ok=True)

MODELS = {"Grounding DINO": "grounding_dino", "BioViL-T": "biovilt",
          "MAIRA-2": "maira2", "CheXagent": "chexagent"}
MCOL = {"Grounding DINO": "#E69F00", "BioViL-T": "#56B4E9",
        "MAIRA-2": "#CC79A7", "CheXagent": "#C9B826"}
GOLDC = "#00C838"
INK = "#21295C"

REGION_ORDER = [
    "right lung", "left lung", "cardiac silhouette", "mediastinum", "trachea",
    "right upper lung zone", "right mid lung zone", "right lower lung zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone",
    "right hilar region", "left hilar region",
    "right costophrenic angle", "left costophrenic angle",
]


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
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


print("loading results ...")
F = {}
for label, d in MODELS.items():
    df = pd.read_csv(f"{ROOT}/outputs/{d}/results_summary.csv")
    df["pred"] = df["pred_box"].map(pb)
    df["gold"] = df["gold_box"].map(pb)
    df["iou_n"] = pd.to_numeric(df["iou"], errors="coerce")
    F[label] = df

cfg = load_config(required_keys=["IMAGE_DIR"])
IMAGE_DIR = cfg["IMAGE_DIR"]
_img_cache = {}


def get_img(image_id):
    if image_id not in _img_cache:
        p = find_image_path(IMAGE_DIR, image_id)
        im = load_image(p)
        if im is None:
            _img_cache[image_id] = None
        else:
            im = im.convert("L")
            W, H = im.size
            d = im.copy(); d.thumbnail((900, 900))
            _img_cache[image_id] = (np.asarray(d), W, H)
    return _img_cache[image_id]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=115, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  saved", name, os.path.getsize(p) // 1024, "KB")


# ══════════════════════ 1. headline metrics ══════════════════════
summ = {}
for label, df in F.items():
    gp = df[df["gold"].notna()]
    sc = gp["iou_n"].notna()
    summ[label] = dict(n=len(gp), scored=int(sc.sum()),
                       det=100.0 * sc.sum() / len(gp),
                       mean=gp.loc[sc, "iou_n"].mean(),
                       median=gp.loc[sc, "iou_n"].median(),
                       r25=(gp.loc[sc, "iou_n"] >= .25).sum() / len(gp),
                       r50=(gp.loc[sc, "iou_n"] >= .50).sum() / len(gp))
S = pd.DataFrame(summ).T
print(S)

order = ["MAIRA-2", "BioViL-T", "CheXagent", "Grounding DINO"]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
ax = axes[0]
vals = [S.loc[m, "mean"] for m in order]
bars = ax.bar(order, vals, color=[MCOL[m] for m in order], edgecolor="white", linewidth=1.5)
ax.axhline(0.0794, ls="--", lw=2, color="#D1495B")
ax.text(3.42, 0.0794 + .012, "whole-image baseline 0.079", ha="right", fontsize=11,
        color="#D1495B", fontweight="bold")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}", ha="center",
            fontsize=14, fontweight="bold", color=INK)
ax.set_ylabel("Mean IoU vs gold", fontsize=13)
ax.set_title("Mean IoU by model", fontsize=16, fontweight="bold", color=INK)
ax.set_ylim(0, max(vals) * 1.25); ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=12)

ax = axes[1]
vals = [S.loc[m, "det"] for m in order]
bars = ax.bar(order, vals, color=[MCOL[m] for m in order], edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center",
            fontsize=14, fontweight="bold", color=INK)
ax.set_ylabel("% of regions with a predicted box", fontsize=13)
ax.set_title("Detection rate by model", fontsize=16, fontweight="bold", color=INK)
ax.set_ylim(0, 118); ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=12)
fig.tight_layout()
save(fig, "fig01_headline_metrics.png")

# ══════════════════════ 2. per-region heatmap ══════════════════════
mat = np.full((len(order), len(REGION_ORDER)), np.nan)
for i, m in enumerate(order):
    g = F[m].groupby("region")["iou_n"].mean()
    for j, r in enumerate(REGION_ORDER):
        if r in g.index:
            mat[i, j] = g[r]
fig, ax = plt.subplots(figsize=(15, 4.2))
im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(mat))
ax.set_xticks(range(len(REGION_ORDER)))
ax.set_xticklabels(REGION_ORDER, rotation=38, ha="right", fontsize=11)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=13)
for i in range(len(order)):
    for j in range(len(REGION_ORDER)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] < np.nanmax(mat) * .6 else "black",
                    fontweight="bold")
ax.set_title("Mean IoU per anatomical region", fontsize=17, fontweight="bold", color=INK, pad=12)
fig.colorbar(im, ax=ax, shrink=.85, label="Mean IoU")
fig.tight_layout()
save(fig, "fig02_region_heatmap.png")

# ══════════════════════ 3. IoU distribution ══════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
data = [F[m].loc[F[m]["iou_n"].notna(), "iou_n"].values for m in order]
bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False, widths=.55,
                medianprops=dict(color="black", lw=2))
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(MCOL[m]); patch.set_alpha(.85); patch.set_edgecolor("white")
for i, d in enumerate(data):
    x = np.random.normal(i + 1, 0.055, min(len(d), 900))
    ax.scatter(x, np.random.choice(d, min(len(d), 900), replace=False), s=3,
               alpha=.14, color=INK, zorder=1)
ax.set_ylabel("IoU vs gold", fontsize=13)
ax.set_title("Distribution of per-region IoU (scored regions only)", fontsize=16,
             fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=12)
ax.set_ylim(-.02, 1.0)
fig.tight_layout()
save(fig, "fig03_iou_distribution.png")

# ══════════════════════ 4. CheXagent offset ══════════════════════
rows = []
for m in order:
    df = F[m]
    sub = df[df["pred"].notna() & df["gold"].notna()]
    for _, r in sub.sample(min(1400, len(sub)), random_state=0).iterrows():
        g = get_img(r["image_id"])
        if g is None:
            continue
        _, W, H = g
        p, gd = r["pred"], r["gold"]
        rows.append({"model": m,
                     "dx": ((p[0] + p[2]) / 2 - (gd[0] + gd[2]) / 2) / W,
                     "dy": ((p[1] + p[3]) / 2 - (gd[1] + gd[3]) / 2) / H})
OFF = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), sharex=True, sharey=True)
for k, m in enumerate(order):
    ax = axes[k]
    sub = OFF[OFF.model == m]
    ax.axhline(0, color="#999", lw=1); ax.axvline(0, color="#999", lw=1)
    ax.scatter(sub.dx, sub.dy, s=7, alpha=.25, color=MCOL[m])
    mx, my = sub.dx.mean(), sub.dy.mean()
    ax.scatter([mx], [my], s=190, marker="X", color="#D1495B", edgecolor="white",
               linewidth=1.6, zorder=5)
    ax.set_title(f"{m}\nmean dy = {my:+.3f}", fontsize=13, fontweight="bold", color=INK)
    ax.set_xlim(-.55, .55); ax.set_ylim(.55, -.55)
    ax.set_xlabel("horizontal offset\n(fraction of width)", fontsize=10)
    if k == 0:
        ax.set_ylabel("vertical offset\n(fraction of height)", fontsize=10)
fig.suptitle("Predicted box centre relative to gold — CheXagent sits systematically high",
             fontsize=16, fontweight="bold", color=INK, y=1.04)
fig.tight_layout()
save(fig, "fig04_centre_offset.png")

# ══════════════════════ 5-9. qualitative galleries ══════════════════════
def gallery(model, fname, title, subtitle, picks):
    n = len(picks)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 3.75))
    axes = np.atleast_2d(axes)
    for k, (img_id, region, v) in enumerate(picks):
        ax = axes[k // ncol, k % ncol]
        g = get_img(img_id)
        if g is None:
            ax.axis("off"); continue
        arr, W, H = g
        ax.imshow(arr, cmap="gray", extent=(0, W, H, 0), interpolation="bilinear")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        row = F[model][(F[model].image_id == img_id) & (F[model].region == region)].iloc[0]
        if row["gold"] is not None:
            x1, y1, x2, y2 = row["gold"]
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                            color=GOLDC, lw=2.6))
        if row["pred"] is not None:
            x1, y1, x2, y2 = row["pred"]
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                            color=MCOL[model], lw=2.2, ls="--"))
        ax.set_title(f"{region}\nIoU {v:.2f}", fontsize=11, color=INK, fontweight="bold")
    for k in range(len(picks), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    handles = [Line2D([0], [0], color=GOLDC, lw=3, label="Gold"),
               Line2D([0], [0], color=MCOL[model], lw=3, ls="--", label=model)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False,
               bbox_to_anchor=(.5, -0.015))
    fig.suptitle(title, fontsize=18, fontweight="bold", color=INK, y=1.055)
    fig.text(.5, 1.005, subtitle, ha="center", fontsize=12, color="#555")
    fig.tight_layout(rect=[0, .035, 1, .965])
    save(fig, fname)


def top_picks(model, n=8, best=True, spread=True):
    df = F[model]
    sub = df[df["pred"].notna() & df["gold"].notna() & df["iou_n"].notna()]
    sub = sub.sort_values("iou_n", ascending=not best)
    sub = sub.drop_duplicates(subset=["image_id"])          # variety of patients
    if spread:
        sub = sub.drop_duplicates(subset=["region"])        # variety of regions
    return [(r.image_id, r.region, r.iou_n) for r in sub.head(n).itertuples()]


def typical_picks(model, n=8):
    """Sample at the MEDIAN of the IoU distribution -- representative, not
    cherry-picked. Spread across distinct patients and distinct regions."""
    df = F[model]
    sub = df[df["pred"].notna() & df["gold"].notna() & df["iou_n"].notna()]
    sub = sub.sort_values("iou_n").reset_index(drop=True)
    lo, hi = int(len(sub) * 0.40), int(len(sub) * 0.60)
    band = sub.iloc[lo:hi]
    band = band.drop_duplicates(subset=["image_id"]).drop_duplicates(subset=["region"])
    if len(band) < n:                       # widen if the band was too thin
        band = sub.iloc[int(len(sub) * .25):int(len(sub) * .75)]
        band = band.drop_duplicates(subset=["image_id"]).drop_duplicates(subset=["region"])
    return [(r.image_id, r.region, r.iou_n) for r in band.head(n).itertuples()]


gallery("MAIRA-2", "fig05_maira2_best.png",
        "MAIRA-2 — strongest cases",
        "Eight different patients and eight different regions; dashed = prediction, solid = gold",
        top_picks("MAIRA-2", 8, best=True))

gallery("MAIRA-2", "fig06_maira2_typical.png",
        "MAIRA-2 — typical (median) cases",
        "Sampled from the middle of the IoU distribution — not cherry-picked",
        typical_picks("MAIRA-2", 8))

gallery("BioViL-T", "fig07_biovilt.png",
        "BioViL-T — typical cases",
        "Sampled from the middle of the distribution: finds the general area, boxes offset or oversized",
        typical_picks("BioViL-T", 8))

gallery("Grounding DINO", "fig08_gdino.png",
        "Grounding DINO — typical cases",
        "Sampled from the middle of the distribution: the dashed prediction is the image border",
        typical_picks("Grounding DINO", 8))

gallery("CheXagent", "fig09_chexagent.png",
        "CheXagent — typical cases",
        "Sampled from the middle of the distribution: boxes displaced upward by ~16% of image height",
        typical_picks("CheXagent", 8))

print("\nALL FIGURES DONE ->", OUT)
