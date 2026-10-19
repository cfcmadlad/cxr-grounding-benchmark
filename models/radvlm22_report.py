"""
radvlm22_report.py
------------------
PDF report of the 22-finding RadVLM run: one page per finding with the 5 images where the
model's two prompts agree most, plus a summary page.

No gold boxes exist for findings on these images, so "top 5" = ranked by self-consistency:
  * the detection prompt ("Are abnormalities present...") lists the finding, AND
  * its box overlaps the box from the single-finding prompt (highest IoU between the two).
This is NOT correctness. Red = single-finding prompt box, cyan = detection prompt box.

Contains MIMIC-CXR images: share only with PhysioNet-credentialed people, keep off OneDrive.
Run (HPC, radvlm env):  python models/radvlm22_report.py   -> OUTPUT_DIR/radvlm22/radvlm22_report.pdf
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cxr_common import load_config, compute_iou  # noqa: E402

cfg = load_config(required_keys=["IMAGE_DIR", "OUTPUT_DIR"])
RUN = os.path.join(cfg["OUTPUT_DIR"], "radvlm22")
TOPK = int(os.environ.get("TOPK", "5"))
RED, CYAN = "#e03c3c", "#18c5d8"


def img_path(iid):
    p = os.path.join(cfg["IMAGE_DIR"], iid + ".jpg")
    return p if os.path.exists(p) else None


def best_agreement(gb, grp):
    return max((compute_iou(a, b) for a in gb for b in grp), default=0.0)


def draw(ax, iid, row):
    im = Image.open(img_path(iid)).convert("L")
    W0, H0 = im.size
    im.thumbnail((700, 700))
    sx, sy = im.size[0] / row.img_w, im.size[1] / row.img_h
    ax.imshow(im, cmap="gray")
    for boxes, col, lw in ((json.loads(row.ground_boxes), RED, 1.8),
                           (json.loads(row.grouped_boxes), CYAN, 1.8)):
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(patches.Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy,
                                           fill=False, edgecolor=col, linewidth=lw))
    ax.axis("off")


def main():
    df = pd.read_csv(os.path.join(RUN, "predictions.csv"))
    df["present"] = df["grouped_present"].astype(str).str.lower().eq("true")
    n_img = df["image_id"].nunique()
    findings = list(dict.fromkeys(df["finding"]))
    out_pdf = os.path.join(RUN, "radvlm22_report.pdf")

    with PdfPages(out_pdf) as pdf:
        # ---- summary page ----
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.05, 0.94, "RadVLM on 22 chest X-ray findings", fontsize=20, weight="bold")
        fig.text(0.05, 0.905, f"{n_img} MIMIC-CXR images (Chest ImaGenome gold set) x 22 findings. "
                 "Predictions only; no gold boxes exist for findings on these images.", fontsize=10)
        s = (df.groupby("finding", sort=False)
               .agg(flagged=("present", "sum"), pct=("present", lambda x: round(100 * x.mean(), 1)))
               .reset_index())
        cell = [[r.finding, int(r.flagged), f"{r.pct}%"] for r in s.itertuples()]
        ax = fig.add_axes([0.2, 0.06, 0.6, 0.8]); ax.axis("off")
        t = ax.table(cellText=cell, colLabels=["Finding", "Images flagged", "% of images"],
                     loc="upper left", cellLoc="left", colWidths=[0.5, 0.25, 0.25])
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
        pdf.savefig(fig); plt.close(fig)

        # ---- one page per finding ----
        for f in findings:
            if f == "No finding":
                continue
            d = df[df.finding == f]
            cand = d[d.present & (d.n_ground_boxes > 0) & (d.grouped_boxes != "[]")].copy()
            cand["agree"] = [best_agreement(json.loads(a), json.loads(b))
                             for a, b in zip(cand.ground_boxes, cand.grouped_boxes)]
            cand = cand.loc[cand["image_id"].map(lambda i: img_path(i) is not None).astype(bool)]
            top = cand.sort_values(["agree", "image_id"], ascending=[False, True]).head(TOPK)
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.05, 0.94, f, fontsize=20, weight="bold")
            fig.text(0.05, 0.905,
                     f"Detection prompt flagged it on {int(d.present.sum())} of {len(d)} images "
                     f"({round(100 * d.present.mean(), 1)}%). Showing the {len(top)} with the highest "
                     "agreement between the two prompts' boxes.", fontsize=10)
            if top.empty:
                fig.text(0.05, 0.6, "Not flagged (with a box) on any image by the detection prompt,\n"
                         "so there is nothing to show. The single-finding prompt still returns a\n"
                         "box on every image, which is not evidence the finding is present.",
                         fontsize=12)
            n = max(len(top), 1)
            for k, r in enumerate(top.itertuples()):
                ax = fig.add_axes([0.03 + k * 0.194, 0.22, 0.185, 0.62])
                draw(ax, r.image_id, r)
                fig.text(0.03 + k * 0.194, 0.17, f"{r.image_id[:13]}\nagreement IoU {r.agree:.2f}",
                         fontsize=8, va="top")
            fig.text(0.05, 0.05, "red = single-finding prompt box   cyan = detection prompt box   "
                     "(model-vs-model agreement, not verified against gold)", fontsize=9, color="#555")
            pdf.savefig(fig); plt.close(fig)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
