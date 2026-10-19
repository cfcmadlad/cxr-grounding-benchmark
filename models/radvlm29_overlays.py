"""Draw RadVLM (red) and gold (green) boxes on every image for every region.

CPU only; reads the finished 29-region run, writes
    <OUTPUT_DIR>/radvlm29/overlays/<region>/<image_id>.jpg
(959 images x 29 regions). Regions without a gold box (spine, abdomen) get the prediction
only, labelled as such. Safe to re-run: existing files are skipped.
"""
import ast
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

CFG = yaml.safe_load(open(os.environ.get("CXR_CONFIG", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))))
IMAGE_DIR = CFG["IMAGE_DIR"]
RUN_DIR = os.path.join(CFG["OUTPUT_DIR"], "radvlm29")
CSV = os.path.join(RUN_DIR, "results_summary.csv")
OUT = os.path.join(RUN_DIR, "overlays")

LONG = 1400
DARK, MUTED, WHITE = (10, 42, 102), (111, 143, 196), (255, 255, 255)
GOLD, PRED = (31, 157, 85), (224, 60, 60)
HDR, FTR = 70, 44


def font(sz):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default(sz)


def box(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        b = ast.literal_eval(v)
        return b if len(b) == 4 else None
    except Exception:
        return None


def slug(r):
    return r.replace(" ", "_")


def do_image(args):
    image_id, rows = args
    todo = [r for r in rows if not os.path.exists(os.path.join(OUT, slug(r["region"]), image_id + ".jpg"))]
    if not todo:
        return 0
    path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    if not os.path.exists(path):
        return -1
    base = Image.open(path).convert("L")
    s = LONG / max(base.size)
    base = base.resize((round(base.size[0] * s), round(base.size[1] * s)), Image.LANCZOS).convert("RGB")
    f1, f2, f3 = font(30), font(18), font(20)
    n = 0
    for r in todo:
        g, p = box(r["gold_box"]), box(r["pred_box"])
        bad_gold = g is not None and not (g[0] < g[2] and g[1] < g[3])
        if bad_gold:
            g = None          # reversed / zero-size annotation: do not draw, do not score
        im = base.copy()
        d = ImageDraw.Draw(im)
        for b, col, w in ((g, GOLD, 6), (p, PRED, 3)):
            if b:
                d.rectangle([min(b[0], b[2]) * s, min(b[1], b[3]) * s, max(b[0], b[2]) * s, max(b[1], b[3]) * s],
                            outline=col, width=w)
        W = im.size[0]
        out = Image.new("RGB", (W, im.size[1] + HDR + FTR), WHITE)
        out.paste(im, (0, HDR))
        o = ImageDraw.Draw(out)
        o.text((16, 8), r["region"].title().replace("Svc", "SVC"), font=f1, fill=DARK)
        o.text((16, 44), f"image {image_id[:8]}", font=f2, fill=MUTED)
        if bad_gold:
            tag = "invalid gold box (excluded)"
        elif g is None:
            tag = "no gold box (not scored)"
        elif p is None:
            tag = "no box returned"
        else:
            tag = f"IoU {float(r['iou']):.3f}"
        o.text((W - 16, 8), tag, font=f1, fill=DARK, anchor="ra")
        o.line([0, HDR - 2, W, HDR - 2], fill=MUTED, width=1)
        y = HDR + im.size[1] + FTR // 2
        o.line([16, y, 56, y], fill=GOLD, width=6)
        o.text((66, y), "Gold box", font=f3, fill=DARK, anchor="lm")
        o.line([190, y, 230, y], fill=PRED, width=3)
        o.text((240, y), "RadVLM prediction", font=f3, fill=DARK, anchor="lm")
        if r["region"] == "mediastinum":
            o.text((W - 16, y), "gold derived: upper mediastinum + cardiac silhouette", font=f2, fill=MUTED, anchor="rm")
        d_ = os.path.join(OUT, slug(r["region"]))
        os.makedirs(d_, exist_ok=True)
        tmp = os.path.join(d_, image_id + ".tmp.jpg")
        out.save(tmp, quality=88)
        os.replace(tmp, os.path.join(d_, image_id + ".jpg"))
        n += 1
    return n


def main():
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    groups = [(k, g.to_dict("records")) for k, g in df.groupby("image_id")]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    print(f"{len(groups)} images, {len(df)} overlays, {workers} workers", flush=True)
    t0, done, made, missing = time.time(), 0, 0, 0
    with Pool(workers) as pool:
        for n in pool.imap_unordered(do_image, groups, chunksize=2):
            done += 1
            if n < 0:
                missing += 1
            else:
                made += n
            if done % 50 == 0 or done == len(groups):
                print(f"{done}/{len(groups)} images, {made} overlays, {missing} missing, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
