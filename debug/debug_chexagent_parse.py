"""Replay real CheXagent raw responses through the FIXED parser.

Reads raw responses straight out of slurm.264874.out (job 264874) and runs them
through the parsing code as it is actually deployed in chexagent_medsam.py --
the parser block is extracted from that file and exec'd, so this tests the
shipped code rather than a copy that could drift.

No model is loaded. Read-only: writes nothing.
"""
import os
import re
import sys

from PIL import Image

REPO = "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed"
LOG = os.path.join(REPO, "slurm.264874.out")
IMG_DIR = "/home/manik/pranjali/Aditya_project/gold_images/gold_images/gold_images"
SRC = os.path.join(REPO, "chexagent_medsam.py")

# ── extract the parser block from the real source file ───────────────────────
src = open(SRC).read()
start = src.index("_NUM = ")
end = src.index("# ── MEDSAM SEGMENTATION")
block = src[start:end]
print("Extracted %d chars of parser source from chexagent_medsam.py" % len(block))
ns = {"re": re}
exec(compile(block, SRC, "exec"), ns)
parse_box_from_response = ns["parse_box_from_response"]
print("Parser loaded:", parse_box_from_response)
print("Fallback patterns remaining:", len(ns["_BOX_PATTERNS"]),
      "(bare-number pattern removed)" if len(ns["_BOX_PATTERNS"]) == 4 else "!! UNEXPECTED")
print()

# ── pull raw responses out of the log ────────────────────────────────────────
RAW_RE = re.compile(r"^\s*RAW\[([^/]+)/([^\]]+)\]: '(.*)'\s*$")
rows = []
with open(LOG, errors="replace") as f:
    for line in f:
        m = RAW_RE.match(line.rstrip("\n"))
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
print("total RAW responses in log:", len(rows))

with_box = [r for r in rows if "<box>" in r[2]]
print("responses containing <box>:", len(with_box))
print()

dims_cache = {}


def dims(image_id):
    if image_id not in dims_cache:
        p = os.path.join(IMG_DIR, image_id + ".jpg")
        try:
            with Image.open(p) as im:
                dims_cache[image_id] = im.size
        except Exception:
            dims_cache[image_id] = None
    return dims_cache[image_id]


print("=" * 100)
print("SAMPLE: 20 real <box> responses through the FIXED parser")
print("=" * 100)
shown = 0
for image_id, region, resp in with_box:
    wh = dims(image_id)
    if wh is None:
        continue
    W, H = wh
    box = parse_box_from_response(resp, W, H)
    print()
    print("[%d] %s / %s   image=%dx%d" % (shown + 1, image_id[:18], region, W, H))
    print("    raw   :", resp[:150])
    print("    parsed:", box)
    if box:
        frac = ((box[2] - box[0]) * (box[3] - box[1])) / float(W * H)
        print("    -> box covers %.1f%% of image; within bounds: %s"
              % (100 * frac, box[0] >= 0 and box[1] >= 0 and box[2] <= W and box[3] <= H))
    shown += 1
    if shown >= 20:
        break

# ── aggregate: how much does the fix recover, and are values sane? ───────────
print()
print("=" * 100)
print("AGGREGATE over ALL %d responses" % len(rows))
print("=" * 100)
parsed = 0
failed = 0
skipped_nodim = 0
areas = []
leaked = 0          # boxes that look like raw 0-999 values leaked through
maxcoord = 0
for image_id, region, resp in rows:
    wh = dims(image_id)
    if wh is None:
        skipped_nodim += 1
        continue
    W, H = wh
    box = parse_box_from_response(resp, W, H)
    if box is None:
        failed += 1
        continue
    parsed += 1
    areas.append(((box[2] - box[0]) * (box[3] - box[1])) / float(W * H))
    maxcoord = max(maxcoord, box[2], box[3])
    # a 0-999 value leaking through on a ~3000x2500 image would give a tiny box
    # pinned near the origin
    if box[2] <= 1000 and box[3] <= 1000 and W > 1500 and H > 1500:
        leaked += 1

print("responses parsed to a box :", parsed)
print("responses with no box     :", failed)
print("skipped (image missing)   :", skipped_nodim)
print()
if areas:
    areas.sort()
    n = len(areas)
    print("box area as fraction of image:")
    print("   min    %.4f" % areas[0])
    print("   median %.4f" % areas[n // 2])
    print("   max    %.4f" % areas[-1])
    print("max x2/y2 coordinate seen :", maxcoord)
    print("suspicious 0-999-looking boxes on large images:", leaked,
          "(want 0)" if leaked == 0 else "<-- INVESTIGATE")
print()
print("DONE")
