import ast
import csv
import gzip
import os
import random
import sys

import yaml
from PIL import Image

R = "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed"
A = "/home/manik/pranjali/Aditya_project"
O = os.path.join(R, "outputs_final")
sys.path.insert(0, R)
from cxr_common import find_image_path, load_gold_annotations, load_image


def literal(path, name):
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


src = os.path.join(R, "models", "medgemma15_medsam.py")
regions = literal(src, "REGIONS")
r2b = literal(src, "REGION_TO_BBOX_NAME")
image_dir = os.path.join(A, "gold_images", "gold_images", "gold_images")
gold = load_gold_annotations(os.path.join(A, "gold_images", "gold_bbox_coordinate_annotations_1000images.csv"))
os.makedirs(os.path.join(O, "shards"), exist_ok=True)
os.makedirs(os.path.join(O, "logs"), exist_ok=True)

with open(os.path.join(R, "outputs", "medgemma15", "results_summary.csv")) as f:
    keep = set(r["image_id"] for r in csv.DictReader(f))
extra = []
rows = []
for image_id in gold:
    p = find_image_path(image_dir, image_id)
    if p is None:
        continue
    if image_id not in keep:
        extra.append(image_id)
        continue
    with Image.open(p) as im:
        w, h = im.size
    for region in regions:
        name = r2b.get(region, region).lower()
        rows.append((image_id + "|" + region, image_id, region, p, w, h, int(name in gold[image_id])))

with gzip.open(os.path.join(O, "queries.csv.gz"), "wt") as f:
    f.write("query_id,image_id,region,path,W,H,gold_present\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

rng = random.Random(0)
by_region = {}
for r in rows:
    by_region.setdefault(r[2], []).append(r[0])
pick = []
for region in regions:
    pick += rng.sample(by_region[region], 13)
chosen = set(pick)
rest = [r[0] for r in rows if r[0] not in chosen]
pick += rng.sample(rest, 200 - len(pick))
order = {r[0]: i for i, r in enumerate(rows)}
pick.sort(key=order.get)
with open(os.path.join(O, "j0_query_ids.txt"), "w") as f:
    f.write("\n".join(pick) + "\n")

plan = [("medgemma", 15, 5.8), ("radvlm", 12, 4.5), ("chexagent", 10, 3.45), ("maira2", 5, 0.72), ("chex", 1, 0.1)]
with open(os.path.join(O, "shard_plan.csv"), "w") as f:
    f.write("model,nshards,queries_per_shard,est_s_per_query,est_hours_per_shard\n")
    for m, n, s in plan:
        q = len(rows) / n
        f.write("%s,%d,%.0f,%s,%.2f\n" % (m, n, q, s, q * s / 3600))

with open(os.path.join(R, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
project_hf = cfg["HF_HOME"]
cfg["OUTPUT_DIR"] = os.path.join(O, "scratch")
cfg["MAX_IMAGES"] = None
for tag, hf in (("default", "/home/manik/.cache/huggingface"), ("project", project_hf)):
    c = dict(cfg)
    c["HF_HOME"] = hf
    with open(os.path.join(O, "config_" + tag + ".yaml"), "w") as f:
        yaml.safe_dump(c, f)

for e in extra:
    print("extra", e, "load_ok", load_image(find_image_path(image_dir, e)) is not None)
print("images", len(set(r[1] for r in rows)), "queries", len(rows), "gold_present", sum(r[6] for r in rows), "j0", len(pick))
