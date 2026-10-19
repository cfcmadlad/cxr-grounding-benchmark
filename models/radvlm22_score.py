"""
radvlm22_score.py
-----------------
CPU scoring of the 22-finding RadVLM run (models/radvlm_22findings.py) against Chest
ImaGenome's GOLD attribute labels ("finding X is present in anatomical region R").

These images have no gold boxes for findings, so this is a WEAK check:
  * presence  -- grouped detection prompt says present/absent vs gold present/absent
                 -> sensitivity / specificity per finding
  * location  -- only where gold says present: does a predicted box's centre fall
                 inside the gold box of an anatomical region where gold puts the finding
                 (pointing-game hit), and the best IoU against the rectangle enclosing
                 those regions (coarse: an anatomical region is not a lesion outline).
Only images that appear in the attribute gold file are scored (about half of them).

VinDr -> Chest ImaGenome name mapping (ATTR_MAP) is approximate and is reported with
counts every run; findings with no matching attribute are listed as UNMAPPED, not scored.
"No finding" gold = the image has none of the mapped findings.

Config: GOLD_ATTR_FILE (config.yaml key or env var) = Chest ImaGenome gold attributes
file, e.g. gold_attributes_relations_500pts_500studies1st.txt.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cxr_common import load_config, compute_iou, get_logger  # noqa: E402

log = get_logger("radvlm22_score")
cfg = load_config(required_keys=["GOLD_CSV", "OUTPUT_DIR"])
RUN_DIR = os.path.join(cfg["OUTPUT_DIR"], "radvlm22")
ATTR_FILE = os.environ.get("GOLD_ATTR_FILE") or cfg.get("GOLD_ATTR_FILE")

# candidate Chest ImaGenome attribute names (lowercase, exact match) per finding
ATTR_MAP = {
    "Aortic enlargement": ["tortuous aorta", "enlarged aorta", "aortic enlargement"],
    "Atelectasis": ["atelectasis", "linear/patchy atelectasis", "lobar/segmental collapse"],
    "Calcification": ["calcified nodule", "vascular calcification", "calcification"],
    "Cardiomegaly": ["enlarged cardiac silhouette", "cardiomegaly"],
    "Clavicle fracture": ["clavicle fracture"],
    "Consolidation": ["consolidation"],
    "Edema": ["pulmonary edema/hazy opacity", "pulmonary edema", "edema"],
    "Emphysema": ["copd/emphysema", "emphysema"],
    "Enlarged pulmonary artery": ["enlarged pulmonary artery"],
    "Interstitial lung disease": ["interstitial lung disease",
                                  "increased reticular markings/ild pattern"],
    "Lung cavity": ["cavitation", "lung cavity"],
    "Lung cyst": ["lung cyst"],
    "Lung opacity": ["lung opacity"],
    "Mediastinal shift": ["mediastinal displacement", "mediastinal shift"],
    "Nodule/mass": ["mass/nodule (not otherwise specified)", "multiple masses/nodules",
                    "lung lesion"],
    "Pleural effusion": ["pleural effusion"],
    "Pleural thickening": ["pleural/parenchymal scarring", "pleural thickening"],
    "Pneumothorax": ["pneumothorax"],
    "Pulmonary fibrosis": ["pulmonary fibrosis"],
    "Rib fracture": ["rib fracture"],
    "Other lesion": [],
}


def load_anatomy_gold(csv_path):
    """{image_id: {region: box}} incl. derived 'mediastinum' (as radvlm_29regions.py)."""
    df = pd.read_csv(csv_path)
    ids = df["image_id"].astype(str).str.strip().str.replace(r"\.dcm$", "", case=False, regex=True)
    names = df["bbox_name"].astype(str).str.lower().str.strip()
    gold = {}
    for iid, name, x1, y1, x2, y2 in zip(ids, names, df["original_x1"], df["original_y1"],
                                         df["original_x2"], df["original_y2"]):
        if float(x2) > float(x1) and float(y2) > float(y1):  # 26 malformed gold boxes are excluded
            gold.setdefault(iid, {})[name] = [float(x1), float(y1), float(x2), float(y2)]
    for g in gold.values():
        um, cs = g.get("upper mediastinum"), g.get("cardiac silhouette")
        if "mediastinum" not in g and um is not None and cs is not None:
            g["mediastinum"] = [min(um[0], cs[0]), min(um[1], cs[1]), max(um[2], cs[2]), max(um[3], cs[3])]
    return gold


def load_attr_gold(path):
    """DataFrame [image_id, region, label, present] of gold attribute labels."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    print("attribute file columns:", list(df.columns))

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_img, c_reg = pick("image_id", "dicom_id"), pick("bbox", "bbox_name", "region")
    c_lab = pick("label_name", "label", "attribute")
    c_pres = pick("context", "relation", "present")
    if not all([c_img, c_reg, c_lab, c_pres]):
        sys.exit(f"cannot find image/region/label/present columns in {path}; "
                 f"got {list(df.columns)} -- adjust load_attr_gold")
    pres = df[c_pres].astype(str).str.lower().str.strip()
    print(f"presence column '{c_pres}' values:", pres.value_counts().to_dict())
    out = pd.DataFrame({
        "image_id": df[c_img].astype(str).str.strip().str.replace(r"\.dcm$", "", case=False, regex=True),
        "region": df[c_reg].astype(str).str.lower().str.strip(),
        "label": df[c_lab].astype(str).str.lower().str.strip(),
        "present": pres.isin(["1", "1.0", "yes", "true", "positive"]),
    })
    return out


def centre_in(box, region_box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region_box[0] <= cx <= region_box[2] and region_box[1] <= cy <= region_box[3]


def main():
    if not ATTR_FILE or not os.path.exists(ATTR_FILE):
        sys.exit(f"GOLD_ATTR_FILE not set or missing ({ATTR_FILE!r}). Point it at the Chest "
                 "ImaGenome gold attributes file (config.yaml key or env var).")
    pred = pd.read_csv(os.path.join(RUN_DIR, "predictions.csv"))
    anat = load_anatomy_gold(cfg["GOLD_CSV"])
    attr = load_attr_gold(ATTR_FILE)

    scored_ids = sorted(set(pred["image_id"].astype(str)) & set(attr["image_id"]))
    print(f"\npredicted images: {pred['image_id'].nunique()} | in attribute gold: "
          f"{attr['image_id'].nunique()} | scored (overlap): {len(scored_ids)}")
    if not scored_ids:
        sys.exit("no overlap between predicted images and attribute gold image ids")
    attr = attr[attr["image_id"].isin(scored_ids)]

    # mapping report
    pos_counts = attr[attr["present"]].groupby("label")["image_id"].nunique().sort_values(ascending=False)
    used = {a for v in ATTR_MAP.values() for a in v}
    print("\nPOSITIVE gold attribute labels (images) -- '*' = used by the mapping:")
    for lab, n in pos_counts.items():
        print(f"  {'*' if lab in used else ' '} {lab:45s} {n}")
    for f, cands in ATTR_MAP.items():
        hit = [c for c in cands if c in pos_counts.index]
        if cands and not hit:
            print(f"  (none of {cands} occur for {f})")

    pos = attr[attr["present"]]
    pos_by = {}  # (image, finding) -> set(regions)
    for f, cands in ATTR_MAP.items():
        s = pos[pos["label"].isin(cands)]
        for iid, reg in zip(s["image_id"], s["region"]):
            pos_by.setdefault((iid, f), set()).add(reg)

    pred = pred[pred["image_id"].astype(str).isin(scored_ids)].copy()
    pred["image_id"] = pred["image_id"].astype(str)
    rows = []
    for r in pred.itertuples(index=False):
        f, iid = r.finding, r.image_id
        said = str(r.grouped_present).lower() == "true"
        if f == "No finding":
            any_pos = any((iid, g) in pos_by for g in ATTR_MAP)
            rows.append(dict(image_id=iid, finding=f, mapped=True, gold_present=not any_pos,
                             pred_present=said))
            continue
        if not any(c in pos_counts.index for c in ATTR_MAP.get(f, [])):
            rows.append(dict(image_id=iid, finding=f, mapped=False))
            continue
        regions = pos_by.get((iid, f), set())
        row = dict(image_id=iid, finding=f, mapped=True, gold_present=bool(regions),
                   pred_present=said, gold_regions="|".join(sorted(regions)))
        rboxes = [anat.get(iid, {}).get(g) for g in regions]
        rboxes = [b for b in rboxes if b is not None]
        if rboxes:
            enc = [min(b[0] for b in rboxes), min(b[1] for b in rboxes),
                   max(b[2] for b in rboxes), max(b[3] for b in rboxes)]
            for key, col in (("ground", r.ground_boxes), ("grouped", r.grouped_boxes)):
                boxes = json.loads(col)
                if key == "grouped" and not boxes:
                    continue
                row[f"{key}_hit"] = any(centre_in(b, rb) for b in boxes for rb in rboxes)
                row[f"{key}_region_iou"] = max((compute_iou(b, enc) for b in boxes), default=0.0)
        rows.append(row)
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(RUN_DIR, "scores_per_query.csv"), index=False)

    out = []
    for f in [x for x in pred["finding"].drop_duplicates()]:
        s = sc[sc["finding"] == f]
        if not s["mapped"].any():
            out.append(dict(finding=f, status="UNMAPPED"))
            continue
        gp, pp = s["gold_present"].astype(bool), s["pred_present"].astype(bool)
        tp, fn = int((gp & pp).sum()), int((gp & ~pp).sum())
        fp, tn = int((~gp & pp).sum()), int((~gp & ~pp).sum())
        d = dict(finding=f, status="scored", n_img=len(s), gold_pos=int(gp.sum()),
                 sens_pct=round(100 * tp / (tp + fn), 1) if tp + fn else np.nan,
                 spec_pct=round(100 * tn / (tn + fp), 1) if tn + fp else np.nan)
        for key in ("ground", "grouped"):
            if f"{key}_hit" in s:
                h = s[f"{key}_hit"].dropna()
                d[f"{key}_hit_pct"] = round(100 * h.astype(bool).mean(), 1) if len(h) else np.nan
                d[f"{key}_n_loc"] = len(h)
                iou = s[f"{key}_region_iou"].dropna()
                d[f"{key}_region_iou"] = round(iou.mean(), 3) if len(iou) else np.nan
        out.append(d)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RUN_DIR, "summary_weak_scores.csv"), index=False)
    print("\n" + "=" * 90)
    print("RadVLM 22 findings -- WEAK scoring vs Chest ImaGenome gold attributes")
    print("=" * 90)
    print(out.to_string(index=False))
    print("\nsens/spec: grouped prompt present/absent vs gold.  *_hit_pct: predicted box centre "
          "inside a gold region for that finding (gold-present images only).  *_region_iou: "
          "IoU vs rectangle enclosing those regions -- coarse, not lesion IoU.")


if __name__ == "__main__":
    main()
