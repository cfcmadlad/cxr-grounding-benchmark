"""
radvlm_22findings.py
--------------------
RadVLM abnormality grounding for 22 VinDr-CXR findings on the gold MIMIC images.
Predictions only -- scoring is a separate CPU step (models/radvlm22_score.py),
because these images have no gold boxes for findings.

Same speed-ups as radvlm_29regions.py (image prefix encoded once per image, its KV
cache shared by every prompt, prompts decoded as one batch, fast path checked against
plain generate() before the long run). New here: finished rows are dropped from the
batch, so one long answer does not keep 21 short ones computing.

Prompts per image (22), worded exactly as in RadVLM's training data
(github.com/uzh-dqbm-cmi/RadVLM, radvlm/data/create_instructions.py):

  * 21 x single-finding grounding, VinDr_CXR_Single_Label_Dataset template
        "Where is the {label} located in this Chest X-ray?"
    RadVLM was trained on this ONLY for findings that are present, so it will
    almost always return a box. Read these as "where would it be", not "is it there".
  * 1 x grouped detection, VinDr_CXR_Dataset template
        "Are abnormalities present on this Chest X-ray? Where exactly can they be found?"
    Trained with negatives ("No abnormalities ... detected"), so this is the
    model's real present/absent answer for all 22 labels, incl. "No finding".

Labels are VinDr's own class names lowercased the way RadVLM's training code does it
("Enlarged PA" -> "enlarged pa", "ILD" stays "ILD"), not the long display names.

Outputs (OUTPUT_DIR/radvlm22/):
  predictions.csv     one row per (image, finding): boxes from the grounding prompt and
                      from the grouped prompt (pixel coords, JSON lists)
  raw_responses.jsonl all 22 raw answers per image
Restartable and locked exactly like radvlm_29regions.py.

Env knobs: SELFTEST=tokens, REPARSE=1 (rebuild predictions.csv from raw_responses.jsonl,
CPU only), VALIDATE_IMAGES (default 3), MAX_NEW_TOKENS (default 128,
grounding prompts), MAX_NEW_TOKENS_GROUPED (default 512), REGION_CHUNK (default auto),
RETRY_FAILED=1.
"""

import os
import re
import sys
import json
import copy
import time
import atexit
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")

from cxr_common import (
    load_config, setup_hf_home, load_image, find_image_path,
    validate_box, clamp_box, get_logger,
)

log = get_logger("radvlm22")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR", "RADVLM_PATH"])
setup_hf_home(cfg)

RADVLM_PATH = cfg["RADVLM_PATH"]
GOLD_CSV = cfg["GOLD_CSV"]
IMAGE_DIR = cfg["IMAGE_DIR"]
OUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "radvlm22")
MAX_IMAGES = cfg.get("MAX_IMAGES")

SELFTEST = os.environ.get("SELFTEST", "")
VALIDATE_IMAGES = int(os.environ.get("VALIDATE_IMAGES", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
MAX_NEW_TOKENS_GROUPED = int(os.environ.get("MAX_NEW_TOKENS_GROUPED", "512"))
REGION_CHUNK = os.environ.get("REGION_CHUNK", "auto")
RETRY_FAILED = os.environ.get("RETRY_FAILED", "") == "1"

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.cache_utils import DynamicCache

# -- FINDINGS -----------------------------------------------------------------
# (display name from the 22-finding list, VinDr class name as in VinDr's CSVs)
FINDINGS = [
    ("Aortic enlargement", "Aortic enlargement"),
    ("Atelectasis", "Atelectasis"),
    ("Calcification", "Calcification"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Clavicle fracture", "Clavicle fracture"),
    ("Consolidation", "Consolidation"),
    ("Edema", "Edema"),
    ("Emphysema", "Emphysema"),
    ("Enlarged pulmonary artery", "Enlarged PA"),
    ("Interstitial lung disease", "ILD"),
    ("Lung cavity", "Lung cavity"),
    ("Lung cyst", "Lung cyst"),
    ("Lung opacity", "Lung Opacity"),
    ("Mediastinal shift", "Mediastinal shift"),
    ("Nodule/mass", "Nodule/Mass"),
    ("Pleural effusion", "Pleural effusion"),
    ("Pleural thickening", "Pleural thickening"),
    ("Pneumothorax", "Pneumothorax"),
    ("Pulmonary fibrosis", "Pulmonary fibrosis"),
    ("Rib fracture", "Rib fracture"),
    ("Other lesion", "Other lesion"),
    ("No finding", "No finding"),
]
assert len(FINDINGS) == 22
LOCALIZABLE = [f for f in FINDINGS if f[0] != "No finding"]  # 21 grounding prompts
GROUPED = "__grouped__"


def trained_label(vindr_name):
    """RadVLM's own lowercasing rule (create_instructions.generate_instruction_location)."""
    if vindr_name[0].isupper() and not vindr_name.isupper():
        return vindr_name.lower()
    return vindr_name


# grouped answers name findings with the same lowercasing; map them back
_LABEL_TO_DISPLAY = {trained_label(v).lower(): d for d, v in FINDINGS}
_LABEL_TO_DISPLAY["infiltration"] = "Infiltration"  # VinDr label not in our 22; kept, never scored
_LABELS_LONGEST_FIRST = sorted(_LABEL_TO_DISPLAY, key=len, reverse=True)

GROUNDING_TEMPLATE = "Where is the {} located in this Chest X-ray?"
GROUPED_PROMPT = ("Are abnormalities present on this Chest X-ray? "
                  "Where exactly can they be found?")

# prompt slots, in order: 21 grounding prompts, then the grouped one
SLOTS = [d for d, _ in LOCALIZABLE] + [GROUPED]
SLOT_TEXT = [GROUNDING_TEMPLATE.format(trained_label(v)) for _, v in LOCALIZABLE] + [GROUPED_PROMPT]
SLOT_MAXTOK = [MAX_NEW_TOKENS] * len(LOCALIZABLE) + [MAX_NEW_TOKENS_GROUPED]
VAL_SLOTS = ["Cardiomegaly", "Pleural effusion", "Nodule/mass", "Enlarged pulmonary artery", GROUPED]


# -- PARSING ------------------------------------------------------------------
_NUM = r"(-?\d*\.?\d+)"  # negatives allowed, clamped to 0 (the 29-region parser lost 379 spine boxes)
_SEP = r"[\s,]+"
_BOX_RE = re.compile(r"\[\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*\]")
_NO_ABN_RE = re.compile(r"\bno\b[^.;]*\b(abnormalit|lesion|finding)"
                        r"|clear of any|does not display any|clean bill of health", re.I)


def parse_boxes(text, img_w, img_h):
    """Every normalized [x1, y1, x2, y2] in the text -> list of pixel boxes."""
    boxes = []
    for m in _BOX_RE.finditer(text or ""):
        v = [min(max(float(m.group(i)), 0.0), 1.0) for i in range(1, 5)]
        b = [int(round(v[0] * img_w)), int(round(v[1] * img_h)),
             int(round(v[2] * img_w)), int(round(v[3] * img_h))]
        if b[2] > b[0] and b[3] > b[1]:
            b = clamp_box(b, img_w, img_h)
            if validate_box(b, img_w, img_h):
                boxes.append(b)
    return boxes


def parse_grouped(text, img_w, img_h):
    """Grouped detection answer -> ({display_name: [boxes]}, said_no_abnormality, unknown_names).

    Training format: "<prefix> a cardiomegaly located at the coordinates [..]; an
    aortic enlargement located at the coordinates [..] and [..]."
    """
    found, unknown = {}, []
    for part in (text or "").split(";"):
        m = re.search(r"\s+(?:is\s+|are\s+)?located at", part, re.I)
        if not m:
            continue
        # the label is whatever known name the text before "located at" ends with; the
        # answer prefix varies ("...identified: a", "The analysis reveals an", ...)
        head = part[:m.start()].strip().lower()
        disp = next((_LABEL_TO_DISPLAY[k] for k in _LABELS_LONGEST_FIRST
                     if re.search(r"(?:^|[\s:])" + re.escape(k) + r"$", head)), None)
        if disp is None:
            unknown.append(head[-60:])
            continue
        found.setdefault(disp, []).extend(parse_boxes(part, img_w, img_h))
    said_none = not found and bool(_NO_ABN_RE.search(text or ""))
    return found, said_none, unknown


# -- PROMPTS / TOKENS ---------------------------------------------------------
IMAGE_MARKER = "<image>\n"


def chat(processor, text):
    conv = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]}]
    return processor.apply_chat_template(conv, add_generation_prompt=True)


def build_prompt_parts(processor):
    fulls = [chat(processor, t) for t in SLOT_TEXT]
    k = fulls[0].index(IMAGE_MARKER) + len(IMAGE_MARKER)
    prefix = fulls[0][:k]
    assert all(f.startswith(prefix) for f in fulls), "prompts do not share the image prefix"
    return fulls, prefix, [f[k:] for f in fulls]


# -- FAST INFERENCE -----------------------------------------------------------

@torch.inference_mode()
def generate_fast(model, tok, enc, suffix_ids, chunk, eos_id):
    """Responses (list of str, SLOTS order) for one image."""
    dev = model.device
    P = enc["input_ids"].shape[1]

    cache = DynamicCache()
    model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
          pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"],
          past_key_values=cache, use_cache=True, num_logits_to_keep=1)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    responses = [None] * len(SLOTS)
    all_idx = list(range(len(SLOTS)))
    chunks = [all_idx[i:i + chunk] for i in range(0, len(all_idx), chunk)]
    can_prune = hasattr(DynamicCache, "batch_select_indices")

    for ci, idxs in enumerate(chunks):
        c = cache if ci == len(chunks) - 1 else copy.deepcopy(cache)
        B = len(idxs)
        c.batch_repeat_interleave(B)

        lens_list = [len(suffix_ids[i]) for i in idxs]
        Lmax = max(lens_list)
        ids_cpu = torch.full((B, Lmax), pad_id, dtype=torch.long)
        for r, i in enumerate(idxs):
            ids_cpu[r, :lens_list[r]] = torch.tensor(suffix_ids[i], dtype=torch.long)
        ids = ids_cpu.to(dev)
        lens = torch.tensor(lens_list, device=dev)

        ar = torch.arange(Lmax, device=dev)[None, :]
        mask = torch.ones((B, P + Lmax), dtype=torch.long, device=dev)
        mask[:, P:] = (ar < lens[:, None]).long()
        out = model(input_ids=ids, attention_mask=mask, position_ids=(P + ar).expand(B, -1),
                    past_key_values=c, use_cache=True)
        cur = out.logits[torch.arange(B, device=dev), lens - 1].float().argmax(-1)
        del out

        # greedy decode; rows that finish are removed from the batch (and the cache)
        rows = list(idxs)                      # slot index of each live row
        maxtok = torch.tensor([SLOT_MAXTOK[i] for i in idxs], device=dev)
        gen = {i: [] for i in idxs}
        t = 0
        while rows:
            fin = (cur == eos_id) | (t >= maxtok)
            cur_l = cur.tolist()
            for r, i in enumerate(rows):
                if not fin[r]:
                    gen[i].append(cur_l[r])
            if bool(fin.any()):
                keep = (~fin).nonzero().flatten()
                if keep.numel() == 0:
                    break
                if can_prune:
                    c.batch_select_indices(keep)
                    rows = [rows[k] for k in keep.tolist()]
                    mask, lens, maxtok, cur = mask[keep], lens[keep], maxtok[keep], cur[keep]
                    fin = fin[keep]
            if not can_prune and bool(fin.all()):
                break
            mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=torch.long, device=dev)], 1)
            out = model(input_ids=cur[:, None], attention_mask=mask,
                        position_ids=(P + lens + t)[:, None], past_key_values=c, use_cache=True)
            nxt = out.logits[:, -1].float().argmax(-1)
            del out
            # without pruning, finished rows keep emitting eos
            cur = torch.where(fin, torch.full_like(nxt, eos_id), nxt) if not can_prune else nxt
            if not can_prune:
                maxtok = torch.where(fin, torch.zeros_like(maxtok), maxtok)
            t += 1

        for i in idxs:
            responses[i] = tok.decode(gen[i], skip_special_tokens=True).strip()
        del c
    del cache
    return responses


@torch.inference_mode()
def generate_baseline(model, processor, img, slot):
    """Reference path: one plain generate() call."""
    i = SLOTS.index(slot)
    inputs = processor(images=img, text=chat(processor, SLOT_TEXT[i]),
                       return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=SLOT_MAXTOK[i], do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip()


def boxes_agree(a, b, W, H, tol=0.02):
    if len(a) != len(b):
        return False
    return all(max(abs(p[0] - q[0]) / W, abs(p[2] - q[2]) / W,
                   abs(p[1] - q[1]) / H, abs(p[3] - q[3]) / H) <= tol
               for p, q in zip(a, b))


# -- RESUME / LOCK (same scheme as radvlm_29regions.py) -----------------------
N_ROWS = len(FINDINGS)


def repair_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    counts = df["image_id"].value_counts()
    bad = set(counts[counts != N_ROWS].index)
    if bad:
        log.warning("dropping %d partially written image(s) from %s", len(bad), csv_path)
        df[~df["image_id"].isin(bad)].to_csv(csv_path, index=False)


def completed_images(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["image_id"], dtype=str)
    counts = df["image_id"].value_counts()
    return set(counts[counts >= N_ROWS].index)


LOCK_DIR = os.path.join(OUT_DIR, ".run.lock")


def _job_alive(jobid):
    if not jobid.isdigit():
        return False
    try:
        r = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%T"],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return True
    return r.returncode == 0 and r.stdout.strip() != ""


def release_lock():
    shutil.rmtree(LOCK_DIR, ignore_errors=True)


def acquire_lock():
    me = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for _ in range(2):
        try:
            os.mkdir(LOCK_DIR)
        except FileExistsError:
            try:
                with open(os.path.join(LOCK_DIR, "owner")) as f:
                    owner = f.read().strip()
            except Exception:
                owner = ""
            if owner and owner != me and _job_alive(owner):
                log.info("another job (%s) is already running this; exiting. "
                         "(stale lock? delete %s)", owner, LOCK_DIR)
                return False
            shutil.rmtree(LOCK_DIR, ignore_errors=True)
            continue
        with open(os.path.join(LOCK_DIR, "owner"), "w") as f:
            f.write(me)
        atexit.register(release_lock)
        return True
    return False


# -- OUTPUT -------------------------------------------------------------------
PRED_FIELDS = ["image_id", "finding", "vindr_label", "img_w", "img_h",
               "n_ground_boxes", "ground_boxes", "grouped_present", "grouped_boxes"]


def rows_for_image(iid, W, H, responses):
    by_slot = dict(zip(SLOTS, responses))
    found, said_none, _ = parse_grouped(by_slot[GROUPED], W, H)
    rows = []
    for disp, vindr in FINDINGS:
        if disp == "No finding":
            gb, present, grp = [], said_none, []
        else:
            gb = parse_boxes(by_slot[disp], W, H)
            present, grp = disp in found, found.get(disp, [])
        rows.append({"image_id": iid, "finding": disp, "vindr_label": vindr,
                     "img_w": W, "img_h": H, "n_ground_boxes": len(gb),
                     "ground_boxes": json.dumps(gb), "grouped_present": present,
                     "grouped_boxes": json.dumps(grp)})
    return rows


def append_rows(csv_path, rows):
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    pd.DataFrame(rows, columns=PRED_FIELDS).to_csv(csv_path, mode="a", header=new, index=False)


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    present = df["grouped_present"].astype(str).str.lower().eq("true")
    out = (df.assign(present=present, has_box=df["n_ground_boxes"] > 0)
             .groupby("finding", sort=False)
             .agg(n_images=("image_id", "size"),
                  grounding_box_rate_pct=("has_box", lambda s: round(100 * s.mean(), 1)),
                  mean_boxes=("n_ground_boxes", lambda s: round(s.mean(), 2)),
                  grouped_says_present_pct=("present", lambda s: round(100 * s.mean(), 1)))
             .reset_index())
    out.to_csv(os.path.join(OUT_DIR, "summary_predictions.csv"), index=False)
    print("\n" + "=" * 72)
    print("RadVLM, 22 findings -- PREDICTIONS ONLY (no gold boxes on these images)")
    print("=" * 72)
    print(out.to_string(index=False))
    print("\ngrounding_box_rate: grounding prompt returned >=1 box (expected ~100%, "
          "the model was only trained on present findings)")
    print("grouped_says_present: the grouped detection prompt listed the finding "
          "(this is the model's present/absent call)")
    print(f"Images: {df['image_id'].nunique()}.  Score with models/radvlm22_score.py")


# -- MAIN ---------------------------------------------------------------------

def selftest_tokens(processor, image_ids):
    fulls, prefix, suffixes = build_prompt_parts(processor)
    tok = processor.tokenizer
    suffix_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in suffixes]
    img = None
    for iid in image_ids:
        p = find_image_path(IMAGE_DIR, iid)
        if p is not None:
            img = load_image(p)
            if img is not None:
                break
    assert img is not None, "no loadable image found"
    pre_ids = processor(images=img, text=prefix, return_tensors="pt")["input_ids"][0].tolist()
    print(f"image size {img.size}, prefix tokens {len(pre_ids)}")
    bad = 0
    for i, slot in enumerate(SLOTS):
        full = processor(images=img, text=fulls[i], return_tensors="pt")["input_ids"][0].tolist()
        ok = full == pre_ids + suffix_ids[i]
        bad += (not ok)
        print(f"  [{'OK ' if ok else 'BAD'}] {SLOT_TEXT[i]}")
    print("TOKEN SELFTEST:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)")
    # parser checks on RadVLM's training answer formats
    W = H = 1000
    f, none, unk = parse_grouped(
        "Yes, the following abnormalities are identified: an aortic enlargement located at the "
        "coordinates [0.45, 0.2, 0.6, 0.4]; a nodule/mass located at the coordinates "
        "[0.1, 0.1, 0.2, 0.2] and [0.7, 0.5, 0.8, 0.6]; an ILD located at the coordinates "
        "[0.1, 0.3, 0.4, 0.8].", W, H)
    ok_p = (set(f) == {"Aortic enlargement", "Nodule/mass", "Interstitial lung disease"}
            and len(f["Nodule/mass"]) == 2 and not none and not unk)
    f2, none2, _ = parse_grouped("No abnormalities or lesions are detected on this Chest X-ray.", W, H)
    ok_p = ok_p and not f2 and none2
    # prefixes without a colon, seen in the smoke test
    f3, _, unk3 = parse_grouped(
        "The analysis reveals an aortic enlargement located at the coordinates [0.55, 0.38, 0.66, "
        "0.53]; a cardiomegaly located at the coordinates [0.44, 0.66, 0.84, 0.86].", W, H)
    f4, _, unk4 = parse_grouped("The findings include a consolidation located at the coordinates "
                                "[0.1, 0.2, 0.3, 0.4].", W, H)
    ok_p = (ok_p and set(f3) == {"Aortic enlargement", "Cardiomegaly"} and not unk3
            and set(f4) == {"Consolidation"} and not unk4)
    print("PARSER SELFTEST:", "PASS" if ok_p else f"FAIL {f} {none} {unk} | {f2} {none2}")
    return bad == 0 and ok_p


def reparse():
    """Rebuild predictions.csv from raw_responses.jsonl (CPU; after a parser fix)."""
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    csv_path = os.path.join(OUT_DIR, "predictions.csv")
    rows, unknown = [], {}
    for line in open(raw_path):
        d = json.loads(line)
        resp = [d["responses"][s] for s in SLOTS]
        rows += rows_for_image(d["image_id"], d["img_w"], d["img_h"], resp)
        for name in parse_grouped(resp[-1], d["img_w"], d["img_h"])[2]:
            unknown[name] = unknown.get(name, 0) + 1
    tmp = csv_path + ".tmp"
    pd.DataFrame(rows, columns=PRED_FIELDS).to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    print(f"re-parsed {len(rows) // N_ROWS} images from {raw_path}")
    if unknown:
        print("Unmapped names in grouped answers (count):", unknown)
    summarize(csv_path)


def main():
    if os.environ.get("REPARSE") == "1":
        reparse()
        return
    gold_ids =(pd.read_csv(GOLD_CSV, usecols=["image_id"])["image_id"].astype(str).str.strip()
                .str.replace(r"\.dcm$", "", case=False, regex=True))
    image_ids = list(dict.fromkeys(gold_ids))  # unique, file order
    processor = AutoProcessor.from_pretrained(RADVLM_PATH)
    tok = processor.tokenizer

    if SELFTEST == "tokens":
        sys.exit(0 if selftest_tokens(processor, image_ids) else 1)

    if not torch.cuda.is_available():
        log.error("no CUDA device -- refusing to run the full job on CPU")
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "predictions.csv")
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    failed_path = os.path.join(OUT_DIR, "failed_images.txt")

    repair_csv(csv_path)
    done = completed_images(csv_path)
    failed = set()
    if os.path.exists(failed_path) and not RETRY_FAILED:
        failed = set(open(failed_path).read().split())

    if MAX_IMAGES:
        image_ids = image_ids[:int(MAX_IMAGES)]
    paths, missing = {}, []
    for iid in image_ids:
        if iid in done or iid in failed:
            continue
        p = find_image_path(IMAGE_DIR, iid)
        (paths.__setitem__(iid, p) if p else missing.append(iid))
    todo = [i for i in image_ids if i in paths]
    log.info("%d gold image ids | %d already done | %d previously failed | %d file not found | %d to do",
             len(image_ids), len(done), len(failed), len(missing), len(todo))
    if missing:
        log.warning("image file not found for %d id(s): %s", len(missing), " ".join(missing[:50]))

    if not todo:
        log.info("nothing to do")
        if os.path.exists(csv_path):
            summarize(csv_path)
        return
    if not acquire_lock():
        return

    t0 = time.time()
    print("Loading RadVLM...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        RADVLM_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    model.eval()
    eos_id = model.generation_config.eos_token_id
    eos_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id
    assert eos_id == tok.convert_tokens_to_ids("<|im_end|>"), "unexpected eos id"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    chunk = int(REGION_CHUNK) if REGION_CHUNK != "auto" else (22 if total_gb >= 70 else 11 if total_gb >= 38 else 6)
    print(f"RadVLM loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)} "
          f"({total_gb:.0f} GB); prompt chunk = {chunk}; "
          f"row pruning = {hasattr(DynamicCache, 'batch_select_indices')}")

    _, prefix_str, suffix_strs = build_prompt_parts(processor)
    suffix_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in suffix_strs]

    def prepare(iid):
        try:
            img = load_image(paths[iid])
            if img is None:
                return iid, None, None, "load_image returned None"
            return iid, img, processor(images=img, text=prefix_str, return_tensors="pt"), None
        except Exception as e:  # noqa: BLE001
            return iid, None, None, f"{type(e).__name__}: {e}"

    # -- validation (fresh runs only): fast path vs plain generate() ----------
    if not done and VALIDATE_IMAGES > 0:
        print(f"\nVALIDATION on {VALIDATE_IMAGES} image(s): fast path vs plain generate()")
        agree = total = 0
        for iid in todo[:VALIDATE_IMAGES]:
            _, img, enc, err = prepare(iid)
            if img is None:
                continue
            W, H = img.size
            t1 = time.time()
            fast = generate_fast(model, tok, enc.to("cuda", torch.float16), suffix_ids, chunk, eos_id)
            t_fast = time.time() - t1
            for slot in VAL_SLOTS:
                base = generate_baseline(model, processor, img, slot)
                f = fast[SLOTS.index(slot)]
                match = boxes_agree(parse_boxes(f, W, H), parse_boxes(base, W, H), W, H)
                agree += match
                total += 1
                print(f"  [{'MATCH' if match else 'DIFF '}] {iid[:13]} {slot}\n"
                      f"      fast: {f!r}\n      base: {base!r}")
            print(f"  (fast path, all 22 prompts of that image: {t_fast:.1f}s)")
        rate = agree / max(total, 1)
        print(f"VALIDATION: {agree}/{total} answers agree ({100 * rate:.0f}%)")
        if total == 0 or rate < 0.7:
            log.error("fast path disagrees with plain generate() -- aborting before the long run")
            sys.exit(3)

    # -- main loop with 1-ahead image preprocessing --------------------------
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(prepare, todo[0])
    n_done = n_fail = 0
    unknown_names = {}
    t_start = time.time()
    for n, iid in enumerate(todo):
        _, img, enc, err = fut.result()
        if n + 1 < len(todo):
            fut = ex.submit(prepare, todo[n + 1])
        if img is None:
            log.error("image_id=%s: %s -- skipped", iid, err)
            open(failed_path, "a").write(iid + "\n")
            n_fail += 1
            continue
        try:
            responses = generate_fast(model, tok, enc.to("cuda", torch.float16),
                                      suffix_ids, chunk, eos_id)
        except Exception:  # noqa: BLE001
            log.exception("image_id=%s: inference failed -- skipped", iid)
            open(failed_path, "a").write(iid + "\n")
            n_fail += 1
            torch.cuda.empty_cache()
            continue

        W, H = img.size
        append_rows(csv_path, rows_for_image(iid, W, H, responses))
        with open(raw_path, "a") as f:
            f.write(json.dumps({"image_id": iid, "img_w": W, "img_h": H,
                                "responses": dict(zip(SLOTS, responses))}) + "\n")
        for name in parse_grouped(responses[-1], W, H)[2]:
            unknown_names[name] = unknown_names.get(name, 0) + 1

        n_done += 1
        if n_done <= 3 or n_done % 10 == 0:
            per = (time.time() - t_start) / n_done
            print(f"[{n_done + len(done)}/{len(image_ids)}] {iid[:13]}  {per:.1f}s/img  "
                  f"ETA {per * (len(todo) - n - 1) / 60:.0f} min")
            if n_done <= 3:
                for slot in ["Cardiomegaly", "Pneumothorax", GROUPED]:
                    print(f"    {slot}: {responses[SLOTS.index(slot)]}")

    print(f"\nFinished: {n_done} images done this run, {n_fail} failed, "
          f"{(time.time() - t_start) / 60:.1f} min")
    if unknown_names:
        print("Finding names in grouped answers that did not map to the 22 (count):", unknown_names)
    summarize(csv_path)


if __name__ == "__main__":
    main()
