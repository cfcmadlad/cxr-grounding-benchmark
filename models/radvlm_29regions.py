"""
radvlm_29regions.py
-------------------
RadVLM grounding for ALL 29 Chest ImaGenome anatomical regions, optimised for speed.

Why a separate script (radvlm_medsam.py is left untouched -- it produced the
validated 15-region results):

  * radvlm_medsam.py makes one independent generate() call per region, so the same
    ~7K image tokens are re-encoded every time, and it also runs MedSAM and saves a
    matplotlib PNG per region. Extended to 29 regions that is roughly 25-35 h for
    959 images.
  * Here the image prefix is run through the model ONCE per image, its KV cache is
    reused by all 29 region prompts, and the 29 short region suffixes are decoded
    together as one batch (custom greedy loop). MedSAM / PNGs are skipped -- scoring
    is box IoU only, masks were never used in any metric.

Why the prefix reuse is exact: RadVLM's chat template renders the image FIRST
("<|im_start|>user\\n<image>\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"), so the
image tokens are an identical prefix for every region prompt. Reusing that prefix's
KV cache does not change what the model sees. Every fresh run additionally checks
the fast path against the plain generate() path on VALIDATE_IMAGES images and aborts
(exit code 3) if they disagree.

Regions / scoring
  * 29 official Chest ImaGenome names. The 15 regions of the original benchmark use
    the SAME prompt wording as before (hilar regions are still asked as "hilar
    region"), so those rows are directly comparable with the 15-region run.
  * Gold boxes come from the raw gold CSV (no renaming). 26 of the 29 names exist
    there. "spine" and "abdomen" were never manually annotated in the gold standard
    (Chest ImaGenome paper, Table 4): predictions are recorded, IoU stays empty.
  * "mediastinum" is not annotated directly either. Following the paper's footnote
    it is DERIVED here as the bounding rectangle enclosing the "upper mediastinum"
    and "cardiac silhouette" gold boxes (our reading of "derived from", not a
    formula given in the paper). "upper mediastinum" is scored against its own box.

Restartable: results are appended per image; a re-run skips finished images. A lock
directory stops two jobs from running at once, so the same job can safely be
submitted to several partitions.

Env knobs: SELFTEST=tokens (CPU-only checks, then exit), VALIDATE_IMAGES (default 3),
MAX_NEW_TOKENS (default 96), REGION_CHUNK (default auto), RETRY_FAILED=1.
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

# -- CONFIG + HF cache (must precede heavy imports) ---------------------------
from cxr_common import (
    load_config, setup_hf_home, load_image, find_image_path,
    validate_box, clamp_box, compute_iou,
    result_row, write_result_row, BOX_FIELDS, get_logger,
)

log = get_logger("radvlm29")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR", "RADVLM_PATH"])
setup_hf_home(cfg)

RADVLM_PATH = cfg["RADVLM_PATH"]
GOLD_CSV = cfg["GOLD_CSV"]
IMAGE_DIR = cfg["IMAGE_DIR"]
OUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "radvlm29")
MAX_IMAGES = cfg.get("MAX_IMAGES")

SELFTEST = os.environ.get("SELFTEST", "")
VALIDATE_IMAGES = int(os.environ.get("VALIDATE_IMAGES", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "96"))
REGION_CHUNK = os.environ.get("REGION_CHUNK", "auto")
RETRY_FAILED = os.environ.get("RETRY_FAILED", "") == "1"

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.cache_utils import DynamicCache

# -- REGIONS ------------------------------------------------------------------
# The 15 regions of the original benchmark first (same order), then the 14 new ones.
REGIONS_PRIOR15 = [
    "right lung", "left lung", "cardiac silhouette", "mediastinum", "trachea",
    "right upper lung zone", "right mid lung zone", "right lower lung zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone",
    "right hilar structures", "left hilar structures",
    "right costophrenic angle", "left costophrenic angle",
]
REGIONS_NEW14 = [
    "upper mediastinum", "aortic arch", "carina", "svc", "cavoatrial junction",
    "right atrium", "right apical zone", "left apical zone",
    "right hemidiaphragm", "left hemidiaphragm", "right clavicle", "left clavicle",
    "spine", "abdomen",
]
REGIONS = REGIONS_PRIOR15 + REGIONS_NEW14
assert len(REGIONS) == 29 and len(set(REGIONS)) == 29

# Wording used in the ORIGINAL 15-region prompts, kept for comparability.
PROMPT_NAME = {
    "right hilar structures": "right hilar region",
    "left hilar structures": "left hilar region",
}
# Validation uses a spread of region types (incl. one with no gold at all).
VAL_REGIONS = ["right lung", "cardiac silhouette", "mediastinum",
               "right hilar structures", "left costophrenic angle",
               "aortic arch", "spine"]


def region_prompt(region):
    name = PROMPT_NAME.get(region, region)
    return (f"Locate the {name} in this chest X-ray. "
            f"Provide the bounding box as [x1, y1, x2, y2] "
            f"with coordinates normalized between 0 and 1.")


# -- GOLD ---------------------------------------------------------------------

def load_gold_29(csv_path):
    """{image_id: {region_name: [x1,y1,x2,y2]}} straight from the raw gold CSV.

    Names are kept as in the file (no renaming) except that "mediastinum" is added
    as the enclosing rectangle of "upper mediastinum" + "cardiac silhouette".
    """
    df = pd.read_csv(csv_path)
    ids = df["image_id"].astype(str).str.strip().str.replace(r"\.dcm$", "", case=False, regex=True)
    names = df["bbox_name"].astype(str).str.lower().str.strip()
    gold = {}
    for iid, name, x1, y1, x2, y2 in zip(
            ids, names, df["original_x1"], df["original_y1"], df["original_x2"], df["original_y2"]):
        gold.setdefault(iid, {})[name] = [float(x1), float(y1), float(x2), float(y2)]
    for g in gold.values():
        um, cs = g.get("upper mediastinum"), g.get("cardiac silhouette")
        if "mediastinum" not in g and um is not None and cs is not None:
            g["mediastinum"] = [min(um[0], cs[0]), min(um[1], cs[1]),
                                max(um[2], cs[2]), max(um[3], cs[3])]
    return gold


# -- BOX PARSING (identical patterns to radvlm_medsam.py) ----------------------
_NUM = r"(\d*\.?\d+)"
_SEP = r"[\s,]+"
_BOX_PATTERNS = [
    r"\[\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*\]",
    r"\(\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*\)",
    r"<box>\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*</box>",
    r"x1\s*[:=]\s*" + _NUM + r".*?y1\s*[:=]\s*" + _NUM +
    r".*?x2\s*[:=]\s*" + _NUM + r".*?y2\s*[:=]\s*" + _NUM,
    _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM,
]


def parse_box_from_response(response, img_w, img_h):
    """Normalized [x1,y1,x2,y2] in free text -> pixel box, or None."""
    if not response:
        return None
    for pattern in _BOX_PATTERNS:
        match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        try:
            vals = [float(match.group(i)) for i in range(1, 5)]
        except (ValueError, IndexError):
            continue
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        x1 = int(round(vals[0] * img_w))
        y1 = int(round(vals[1] * img_h))
        x2 = int(round(vals[2] * img_w))
        y2 = int(round(vals[3] * img_h))
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return None


# -- PROMPTS / TOKENS ---------------------------------------------------------
IMAGE_MARKER = "<image>\n"


def build_prompt_parts(processor):
    """(full_prompts, shared_prefix_str, per-region_suffix_strs).

    The prefix ends right after "<image>\\n"; everything region-specific is in the
    suffix, so the (expanded) image tokens are shared by all regions.
    """
    fulls = []
    for region in REGIONS:
        conv = [{"role": "user", "content": [
            {"type": "text", "text": region_prompt(region)}, {"type": "image"}]}]
        fulls.append(processor.apply_chat_template(conv, add_generation_prompt=True))
    k = fulls[0].index(IMAGE_MARKER) + len(IMAGE_MARKER)
    prefix = fulls[0][:k]
    assert all(f.startswith(prefix) for f in fulls), "prompts do not share the image prefix"
    return fulls, prefix, [f[k:] for f in fulls]


# -- FAST INFERENCE -----------------------------------------------------------

@torch.inference_mode()
def ground_regions_fast(model, tok, enc, suffix_ids, chunk, eos_id):
    """Responses (list of str, REGIONS order) for one image.

    enc: processor output for the shared prefix, already on device / fp16.
    """
    dev = model.device
    prefix_ids = enc["input_ids"]
    P = prefix_ids.shape[1]

    # 1) prefix (image) pass -- once per image
    cache = DynamicCache()
    model(input_ids=prefix_ids, attention_mask=enc["attention_mask"],
          pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"],
          past_key_values=cache, use_cache=True, num_logits_to_keep=1)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    responses = [None] * len(REGIONS)
    all_idx = list(range(len(REGIONS)))
    chunks = [all_idx[i:i + chunk] for i in range(0, len(all_idx), chunk)]

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

        # 2) region suffixes, right-padded; padded slots are masked out for good
        ar = torch.arange(Lmax, device=dev)[None, :]
        mask = torch.ones((B, P + Lmax), dtype=torch.long, device=dev)
        mask[:, P:] = (ar < lens[:, None]).long()
        pos = (P + ar).expand(B, -1)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    past_key_values=c, use_cache=True)
        cur = out.logits[torch.arange(B, device=dev), lens - 1].float().argmax(-1)
        del out

        # 3) batched greedy decode (positions follow each row's true length)
        gen = torch.full((B, MAX_NEW_TOKENS), eos_id, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(MAX_NEW_TOKENS):
            is_eos = cur == eos_id
            gen[:, t] = torch.where(done | is_eos, torch.full_like(cur, eos_id), cur)
            done = done | is_eos
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=1)
            out = model(input_ids=cur[:, None], attention_mask=mask,
                        position_ids=(P + lens + t)[:, None],
                        past_key_values=c, use_cache=True)
            cur = out.logits[:, -1].float().argmax(-1)
            del out

        for r, toks in enumerate(gen.cpu().tolist()):
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            responses[idxs[r]] = tok.decode(toks, skip_special_tokens=True).strip()
        del c
    del cache
    return responses


@torch.inference_mode()
def ground_region_baseline(model, processor, img, region):
    """Reference path: one plain generate() call (same as radvlm_medsam.py)."""
    conv = [{"role": "user", "content": [
        {"type": "text", "text": region_prompt(region)}, {"type": "image"}]}]
    full = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=img, text=full, return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip()


# -- RESUME / LOCK HELPERS ----------------------------------------------------

def repair_csv(csv_path):
    """Drop rows of images that were only partly written (crash mid-image)."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    counts = df["image_id"].value_counts()
    bad = set(counts[counts != len(REGIONS)].index)
    if bad:
        log.warning("dropping %d partially written image(s) from %s", len(bad), csv_path)
        df[~df["image_id"].isin(bad)].to_csv(csv_path, index=False)


def completed_images(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["image_id"], dtype=str)
    counts = df["image_id"].value_counts()
    return set(counts[counts >= len(REGIONS)].index)


LOCK_DIR = os.path.join(OUT_DIR, ".run.lock")


def _job_alive(jobid):
    """Conservative: any doubt => treat the lock holder as alive."""
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


# -- SUMMARY ------------------------------------------------------------------

def summarize(csv_path):
    df = pd.read_csv(csv_path)
    df["iou_num"] = pd.to_numeric(df["iou"], errors="coerce")
    df["det"] = df["detected"].astype(str).str.lower().eq("true")
    df["has_gold"] = df["gold_box"].notna() & (df["gold_box"].astype(str).str.len() > 0)
    rows = []
    for region in REGIONS:
        s = df[df["region"] == region]
        g = s[s["has_gold"]]
        rows.append({
            "region": region,
            "n_queries": len(s),
            "detection_rate_pct": round(100 * s["det"].mean(), 1) if len(s) else np.nan,
            "n_with_gold": len(g),
            "cond_mean_iou": round(g["iou_num"].mean(), 3) if g["iou_num"].notna().any() else np.nan,
            "uncond_mean_iou": round(g["iou_num"].fillna(0).mean(), 3) if len(g) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "summary_29regions.csv"), index=False)
    print("\n" + "=" * 72)
    print("SUMMARY (RadVLM, 29 regions)")
    print("=" * 72)
    print(out.to_string(index=False))
    scored = df[df["has_gold"]]
    print(f"\nImages: {df['image_id'].nunique()}   rows: {len(df)}")
    print(f"Overall (regions with gold, {scored['region'].nunique()} regions): "
          f"conditional IoU {scored['iou_num'].mean():.3f}, "
          f"unconditional IoU {scored['iou_num'].fillna(0).mean():.3f}")
    print("No gold (predictions only): spine, abdomen.  mediastinum gold is DERIVED "
          "(upper mediastinum + cardiac silhouette).")


# -- MAIN ---------------------------------------------------------------------

def selftest_tokens(processor, gold):
    """CPU-only: prefix+suffix token ids must equal the full processor path."""
    fulls, prefix, suffixes = build_prompt_parts(processor)
    tok = processor.tokenizer
    suffix_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in suffixes]
    img = None
    for iid in gold:
        p = find_image_path(IMAGE_DIR, iid)
        if p is not None:
            img = load_image(p)
            if img is not None:
                break
    assert img is not None, "no loadable image found"
    pre = processor(images=img, text=prefix, return_tensors="pt")
    pre_ids = pre["input_ids"][0].tolist()
    n_img_tok = int((pre["input_ids"] == processor.image_token_id).sum()) if hasattr(
        processor, "image_token_id") else -1
    print(f"image size {img.size}, prefix tokens {len(pre_ids)} (image tokens {n_img_tok}), "
          f"pixel_values {tuple(pre['pixel_values'].shape)}")
    bad = 0
    for i, region in enumerate(REGIONS):
        full = processor(images=img, text=fulls[i], return_tensors="pt")["input_ids"][0].tolist()
        mine = pre_ids + suffix_ids[i]
        ok = full == mine
        bad += (not ok)
        print(f"  [{'OK ' if ok else 'BAD'}] {region:26s} suffix {len(suffix_ids[i]):3d} tok")
    print("TOKEN SELFTEST:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)")
    print("eos id from tokenizer:", tok.convert_tokens_to_ids("<|im_end|>"))
    g = [gold[i] for i in list(gold)[:200]]
    from collections import Counter
    c = Counter(name for d in g for name in d)
    print("gold names in first 200 images:", {k: c[k] for k in REGIONS})
    return bad == 0


def main():
    gold = load_gold_29(GOLD_CSV)
    processor = AutoProcessor.from_pretrained(RADVLM_PATH)
    tok = processor.tokenizer

    if SELFTEST == "tokens":
        ok = selftest_tokens(processor, gold)
        sys.exit(0 if ok else 1)

    if not torch.cuda.is_available():
        log.error("no CUDA device -- refusing to run the full job on CPU")
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "results_summary.csv")
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    failed_path = os.path.join(OUT_DIR, "failed_images.txt")

    repair_csv(csv_path)
    done = completed_images(csv_path)
    failed = set()
    if os.path.exists(failed_path) and not RETRY_FAILED:
        failed = set(open(failed_path).read().split())

    image_ids = list(gold.keys())
    if MAX_IMAGES:
        image_ids = image_ids[:int(MAX_IMAGES)]
    paths, missing = {}, []
    for iid in image_ids:
        if iid in done or iid in failed:
            continue
        p = find_image_path(IMAGE_DIR, iid)
        (paths.__setitem__(iid, p) if p else missing.append(iid))
    todo = [i for i in image_ids if i in paths]
    log.info("%d gold images | %d already done | %d previously failed | %d file not found | %d to do",
             len(image_ids), len(done), len(failed), len(missing), len(todo))
    if missing:
        log.warning("image file not found for %d id(s) (skipped, as in all other runs)", len(missing))

    if not todo:
        log.info("nothing to do")
        if os.path.exists(csv_path):
            summarize(csv_path)
        return
    if not acquire_lock():
        return

    # -- model --------------------------------------------------------------
    t0 = time.time()
    print("Loading RadVLM...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        RADVLM_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    model.eval()
    eos_id = model.generation_config.eos_token_id
    eos_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id
    assert eos_id == tok.convert_tokens_to_ids("<|im_end|>"), "unexpected eos id"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    chunk = int(REGION_CHUNK) if REGION_CHUNK != "auto" else (29 if total_gb >= 70 else 14 if total_gb >= 38 else 8)
    print(f"RadVLM loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)} "
          f"({total_gb:.0f} GB); region chunk = {chunk}")

    _, prefix_str, suffix_strs = build_prompt_parts(processor)
    suffix_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in suffix_strs]

    def prepare(iid):
        try:
            img = load_image(paths[iid])
            if img is None:
                return iid, None, None, "load_image returned None"
            enc = processor(images=img, text=prefix_str, return_tensors="pt")
            return iid, img, enc, None
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
            fast = ground_regions_fast(model, tok, enc.to("cuda", torch.float16),
                                       suffix_ids, chunk, eos_id)
            t_fast = time.time() - t1
            for region in VAL_REGIONS:
                base = ground_region_baseline(model, processor, img, region)
                f = fast[REGIONS.index(region)]
                bf, bb = parse_box_from_response(f, W, H), parse_box_from_response(base, W, H)
                if bf is None or bb is None:
                    match = bf is None and bb is None
                else:
                    match = max(abs(bf[0] - bb[0]) / W, abs(bf[2] - bb[2]) / W,
                                abs(bf[1] - bb[1]) / H, abs(bf[3] - bb[3]) / H) <= 0.02
                agree += match
                total += 1
                print(f"  [{'MATCH' if match else 'DIFF '}] {iid[:13]} {region:24s}\n"
                      f"      fast: {f!r}\n      base: {base!r}")
            print(f"  (fast path for all 29 regions of that image: {t_fast:.1f}s)")
        rate = agree / max(total, 1)
        print(f"VALIDATION: {agree}/{total} boxes agree ({100 * rate:.0f}%)")
        if total == 0 or rate < 0.7:
            log.error("fast path disagrees with plain generate() -- aborting before the long run")
            sys.exit(3)

    # -- main loop with 1-ahead image preprocessing --------------------------
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(prepare, todo[0])
    n_done = n_fail = 0
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
            responses = ground_regions_fast(model, tok, enc.to("cuda", torch.float16),
                                            suffix_ids, chunk, eos_id)
        except Exception:  # noqa: BLE001
            log.exception("image_id=%s: inference failed -- skipped", iid)
            open(failed_path, "a").write(iid + "\n")
            n_fail += 1
            torch.cuda.empty_cache()
            continue

        W, H = img.size
        rows = []
        for region, resp in zip(REGIONS, responses):
            gold_box = gold[iid].get(region)
            pred = parse_box_from_response(resp, W, H)
            if pred is not None:
                pred = clamp_box(pred, W, H)
                if not validate_box(pred, W, H):
                    pred = None
            iou = compute_iou(pred, gold_box) if (pred is not None and gold_box is not None) else None
            rows.append(result_row(iid, region, iou, pred is not None, gold_box, pred))
        for row in rows:
            write_result_row(csv_path, row, BOX_FIELDS)
        with open(raw_path, "a") as f:
            f.write(json.dumps({"image_id": iid, "responses": dict(zip(REGIONS, responses))}) + "\n")

        n_done += 1
        torch.cuda.empty_cache()
        if n_done <= 3 or n_done % 10 == 0:
            el = time.time() - t_start
            per = el / n_done
            ious = [r["iou"] for r in rows if r["iou"] != ""]
            print(f"[{n_done + len(done)}/{len(image_ids)}] {iid[:13]}  "
                  f"{per:.1f}s/img  ETA {per * (len(todo) - n - 1) / 60:.0f} min  "
                  f"(this image: mean IoU {np.mean(ious):.3f} over {len(ious)} scored regions)")
            if n_done <= 2:
                for region, resp in list(zip(REGIONS, responses))[:3] + list(zip(REGIONS, responses))[-2:]:
                    print(f"    {region}: {resp}")

    print(f"\nFinished: {n_done} images done this run, {n_fail} failed, "
          f"{(time.time() - t_start) / 60:.1f} min")
    summarize(csv_path)


if __name__ == "__main__":
    main()
