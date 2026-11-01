import argparse
import csv
import gzip
import importlib
import json
import os
import random
import sys
import time

R = "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed"
O = os.path.join(R, "outputs_final")
SCRIPT = {
    "medgemma": "medgemma15_medsam",
    "maira2": "maira2_medsam",
    "chexagent": "chexagent_medsam",
    "radvlm": "radvlm_medsam",
    "chex": "chex_medsam",
}
UNIFORM = {
    "radvlm": "Locate the {region} in this chest X-ray. Provide the bounding box as [x1, y1, x2, y2] with coordinates normalized between 0 and 1.",
    "maira2": "phrase grounding, phrase: findings in the {region}",
    "chexagent": 'Perform phrase grounding for "{region}" in this chest X-ray. Provide the bounding box as [x1, y1, x2, y2] in pixel coordinates.',
    "medgemma": 'Detect the "{region}" in this chest X-ray image. Return the bounding box as a JSON list with one object: [{"box_2d": [y0, x0, y1, x1], "label": "{region}"}]. Output ONLY the JSON, nothing else.',
    "chex": "ChEX anatomy prompts, one batched forward pass per image",
}
NATIVE = {
    "radvlm": "Where is the {region} located on the image?",
    "maira2": "phrase grounding, phrase: {region}",
    "chexagent": "Please locate the following phrase: {region}",
    "medgemma": 'Where is the {region}? Return the bounding box as a JSON list with one object: [{"box_2d": [y0, x0, y1, x1], "label": "{region}"}]. Output ONLY the JSON, nothing else.',
}


def conv(b):
    return None if b is None else [float(v) for v in b]


def medgemma_native(m, torch, img, region):
    text = NATIVE["medgemma"].replace("{region}", region)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    inputs = m.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(m.medgemma_model.device, dtype=torch.bfloat16)
    n = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        gen = m.medgemma_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    raw = m.processor.decode(gen[0][n:], skip_special_tokens=True)
    return m.parse_box_from_response(raw, img.size[0], img.size[1]), raw


def chexagent_native(m, torch, img, region):
    prompt = NATIVE["chexagent"].replace("{region}", region)
    inputs = m.processor(
        images=[img], text=" USER: <s>" + prompt + " ASSISTANT: <s>", return_tensors="pt"
    ).to(device=m.DEVICE, dtype=torch.float16)
    with torch.no_grad():
        out = m.chexagent_model.generate(**inputs, generation_config=m.generation_config)[0]
    raw = m.processor.tokenizer.decode(out, skip_special_tokens=True)
    return m.parse_box_from_response(raw, img.size[0], img.size[1]), raw


def predict(m, torch, model, native, img, region):
    if model == "medgemma":
        if native:
            return medgemma_native(m, torch, img, region)
        return m.medgemma_ground_region(img, region)
    if model == "maira2":
        return m.maira_ground_phrase(img, region if native else "findings in the " + region)
    if model == "chexagent":
        if native:
            return chexagent_native(m, torch, img, region)
        return m.chexagent_ground_phrase(img, region)
    tpl = NATIVE["radvlm"] if native else UNIFORM["radvlm"]
    raw, _ = m.inference_radvlm(m.radvlm_model, m.radvlm_processor, img, tpl.replace("{region}", region))
    return m.parse_box_from_response(raw, img.size[0], img.size[1]), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SCRIPT))
    ap.add_argument("--mode", default="uniform", choices=["uniform", "flip", "native"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shards", default="0")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--subset", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "native" and a.model == "chex":
        sys.exit(4)

    os.environ["CXR_CONFIG"] = os.path.join(O, "config_%s.yaml" % os.environ.get("CFG", "default"))
    os.environ.setdefault("HF_HOME", "/home/manik/.cache/huggingface")
    sys.path[:0] = [R, os.path.join(R, "models")]
    import numpy as np
    import torch
    import transformers
    from PIL import ImageOps

    class Stub:
        def to(self, *x, **k):
            return self

        def eval(self):
            return self

    for n in ("SamModel", "SamProcessor"):
        c = getattr(transformers, n, None)
        if c is not None:
            setattr(c, "from_pretrained", classmethod(lambda cls, *x, **k: Stub()))

    if not torch.cuda.is_available():
        sys.exit(3)
    from cxr_common import clamp_box, load_image, validate_box

    m = importlib.import_module(SCRIPT[a.model])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    native = a.mode == "native"
    prompt = (NATIVE if native else UNIFORM).get(a.model, UNIFORM[a.model])

    with gzip.open(os.path.join(O, "queries.csv.gz"), "rt") as f:
        all_rows = list(csv.DictReader(f))
    if a.subset:
        with open(os.path.join(O, a.subset + "_query_ids.txt")) as f:
            keep = set(f.read().splitlines())
        all_rows = [r for r in all_rows if r["query_id"] in keep]
    images = list(dict.fromkeys(r["image_id"] for r in all_rows))
    parts = np.array_split(np.arange(len(images)), a.nshards)

    for shard in [int(s) for s in a.shards.split(",")]:
        out = os.path.join(O, "shards", "%s_%s_%s_%03dof%03d.jsonl.gz" % (a.tag, a.model, a.mode, shard, a.nshards))
        if os.path.exists(out):
            continue
        mine = set(images[int(i)] for i in parts[shard])
        rows = [r for r in all_rows if r["image_id"] in mine]
        if a.limit:
            rows = rows[: a.limit]
        groups = {}
        for r in rows:
            groups.setdefault(r["image_id"], []).append(r)

        part = out + ".part"
        t_start = time.perf_counter()
        done = 0
        errors = 0
        with gzip.open(part, "wt", encoding="utf-8") as f:
            for iid, qs in groups.items():
                img = load_image(qs[0]["path"])
                res = None
                ierr = ""
                if img is None:
                    ierr = "load_image returned None"
                else:
                    if a.mode == "flip":
                        img = ImageOps.mirror(img)
                    if a.model == "chex":
                        try:
                            res = m.chex_ground_regions(img)
                        except Exception as e:
                            ierr = repr(e)[:200]
                for q in qs:
                    region = q["region"]
                    t0 = time.perf_counter()
                    raw, box, err = "", None, ierr
                    if img is not None and not err:
                        try:
                            if a.model == "chex":
                                box = res.get(region)
                            else:
                                box, raw = predict(m, torch, a.model, native, img, region)
                        except Exception as e:
                            err = repr(e)[:200]
                    final = None
                    if box is not None:
                        W, H = img.size
                        c = clamp_box(box, W, H)
                        final = c if validate_box(c, W, H) else None
                    row = {
                        "query_id": q["query_id"], "image_id": iid, "region": region, "mode": a.mode,
                        "W": int(q["W"]), "H": int(q["H"]), "raw": raw, "box_raw": conv(box),
                        "box": conv(final), "err": err, "sec": round(time.perf_counter() - t0, 3),
                    }
                    f.write(json.dumps(row) + "\n")
                    done += 1
                    errors += 1 if err else 0
                    if done % 50 == 0:
                        f.flush()
                    if done >= 5 and errors == done:
                        sys.exit(2)
            meta = {"_meta": {
                "tag": a.tag, "model": a.model, "mode": a.mode, "shard": shard, "nshards": a.nshards,
                "n": done, "errors": errors, "seconds": round(time.perf_counter() - t_start, 1),
                "prompt": prompt, "torch": torch.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0), "seed": 0,
            }}
            f.write(json.dumps(meta) + "\n")
        os.replace(part, out)


main()
