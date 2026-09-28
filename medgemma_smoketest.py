"""
medgemma_smoketest.py
----------------------
Standalone smoke test for MedGemma 1.5 (4B) before writing the full
medgemma15_medsam.py pipeline. Verifies:
  1. Model + processor load from HF (gated repo, needs HF_TOKEN).
  2. A real gold CXR image can be loaded.
  3. The model can be prompted for a bounding box of a named anatomical region.
  4. The RAW output format is printed so the box parser can be written to match
     reality, not assumptions.
"""
import os
import sys
import json
import time

os.environ.setdefault("HF_HOME", "/home/manik/pranjali/Aditya_project/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/manik/pranjali/Aditya_project/.cache/huggingface")

sys.path.insert(0, "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed")
from cxr_common import load_config, load_gold_annotations, find_image_path, load_image

cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR"])
GOLD_CSV = cfg["GOLD_CSV"]
IMAGE_DIR = cfg["IMAGE_DIR"]

print("=" * 70)
print("STEP 1: imports")
t0 = time.time()
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
import transformers
print("transformers", transformers.__version__)
print("torch file:", torch.__file__)
print(f"[{time.time()-t0:.1f}s]")

print("=" * 70)
print("STEP 2: AutoConfig")
from transformers import AutoConfig
MODEL_ID = "google/medgemma-1.5-4b-it"
config = AutoConfig.from_pretrained(MODEL_ID)
print("config class:", type(config).__name__)
print(f"[{time.time()-t0:.1f}s]")

print("=" * 70)
print("STEP 3: load processor + model")
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(MODEL_ID)
print("processor loaded:", type(processor).__name__)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model.eval()
print("model loaded:", type(model).__name__)
print(f"[{time.time()-t0:.1f}s]")

print("=" * 70)
print("STEP 4: load one real gold image")
gold = load_gold_annotations(GOLD_CSV)
image_ids = list(gold.keys())
print("first 3 image ids:", image_ids[:3])
image_id = image_ids[0]
img_path = find_image_path(IMAGE_DIR, image_id)
print("image path:", img_path)
pil_img = load_image(img_path)
print("image size (W,H):", pil_img.size)
print("gold boxes for this image:", gold[image_id])

print("=" * 70)
print("STEP 5: prompt for a bounding box")

region = "right lung"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": (
                f'Detect the "{region}" in this chest X-ray image. '
                f'Return the bounding box as a JSON list with one object: '
                f'[{{"box_2d": [y0, x0, y1, x1], "label": "{region}"}}]. '
                f'Output ONLY the JSON, nothing else.'
            )},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

input_len = inputs["input_ids"].shape[-1]
print("input_len:", input_len)
print("pixel_values shape:", inputs.get("pixel_values").shape if "pixel_values" in inputs else "NONE")

with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)

gen_tokens = gen[0][input_len:]
response = processor.decode(gen_tokens, skip_special_tokens=True)
print("=" * 70)
print("RAW RESPONSE for region='%s':" % region)
print(repr(response))
print("=" * 70)

# Try to parse it as JSON to see the box convention.
try:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    parsed = json.loads(cleaned)
    print("PARSED JSON:", parsed)
except Exception as e:
    print("JSON parse failed:", e)

print(f"TOTAL TIME [{time.time()-t0:.1f}s]")
print("SMOKETEST DONE")
