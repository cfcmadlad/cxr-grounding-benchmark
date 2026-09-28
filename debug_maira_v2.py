"""Standalone MAIRA-2 grounding diagnostic.

Bypasses maira2_medsam.maira_ground_phrase (which raises TypeError before it can
print anything) and calls the processor with its real signature:

    adjust_box_for_original_image_size(box, width, height) -> NORMALISED box

Tests several phrase formats on a few real gold images and prints, for each:
  - the raw decoded text
  - what convert_output_to_plaintext_or_grounded_sequence() returns
  - the normalised box, and the pixel box after scaling by (W, H)

Read-only: writes nothing outside stdout.
"""
import glob
import os
import sys
import traceback

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

IMG_DIR = "/home/manik/pranjali/Aditya_project/gold_images/gold_images/gold_images"
N_IMAGES = 3

PHRASES = [
    "findings in the right lung",          # what the benchmark currently sends
    "right lung",                          # bare anatomical region
    "The right lung is clear.",            # a real finding sentence (MAIRA-2's training form)
    "Opacity in the right lower lung zone.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch:", torch.__version__, "device:", DEVICE, flush=True)

print("Loading MAIRA-2 from cache...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/maira-2", trust_remote_code=True, torch_dtype=torch.float16
).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
print("loaded.", flush=True)

# Show the real signature, so the report has it straight from the installed code.
import inspect
print()
print("SIGNATURE adjust_box_for_original_image_size:",
      inspect.signature(processor.adjust_box_for_original_image_size))
print("SIGNATURE format_and_preprocess_phrase_grounding_input:",
      inspect.signature(processor.format_and_preprocess_phrase_grounding_input))
print(flush=True)

paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:N_IMAGES]
print("images under test:", [os.path.basename(p) for p in paths])

for path in paths:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    print()
    print("=" * 78)
    print("IMAGE", os.path.basename(path), "size=", (W, H))
    print("=" * 78, flush=True)

    for phrase in PHRASES:
        print()
        print("--- phrase:", repr(phrase))
        try:
            inputs = processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=img, phrase=phrase, return_tensors="pt"
            ).to(DEVICE, torch.float16)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, use_cache=True)

            plen = inputs["input_ids"].shape[-1]
            raw = processor.decode(out[0][plen:], skip_special_tokens=True)
            print("    raw decoded  :", repr(raw))

            # skip_special_tokens=True can strip <obj>/<box>; also show the
            # un-stripped decode so we can see whether the tags survive.
            raw_keep = processor.decode(out[0][plen:], skip_special_tokens=False)
            print("    raw (keep sp):", repr(raw_keep[:300]))

            try:
                pred = processor.convert_output_to_plaintext_or_grounded_sequence(raw)
                print("    parsed       :", repr(pred))
            except Exception as e:
                print("    parsed       : RAISED", type(e).__name__, e)
                pred = None

            if isinstance(pred, list) and pred:
                _, boxes = pred[0]
                if boxes:
                    nb = boxes[0]
                    print("    norm box (model 518-crop space):", nb)
                    adj = processor.adjust_box_for_original_image_size(nb, W, H)
                    print("    adjusted (still NORMALISED 0-1):", adj)
                    px = [int(adj[0] * W), int(adj[1] * H),
                          int(adj[2] * W), int(adj[3] * H)]
                    print("    -> PIXEL box                   :", px)
                    print("    what current code does (int() on normalised):",
                          [int(v) for v in adj], "  <-- collapses to zeros")
                else:
                    print("    no boxes in first grounded phrase")
            else:
                print("    plain-text response, no grounded phrases")
        except Exception:
            print("    EXCEPTION:")
            traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

print()
print("DONE")
