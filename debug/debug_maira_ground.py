import sys
sys.path.insert(0, "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed")
from maira2_medsam import maira_ground_phrase, maira_processor, maira_model, load_image
import glob

img_dir = "/home/manik/pranjali/Aditya_project/gold_images/gold_images/gold_images"
sample_path = sorted(glob.glob(img_dir + "/*.jpg"))[0]
print("Using image:", sample_path)

pil_img = load_image(sample_path)
phrase = "findings in the right lung"
box, raw_text = maira_ground_phrase(pil_img, phrase)

print("=== RAW DECODED TEXT ===")
print(repr(raw_text))
print("=== PARSED BOX ===")
print(box)
