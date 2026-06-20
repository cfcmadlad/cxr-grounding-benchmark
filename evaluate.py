"""
evaluate.py
-----------
Aggregates results from all model scripts into:
  1. Per-region IoU table across all models (main results table for the paper)
  2. mAP@0.5 per model (matches RadVLM paper's reported metric)
  3. Per-region Dice table (for BiomedParse and any mask-based results)
  4. Bar charts comparing all models per region
  5. Overall summary CSV

Run this AFTER all model scripts have completed.
Each model script saves a results_summary.csv in its output directory.
This script reads all of them and produces a unified comparison.

Usage:
    python evaluate.py

No arguments needed — all paths are configured in the CONFIG block below.
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_OUTPUT_DIRS = {
    "Grounding DINO": "./outputs/grounding_dino",
    "BioViL-T":       "./outputs/biovilt",
    "MAIRA-2":        "./outputs/maira2",
    "CheXagent":      "./outputs/chexagent",
    "RadVLM":         "./outputs/radvlm",
    "BiomedParse":    "./outputs/biomedparse",
}
EVAL_OUTPUT_DIR = "./outputs/evaluation"
# ──────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

REGIONS = [
    "right lung",
    "left lung",
    "cardiac silhouette",
    "mediastinum",
    "trachea",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "right hilar region",
    "left hilar region",
    "right costophrenic angle",
    "left costophrenic angle",
]

# ── LOAD ALL RESULTS ──────────────────────────────────────────────────────────

def load_model_results(model_name, output_dir):
    csv_path = Path(output_dir) / "results_summary.csv"
    if not csv_path.exists():
        print(f"  WARNING: {model_name} — results_summary.csv not found at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["model"] = model_name
    return df

print("Loading model results...")
all_dfs = []
for model_name, output_dir in MODEL_OUTPUT_DIRS.items():
    df = load_model_results(model_name, output_dir)
    if df is not None:
        all_dfs.append(df)
        print(f"  {model_name}: {len(df)} rows loaded")
    else:
        print(f"  {model_name}: SKIPPED (no results found)")

if not all_dfs:
    raise RuntimeError(
        "No model results found. Run the model scripts first, then re-run evaluate.py."
    )

combined = pd.concat(all_dfs, ignore_index=True)
models_available = combined["model"].unique().tolist()
print(f"\nModels available: {models_available}")

# ── mAP @ IoU 0.5 ────────────────────────────────────────────────────────────

def compute_map_at_05(df):
    """
    mAP@0.5: fraction of detections with IoU >= 0.5.
    Matches the metric used in the RadVLM paper (Table 4).
    Only considers rows where a prediction was made (detected=True)
    and a gold box exists.
    """
    valid = df[df["detected"] == True].copy()
    valid = valid[valid["gold_box"].notna()]
    valid = valid[valid["iou"].notna()]
    if len(valid) == 0:
        return 0.0
    return (valid["iou"] >= 0.5).mean() * 100

# ── PER-REGION IoU TABLE ──────────────────────────────────────────────────────

print("\nBuilding per-region IoU table...")
iou_rows = []
for model in models_available:
    model_df = combined[combined["model"] == model]
    row = {"Model": model}
    for region in REGIONS:
        region_df = model_df[model_df["region"] == region]
        mean_iou = region_df["iou"].dropna().mean()
        row[region] = round(mean_iou, 3) if not np.isnan(mean_iou) else None
    row["Mean IoU"] = round(
        model_df["iou"].dropna().mean(), 3
    ) if model_df["iou"].dropna().shape[0] > 0 else None
    row["mAP@0.5"] = round(compute_map_at_05(model_df), 1)
    iou_rows.append(row)

iou_table = pd.DataFrame(iou_rows).set_index("Model")

# Print nicely
print("\n" + "="*80)
print("PER-REGION IoU TABLE")
print("="*80)
print(iou_table.to_string())

iou_table.to_csv(Path(EVAL_OUTPUT_DIR) / "per_region_iou_table.csv")
print(f"\nSaved: {EVAL_OUTPUT_DIR}/per_region_iou_table.csv")

# ── PER-REGION DICE TABLE (for BiomedParse) ───────────────────────────────────

if "dice" in combined.columns and combined["dice"].notna().any():
    print("\nBuilding per-region Dice table...")
    dice_rows = []
    for model in models_available:
        model_df = combined[combined["model"] == model]
        if "dice" not in model_df.columns or model_df["dice"].isna().all():
            continue
        row = {"Model": model}
        for region in REGIONS:
            region_df = model_df[model_df["region"] == region]
            mean_dice = region_df["dice"].dropna().mean() if "dice" in region_df.columns else None
            row[region] = round(mean_dice, 3) if mean_dice is not None and not np.isnan(mean_dice) else None
        row["Mean Dice"] = round(model_df["dice"].dropna().mean(), 3) if model_df["dice"].dropna().shape[0] > 0 else None
        dice_rows.append(row)

    if dice_rows:
        dice_table = pd.DataFrame(dice_rows).set_index("Model")
        print("\n" + "="*80)
        print("PER-REGION DICE TABLE")
        print("="*80)
        print(dice_table.to_string())
        dice_table.to_csv(Path(EVAL_OUTPUT_DIR) / "per_region_dice_table.csv")
        print(f"\nSaved: {EVAL_OUTPUT_DIR}/per_region_dice_table.csv")

# ── OVERALL SUMMARY TABLE ─────────────────────────────────────────────────────

print("\nBuilding overall summary...")
summary_rows = []
for model in models_available:
    model_df = combined[combined["model"] == model]
    row = {
        "Model":          model,
        "N Images":       model_df["image_id"].nunique(),
        "Detection Rate": f"{model_df['detected'].mean()*100:.1f}%",
        "Mean IoU":       round(model_df["iou"].dropna().mean(), 3)
                          if model_df["iou"].dropna().shape[0] > 0 else None,
        "mAP@0.5 (%)":   round(compute_map_at_05(model_df), 1),
    }
    if "dice" in model_df.columns and model_df["dice"].notna().any():
        row["Mean Dice"] = round(model_df["dice"].dropna().mean(), 3)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("Model")
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
print(summary_df.to_string())
summary_df.to_csv(Path(EVAL_OUTPUT_DIR) / "overall_summary.csv")
print(f"\nSaved: {EVAL_OUTPUT_DIR}/overall_summary.csv")

# ── BAR CHART: mAP@0.5 PER MODEL ─────────────────────────────────────────────

print("\nGenerating bar charts...")

map_vals   = [r["mAP@0.5 (%)"] for r in summary_rows]
model_names = [r["Model"] for r in summary_rows]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(model_names, map_vals, color=colors[:len(model_names)], edgecolor="white", width=0.6)
ax.set_title("mAP@IoU0.5 by Model — Chest ImaGenome Anatomical Region Grounding", fontsize=12)
ax.set_ylabel("mAP@0.5 (%)")
ax.set_ylim(0, 100)
ax.axhline(y=85.3, color="gray", linestyle="--", linewidth=1, label="RadVLM reported (85.3%)")
ax.legend(fontsize=9)
for bar, val in zip(bars, map_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
map_chart_path = Path(EVAL_OUTPUT_DIR) / "map_by_model.png"
plt.savefig(map_chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {map_chart_path}")

# ── BAR CHART: PER-REGION IoU, ALL MODELS ────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 6))
n_models  = len(models_available)
n_regions = len(REGIONS)
x         = np.arange(n_regions)
bar_width = 0.8 / n_models

for i, model in enumerate(models_available):
    model_df = combined[combined["model"] == model]
    iou_per_region = []
    for region in REGIONS:
        region_df = model_df[model_df["region"] == region]
        mean_iou = region_df["iou"].dropna().mean()
        iou_per_region.append(mean_iou if not np.isnan(mean_iou) else 0)
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x + offset, iou_per_region, bar_width,
           label=model, color=colors[i % len(colors)], alpha=0.85)

ax.set_title("Per-Region Mean IoU by Model — Chest ImaGenome Anatomical Grounding", fontsize=12)
ax.set_ylabel("Mean IoU")
ax.set_xticks(x)
ax.set_xticklabels(
    [r.replace(" ", "\n") for r in REGIONS],
    fontsize=7, rotation=45, ha="right"
)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", fontsize=8)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
region_chart_path = Path(EVAL_OUTPUT_DIR) / "per_region_iou_by_model.png"
plt.savefig(region_chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {region_chart_path}")

# ── BAR CHART: PER-REGION MEAN IoU HEATMAP STYLE ─────────────────────────────

pivot = combined.groupby(["model", "region"])["iou"].mean().unstack("region")
pivot = pivot.reindex(columns=REGIONS)
pivot = pivot.reindex(index=models_available)

fig, ax = plt.subplots(figsize=(18, 4))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n_regions))
ax.set_xticklabels(REGIONS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(models_available)))
ax.set_yticklabels(models_available, fontsize=9)
ax.set_title("Per-Region IoU Heatmap (green = high IoU)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Mean IoU")

for i in range(len(models_available)):
    for j in range(n_regions):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if 0.3 < val < 0.8 else "white")

plt.tight_layout()
heatmap_path = Path(EVAL_OUTPUT_DIR) / "iou_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {heatmap_path}")

# ── DONE ──────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"All outputs saved to: {EVAL_OUTPUT_DIR}/")
print("  per_region_iou_table.csv   — main results table")
print("  overall_summary.csv        — per-model summary")
print("  map_by_model.png           — mAP@0.5 bar chart")
print("  per_region_iou_by_model.png — per-region grouped bar chart")
print("  iou_heatmap.png            — IoU heatmap across all models and regions")
if "dice" in combined.columns and combined["dice"].notna().any():
    print("  per_region_dice_table.csv  — Dice scores (BiomedParse)")
