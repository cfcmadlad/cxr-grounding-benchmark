"""
compare_results.py
------------------
Reads results/results.json and prints a markdown table comparing all six models
across every metric, bolding the best value in each column. Saves the table to
results/comparison_table.md. Models that have not finished yet are shown as
"pending" and simply omitted from the best-value calculation, so this can be run
safely after every single job (including before any job has completed).

Usage:
    python compare_results.py
"""

import os
import sys
import json

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison_table.md")

# Fixed model order (and the full set we always show, even if pending).
MODEL_ORDER = [
    "Grounding DINO",
    "BioViL-T",
    "MAIRA-2",
    "CheXagent",
    "RadVLM",
    "BiomedParse",
]

# (metric key, column header, higher_is_better or None if not ranked)
METRIC_COLUMNS = [
    ("iou_mean",       "IoU mean",  True),
    ("iou_median",     "IoU median", True),
    ("recall_at_0.1",  "R@0.1",     True),
    ("recall_at_0.25", "R@0.25",    True),
    ("recall_at_0.5",  "R@0.5",     True),
    ("map_at_0.5",     "mAP@0.5",   True),
    ("num_samples",    "N",         None),
    ("num_failed",     "Failed",    False),
]

_INT_KEYS = {"num_samples", "num_failed"}

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False


def _default_entry():
    return {"run_date": "", "status": "pending",
            "metrics": {k: None for k, _, _ in METRIC_COLUMNS}}


def load_results():
    """Load results.json (shared lock). Returns a dict keyed by model name."""
    if not os.path.exists(RESULTS_JSON):
        return {name: _default_entry() for name in MODEL_ORDER}
    with open(RESULTS_JSON, "r") as f:
        if _HAVE_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                print("WARNING: results.json is not valid JSON; showing all pending.",
                      file=sys.stderr)
                data = {}
        finally:
            if _HAVE_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    if not isinstance(data, dict):
        data = {}
    # Ensure every expected model is present.
    for name in MODEL_ORDER:
        data.setdefault(name, _default_entry())
    return data


def _fmt(key, val):
    if val is None:
        return "-"
    if key in _INT_KEYS:
        try:
            return str(int(val))
        except (TypeError, ValueError):
            return str(val)
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return str(val)


def _best_values(data):
    """For each ranked metric column, the best numeric value across complete models."""
    best = {}
    for key, _, higher in METRIC_COLUMNS:
        if higher is None:
            continue
        vals = []
        for name in MODEL_ORDER:
            entry = data.get(name, {})
            if entry.get("status") != "complete":
                continue
            v = (entry.get("metrics") or {}).get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            best[key] = max(vals) if higher else min(vals)
    return best


def build_table(data):
    best = _best_values(data)
    headers = ["Model", "Status", "Run date"] + [h for _, h, _ in METRIC_COLUMNS]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for name in MODEL_ORDER:
        entry = data.get(name, _default_entry())
        status = entry.get("status", "pending")
        run_date = entry.get("run_date", "") or "-"
        metrics = entry.get("metrics") or {}
        row = [name, status, run_date]
        for key, _, higher in METRIC_COLUMNS:
            val = metrics.get(key) if status == "complete" else None
            cell = _fmt(key, val)
            if (higher is not None and val is not None and key in best
                    and abs(float(val) - best[key]) < 1e-9 and cell != "—"):
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_results()
    table = build_table(data)

    n_complete = sum(1 for n in MODEL_ORDER if data.get(n, {}).get("status") == "complete")
    header = (f"# CXR Grounding Benchmark — Model Comparison\n\n"
              f"{n_complete}/{len(MODEL_ORDER)} models complete.\n\n")

    print("\n" + table + "\n")

    with open(COMPARISON_MD, "w") as f:
        f.write(header)
        f.write(table)
        f.write("\n")
    print(f"Saved comparison table to {COMPARISON_MD}")


if __name__ == "__main__":
    main()
