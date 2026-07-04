"""
setup_check.py
--------------
Pre-flight verification for the Sharanga HPC run. Checks that everything the
model scripts need is present and correct BEFORE any Slurm job is submitted.

Each check prints "PASS" or "FAIL: <specific reason>". The script exits with
code 1 if any check fails, so it can be used as a CI / job-dependency gate.

Usage:
    python setup_check.py
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("CXR_CONFIG", os.path.join(REPO_ROOT, "config.yaml"))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

GOLD_REQUIRED_COLUMNS = ["image_id", "bbox_name", "x", "y", "w", "h"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".dcm", ".dicom")
CONDA_ENVS = ["gdino", "biovilt", "chexagent", "maira2", "radvlm", "biomedparse"]

_failures = 0


def report(name, ok, detail=""):
    global _failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        _failures += 1
    line = f"[{status}] {name}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok


def _dir_writable(path):
    """True if we can create ``path`` (if needed) and write a temp file in it."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".setup_check_write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def load_config_raw():
    """Load config.yaml directly (no exit-on-error, unlike cxr_common.load_config)."""
    try:
        import yaml
    except ImportError:
        report("PyYAML installed", False, "pip install pyyaml")
        return None
    report("PyYAML installed", True)
    if not os.path.exists(CONFIG_PATH):
        report("config.yaml exists", False, CONFIG_PATH)
        return None
    report("config.yaml exists", True, CONFIG_PATH)
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        report("config.yaml parses", False, str(e))
        return None
    if not isinstance(cfg, dict):
        report("config.yaml parses", False, "top level is not a mapping")
        return None
    report("config.yaml parses", True)
    return cfg


def check_gold_csv(cfg):
    path = cfg.get("GOLD_CSV")
    if not path:
        report("GOLD_CSV set", False, "empty/missing in config.yaml")
        return
    if not os.path.isfile(path):
        report("GOLD_CSV exists", False, path)
        return
    report("GOLD_CSV exists", True, path)
    try:
        import pandas as pd
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
    except Exception as e:
        report("GOLD_CSV readable", False, str(e))
        return
    missing = [c for c in GOLD_REQUIRED_COLUMNS if c not in cols]
    report(
        "GOLD_CSV has columns " + ",".join(GOLD_REQUIRED_COLUMNS),
        not missing,
        f"missing {missing}; found {cols}" if missing else f"found {cols}",
    )


def check_image_dir(cfg):
    path = cfg.get("IMAGE_DIR")
    if not path:
        report("IMAGE_DIR set", False, "empty/missing in config.yaml")
        return
    if not os.path.isdir(path):
        report("IMAGE_DIR exists", False, path)
        return
    report("IMAGE_DIR exists", True, path)
    # Find at least one image file (recursive), stopping early.
    found = None
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(IMAGE_EXTS):
                found = os.path.join(root, fn)
                break
        if found:
            break
    report("IMAGE_DIR has >=1 image", found is not None,
           found if found else f"no {IMAGE_EXTS} files under {path}")


def check_radvlm(cfg):
    path = cfg.get("RADVLM_PATH")
    if not path:
        report("RADVLM_PATH set", False, "empty/missing in config.yaml")
        return
    if not os.path.isdir(path):
        report("RADVLM_PATH exists", False, path)
        return
    report("RADVLM_PATH exists", True, path)
    try:
        empty = len(os.listdir(path)) == 0
    except Exception as e:
        report("RADVLM_PATH not empty", False, str(e))
        return
    report("RADVLM_PATH not empty", not empty,
           "directory is empty" if empty else "")


def check_biomedparse(cfg):
    path = cfg.get("BIOMEDPARSE_REPO")
    if not path:
        report("BIOMEDPARSE_REPO set", False, "empty/missing in config.yaml")
        return
    report("BIOMEDPARSE_REPO exists", os.path.isdir(path), path)


def check_writable_dirs(cfg):
    out = cfg.get("OUTPUT_DIR")
    if out:
        report("OUTPUT_DIR writable", _dir_writable(out), out)
    else:
        report("OUTPUT_DIR set", False, "empty/missing in config.yaml")
    hf = cfg.get("HF_HOME")
    if hf:
        report("HF_HOME writable", _dir_writable(hf), hf)
    report("results/ writable", _dir_writable(RESULTS_DIR), RESULTS_DIR)


def check_conda_envs():
    try:
        out = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        for env in CONDA_ENVS:
            report(f"conda env '{env}'", False, f"could not run 'conda env list': {e}")
        return
    names = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
    for env in CONDA_ENVS:
        report(f"conda env '{env}'", env in names,
               "not found" if env not in names else "")


def main():
    print("=" * 70)
    print("CXR grounding benchmark - setup verification")
    print("=" * 70)

    cfg = load_config_raw()
    if cfg is not None:
        check_gold_csv(cfg)
        check_image_dir(cfg)
        check_radvlm(cfg)
        check_biomedparse(cfg)
        check_writable_dirs(cfg)
    check_conda_envs()

    print("=" * 70)
    if _failures:
        print(f"RESULT: {_failures} check(s) FAILED - fix before submitting jobs.")
        sys.exit(1)
    print("RESULT: all checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
