import argparse
import json
import math
import os
import subprocess
import time

R = os.environ.get("AUTOPILOT_ROOT", "/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed")
S = os.path.join(R, "outputs_final")
USER = "manik"
MAX_JOBS = 8
MIN_FREE_INODES = 100000
MAX_ATTEMPTS = 2
EXPECTED = [
    ("j2", "chex", "flip", 1),
    ("j2", "maira2", "flip", 5),
    ("j1", "maira2", "native", 5),
    ("j2", "radvlm", "flip", 12),
    ("j1", "radvlm", "native", 12),
    ("j2", "chexagent", "flip", 10),
    ("j1", "chexagent", "native", 10),
    ("j3", "medgemma", "uniform", 15),
    ("j2", "medgemma", "flip", 15),
    ("j1", "medgemma", "native", 15),
]
MODELS = ("chex", "maira2", "radvlm", "chexagent", "medgemma")
SEC_PER_SHARD = {"radvlm": 1200, "medgemma": 700, "maira2": 3000, "chexagent": 5000, "chex": 2500}
COMBINED = {424050, 424051, 424052, 424053, 427231, 427232}
PARTS = ("gpu_h100_4", "gpu_h200_8")


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def read_jobs():
    jobs = {}
    path = os.path.join(S, "jobs.log")
    if not os.path.exists(path):
        return jobs
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) < 9 or not t[0].isdigit():
                continue
            jid = int(t[0])
            second = None
            for x in t[9:]:
                if x.startswith("+"):
                    second = x[1:]
            if second is None and jid in COMBINED:
                second = "native:j1"
            jobs[jid] = {
                "model": t[1], "mode": t[2], "tag": t[3], "nsh": int(t[4]),
                "shards": [int(s) for s in t[5].split("+")], "j0": t[6], "tl": t[7],
                "part": t[8], "second": second,
            }
    return jobs


def active_jobs(jobs):
    code, out, _ = run(["squeue", "-u", USER, "-h", "-o", "%i|%T"])
    act = {}
    for line in out.splitlines():
        a, _, st = line.partition("|")
        if a.isdigit() and int(a) in jobs:
            act[int(a)] = st
    return act


def shard_path(tag, model, mode, i, n):
    return os.path.join(S, "shards", "%s_%s_%s_%03dof%03d.jsonl.gz" % (tag, model, mode, i, n))


def missing(tag, model, mode, n):
    return [i for i in range(n) if not os.path.exists(shard_path(tag, model, mode, i, n))]


def covered(tag, model, mode, n, jobs, act):
    cov = set()
    for jid in act:
        j = jobs[jid]
        if j["model"] != model or j["nsh"] != n:
            continue
        if (j["tag"], j["mode"]) == (tag, mode) or j["second"] == "%s:%s" % (mode, tag):
            cov.update(j["shards"])
    return cov


def j0_missing(model):
    return not os.path.exists(os.path.join(S, "shards", "j0a_%s_uniform_000of001.jsonl.gz" % model))


def j0_covered(model, jobs, act):
    return any(jobs[j]["model"] == model and jobs[j]["j0"] == "1" for j in act)


def done_count(model):
    return sum(n - len(missing(tag, m, mode, n)) for tag, m, mode, n in EXPECTED if m == model)


def build_requests(jobs, act):
    reqs = []
    for model in MODELS:
        items = []
        for tag, m, mode, n in EXPECTED:
            if m != model:
                continue
            cov = covered(tag, m, mode, n, jobs, act)
            u = [i for i in missing(tag, m, mode, n) if i not in cov]
            if u:
                items.append([tag, mode, n, u])
        used = set()
        first = True
        for a in range(len(items)):
            if a in used:
                continue
            used.add(a)
            second = None
            for b in range(a + 1, len(items)):
                if b not in used and items[b][2] == items[a][2] and items[b][3] == items[a][3]:
                    used.add(b)
                    second = (items[b][1], items[b][0])
                    break
            tag, mode, n, u = items[a]
            need_j0 = first and j0_missing(model) and not j0_covered(model, jobs, act)
            first = False
            phases = 2 if second else 1
            sec = SEC_PER_SHARD[model] * len(u) * phases * 1.5 + 3600 + (1800 if need_j0 else 0)
            hours = min(22, max(1, int(math.ceil(sec / 3600.0))))
            reqs.append({
                "model": model, "mode": mode, "tag": tag, "nsh": n, "shards": u, "second": second,
                "j0": need_j0, "tl": "%02d:00:00" % hours,
            })
    return reqs


def pick_partition(model, jobs, act):
    if model == "chex":
        return "gpu_v100_2"
    load = {p: sum(1 for j in act if jobs[j]["part"] == p) for p in PARTS}
    return min(PARTS, key=lambda p: load[p])


def free_inodes():
    code, out, _ = run(["lfs", "quota", "-h", "/home/manik"])
    try:
        t = out.splitlines()[2].split()
        return int(t[7]) - int(t[5])
    except Exception:
        return None


def load_state():
    p = os.path.join(S, "autopilot_state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(os.path.join(S, "autopilot_state.json"), "w") as f:
        json.dump(state, f, indent=1)


def submit(req, part):
    env = dict(os.environ)
    if req["second"]:
        env["MODE2"], env["TAG2"] = req["second"]
    shards = "+".join(str(s) for s in req["shards"])
    cmd = ["bash", os.path.join(R, "hpc", "submit.sh"), req["model"], req["mode"], req["tag"],
           str(req["nsh"]), shards, "1" if req["j0"] else "0", req["tl"], part]
    code, out, err = run(cmd, env)
    jid = out.strip()
    if code or not jid.isdigit():
        return None, err.strip()
    tail = " +%s:%s" % req["second"] if req["second"] else ""
    with open(os.path.join(S, "jobs.log"), "a") as f:
        f.write("%s %s %s %s %d %s %s %s %s%s\n" % (
            jid, req["model"], req["mode"], req["tag"], req["nsh"], shards,
            "1" if req["j0"] else "0", req["tl"], part, tail))
    return int(jid), ""


def write_manifest(jobs, act, state):
    ids = sorted(jobs)
    states = {}
    if ids:
        code, out, _ = run(["sacct", "-j", ",".join(str(i) for i in ids), "-X", "-n", "-P",
                            "--format=JobID,State,Elapsed,ExitCode"])
        for line in out.splitlines():
            t = line.split("|")
            if len(t) == 4 and t[0].isdigit():
                states[int(t[0])] = {"state": t[1], "elapsed": t[2], "exit": t[3]}
    manifest = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": {str(i): dict(jobs[i], **states.get(i, {})) for i in ids},
        "progress": {"%s_%s_%s" % (tag, m, mode): {"done": n - len(missing(tag, m, mode, n)), "expected": n}
                     for tag, m, mode, n in EXPECTED},
        "blocked": [m for m, v in state.items() if v.get("blocked")],
    }
    with open(os.path.join(S, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    jobs = read_jobs()
    act = active_jobs(jobs)
    state = load_state()
    for model in MODELS:
        v = state.setdefault(model, {"attempts": 0, "done_at": 0})
        d = done_count(model)
        if d > v["done_at"]:
            v["attempts"] = 0
            v["blocked"] = False
        v["done_at"] = max(v["done_at"], d)
    print("active jobs:", {j: act[j] for j in sorted(act)})
    for tag, m, mode, n in EXPECTED:
        cov = sorted(covered(tag, m, mode, n, jobs, act))
        print("%s %s %s done %d of %d, missing %s, covered by active jobs %s" % (
            tag, m, mode, n - len(missing(tag, m, mode, n)), n, missing(tag, m, mode, n), cov))
    free = free_inodes()
    print("free inodes:", free)
    if free is not None and free < MIN_FREE_INODES:
        print("STOP: free inodes below %d, no submissions" % MIN_FREE_INODES)
        write_manifest(jobs, act, state)
        save_state(state)
        return
    reqs = build_requests(jobs, act)
    submitted = 0
    for r in reqs:
        v = state[r["model"]]
        if v["attempts"] >= MAX_ATTEMPTS:
            v["blocked"] = True
            print("BLOCKED %s: %d resubmissions made without progress, diagnose the logs" % (r["model"], v["attempts"]))
            continue
        if len(act) >= MAX_JOBS:
            print("job cap reached, not submitting:", r["model"], r["tag"], r["mode"], r["shards"])
            continue
        part = pick_partition(r["model"], jobs, act)
        line = "%s %s %s shards %s second %s j0 %s limit %s on %s" % (
            r["model"], r["tag"], r["mode"], r["shards"], r["second"], r["j0"], r["tl"], part)
        if not a.submit:
            print("DRY RUN would submit:", line)
            continue
        jid, err = submit(r, part)
        if jid is None:
            print("submit failed:", line, "|", err[:200])
            if "QOS" in err:
                break
            continue
        submitted += 1
        v["attempts"] += 1
        jobs[jid] = {"model": r["model"], "mode": r["mode"], "tag": r["tag"], "nsh": r["nsh"],
                     "shards": r["shards"], "j0": "1" if r["j0"] else "0", "tl": r["tl"], "part": part,
                     "second": "%s:%s" % r["second"] if r["second"] else None}
        act[jid] = "PENDING"
        print("submitted", jid, line)
    write_manifest(jobs, act, state)
    save_state(state)


if __name__ == "__main__":
    main()
