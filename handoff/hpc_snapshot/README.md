# Cluster script snapshot

These files are reconstructed from the commands run in the cloud session on 1 November 2026. The live versions on the cluster (`/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/hpc/`) are authoritative and may differ in small ways, for example `make_tables.py` here has its Phase 1 filter written in one pass, and `job.sbatch` on the cluster has the conda path produced by `conda info --base`. `autopilot.py` is new and is not on the cluster yet: install it as described in section 0 of the handoff prompt. It was tested only against a mocked cluster. Not included because they are short and described in `../NEXT_SESSION_PROMPT.md`: `j0_compare.py`, `p1_compare.py`, `p1_raw_compare.py`, `archtest.sh`.

Do not copy these over the cluster files without diffing them first. They are kept outside the project's `hpc/` path on purpose, so pulling this branch on the cluster cannot clash with untracked files there.
