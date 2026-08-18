import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 70 cases: 0..69
cases = list(range(0, 70))
work_dir = Path("/home/irtiza/scratch/mesh/infill1/")

concurrent_jobs = 6

def main():
    # IMPORTANT:
    # Submitting jobs from Python in parallel does NOT limit how many run on the cluster.
    # To enforce license limits, use a Slurm job array with a concurrency cap: %concurrent_jobs.
    if not cases:
        print("[WARN] No cases to submit.", flush=True)
        return

    array_lo = int(min(cases))
    array_hi = int(max(cases))
    array_spec = f"{array_lo}-{array_hi}%{int(concurrent_jobs)}"

    array_script = work_dir / "sample_infill_array.sh"
    if not array_script.exists():
        raise FileNotFoundError(
            f"Missing array script: {array_script}"
        )

    cmd = f"cd {work_dir} && sbatch --array={array_spec} {array_script.name}"
    print(f"[INFO] Submitting array: {cmd}", flush=True)
    rc = os.system(cmd)
    if rc != 0:
        print(f"[ERROR] sbatch failed rc={rc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()






