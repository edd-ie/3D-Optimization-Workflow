
import os
import csv
import math
import numpy as np
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# inputs & parameters
cases = list(range(50,68))



num_cases = len(cases)
total_runs = 10
pred_file = "infill1_predictions.txt"
default = "create_def.pre"
############################################

try:
    predictions = np.loadtxt(Path(pred_file), ndmin=2)
except ValueError:
    # Predictions file has a non-comment header row (J_pred_mu ...); skip it.
    predictions = np.loadtxt(Path(pred_file), ndmin=2, skiprows=1)
j_values = np.asarray(predictions, dtype=float)[:, 0]
work_dir = Path(os.getcwd())
default_file = Path(work_dir / default)


def run_one_case(args):
    case, j_value, work_dir_str, default_file_str = args
    work_dir = Path(work_dir_str)
    default_file = Path(default_file_str)

    case_cas = work_dir / f"ORCA_gridsample_blade{case}.cas"
    if not case_cas.exists():
        # Only generate a .pre/.def if the mesh exists for this case
        return case, None

    pre_file_updated = work_dir / f"create_def_updated_{case}.pre"
    with open(default_file, "r") as file:
        contents = file.read()
        contents = contents.replace("ORCA_gridsample_blade43.cas", f"ORCA_gridsample_blade{case}.cas")
        contents = contents.replace("J=0.5", f"J=if( Accumulated Time Step>0 && Accumulated Time Step < 601, {j_value}-0.05, if( Accumulated Time Step >600 && Accumulated Time Step < 1201, {j_value}, {j_value}+0.05))")
        contents = contents.replace(
            "writeCaseFile filename=/home/irtiza/mesh/new_opt/sample43_infill.def",
            f"writeCaseFile filename=/home/irtiza/mesh/new_opt/sample{case}_infill.def"
        )
    with open(pre_file_updated, "w") as file:
        file.write(contents)

    print(f"pre file updated complete for case {case}")
    cmd_pre = f"/opt/software/ansys_inc/v242/CFX/bin/cfx5pre -batch {pre_file_updated}"
    os.system(cmd_pre)

    ok = (work_dir / f"sample{case}_infill.def").exists()
    return case, ok


def main():
    tasks = []
    for case, j_value in zip(cases, j_values):
        tasks.append((case, float(j_value), str(work_dir), str(default_file)))

    max_workers = max(1, min(int(total_runs), len(tasks)))
    print(f"[INFO] Submitting {len(tasks)} def-gen jobs (concurrency={max_workers})", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one_case, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                case, ok = fu.result()
                if ok is None:
                    print(f"[SKIP] Missing ORCA_gridsample_blade{case}.cas", flush=True)
                    continue
                if ok:
                    print(f"[OK] DEF generated for case {case}", flush=True)
                else:
                    print(f"[WARN] DEF missing for case {case}", flush=True)
            except Exception as e:
                print(f"[ERROR] Worker failed: {repr(e)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()