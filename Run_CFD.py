import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# inputs & parameters
num_cases = 45
total_runs = 15
core_number = 6
############################################

work_dir = Path(os.getcwd())
cases = list(range(0, num_cases))


def run_one_case(args):
    case, work_dir_str = args
    work_dir = Path(work_dir_str)

    def_file = work_dir / f"sample{case}_infill.def"
    if not def_file.exists():
        return case, None, "missing .def"

    out_prefix = work_dir / f"sample{case}_infill"
    cmd = (
        f'/opt/software/ansys_inc/v242/CFX/bin/cfx5solve -def "{def_file}" '
        '-start-method "Intel MPI Local Parallel" -double -part {core_number} -batch '
        f'-fullname "{out_prefix}" -save'
    )
    rc = os.system(cmd)
    ok = (rc == 0)
    message = f"os.system return code={rc}"
    return case, ok, message


def main():
    tasks = [(case, str(work_dir)) for case in cases]
    max_workers = max(1, min(int(total_runs), len(tasks)))
    print(f"[INFO] Submitting {len(tasks)} solve jobs (concurrency={max_workers})", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one_case, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                case, ok, message = fu.result()
                if ok is None:
                    print(f"[SKIP] Missing sample{case}_infill.def", flush=True)
                    continue
                if ok:
                    print(f"[OK] Solve completed for case {case}", flush=True)
                else:
                    print(f"[WARN] Solve failed for case {case}: {message}", flush=True)
            except Exception as e:
                print(f"[ERROR] Worker failed: {repr(e)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
