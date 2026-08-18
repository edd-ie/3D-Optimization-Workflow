import numpy as np
import os
import csv
from pathlib import Path


cases = list (range(30,45))  # 0..59  
work_dir = Path("/home/irtiza/scratch/mesh/infill6")
All_iter_results = np.empty((0, 10))
failed_cases = []
for case in cases:
    res_file = work_dir / f"sample{case}_infill_001.res"
    if not res_file.exists():
        failed_cases.append(case)

for case in cases:
    if case not in failed_cases:
        case_dir = work_dir / f"sample{case}_infill_001"
   

        if case_dir is not None:
            bak_path1 = case_dir / "500_full.bak"
            bak_path2 = case_dir / "1000_full.bak"

            # CFX-Post (run in the per-case folder so CSV is per-case) 
            cse_path1 = work_dir / "export_results500.cse"
            cse_path2 = work_dir / "export_results1000.cse"
            # Use bash so `module load ...` works on typical Linux/HPC environments.
            cmd_post1 = f'bash -lc \'cd "{case_dir}" && module load ansys && cfx5post -batch "{cse_path1}" "{bak_path1}"\''
            cmd_post2 = f'bash -lc \'cd "{case_dir}" && module load ansys && cfx5post -batch "{cse_path2}" "{bak_path2}"\''
            os.system(cmd_post1)
            os.system(cmd_post2)

   
        template_cse_dst = work_dir / "export_results1600.cse"

        res_dir = work_dir / f"sample{case}_infill_001.res"
     


        res_run_dir = res_dir
        res_case_suffix = None
   

        if res_run_dir is not None and template_cse_dst.exists():
            with open(template_cse_dst, "r") as file:
                contents = file.read()
                contents = contents.replace("Case Name = Case sample0_001", f"Case Name = Case sample{case}_001")
                contents = contents.replace("Export File = export1600.csv", f"Export File = export1600_{case}.csv")
            custom_cse = work_dir / f"export_results1600_{case}.cse"
            with open(custom_cse, "w") as file:
                file.write(contents)

            cmd_post1600 = f'bash -lc \'cd "{work_dir}" && module load ansys && cfx5post -batch "{custom_cse}" "{res_run_dir}"\''
            os.system(cmd_post1600)

        # Use the actual per-case folder that exists (sample{case}_00*)
        csv_path1 = case_dir / "export500.csv"
        csv_path2 = case_dir / "export1000.csv"
        csv_path3 = Path(f"/home/irtiza/scratch/mesh/infill6/export1600_{case}.csv")

        rows_to_add = []

        def read_first_data_row(p: Path):
            with open(p, newline="") as f:
                reader = csv.reader(f)
                r = next(reader, None)
                # skip a likely header row if present
                if r is not None and r and not any(ch.isdigit() for ch in "".join(r[:3])):
                    r = next(reader, None)
                return r

        if csv_path1 is not None and csv_path1.exists():
            r = read_first_data_row(csv_path1)
            if r is not None:
                J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP = map(float, (r[3], r[5], r[4], r[6], r[7], r[2], r[8], r[0], r[1]))
                iter_results = np.array([case, J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP], dtype=float)
                rows_to_add.append(iter_results.reshape(1, -1))

        if csv_path2 is not None and csv_path2.exists():
            r = read_first_data_row(csv_path2)
            if r is not None:
                J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP = map(float, (r[3], r[5], r[4], r[6], r[7], r[2], r[8], r[0], r[1]))
                iter_results2 = np.array([case, J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP], dtype=float)
                rows_to_add.append(iter_results2.reshape(1, -1))

        if csv_path3 is not None and csv_path3.exists():
            r = read_first_data_row(csv_path3)
            if r is not None:
                J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP = map(float, (r[3], r[5], r[4], r[6], r[7], r[2], r[8], r[0], r[1]))
                iter_results3 = np.array([case, J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP], dtype=float)
                rows_to_add.append(iter_results3.reshape(1, -1))

        if rows_to_add:
            iter_results_mat = np.vstack(rows_to_add)
            All_iter_results = np.vstack((All_iter_results, iter_results_mat))

out_name = "Results_infill6_2.txt"
np.savetxt(
    out_name,
    All_iter_results,
    header="case   J   kT   kQ   thrust   torque   drag   total_resistance   CD   CP",
    fmt="%.10g",
)


