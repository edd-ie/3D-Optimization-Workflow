import os
# import subprocess
# import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


sample_ID = list(range(0,70))
mesh_script_base = 'mesh_script__automated'
# file_ID = ['','_29layers','_addedInlfationIterations','_addedLEsolve']
# file_ID = ['_noSkewTol']
file_ID = ['_reDimStripDomain_addedsolve']
PW_path = '/opt/software/Pointwise/Pointwise2023.2/pointwise'
total_runs = 15

def _run_one_sample(args):
    sample_num, text_2_replace, mesh_script_file, mesh_script_file_updated, pw_path = args
    with open(mesh_script_file,'r') as file:
        file_contents = file.read()
        new_text = f"sample_blade{sample_num}"
        updated_contents = file_contents.replace(text_2_replace,new_text)

    with open(mesh_script_file_updated,'w') as file:
        file.write(updated_contents)

    print(f'Generating mesh for sample {sample_num}...\n')
    cmd_PW = f"{pw_path} -b {mesh_script_file_updated}"
    code = os.system(cmd_PW)
    return sample_num, int(code)


def loop_seeds(text_2_replace,mesh_script_file, mesh_script_file_updated_base):
    # Parallelize per-sample (1 cpu/job). Make updated GLF unique per sample to avoid overwrites.
    tasks = [
        (sample_num, text_2_replace, mesh_script_file,
         f"{mesh_script_file_updated_base[:-4]}_{sample_num}_2.glf", PW_path)
        for sample_num in sample_ID
    ]
    max_workers = max(1, min(int(total_runs), len(tasks)))
    print(f"[INFO] Submitting {len(tasks)} mesh jobs (concurrency={max_workers})", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one_sample, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                sample_num, code = fu.result()
                if code == 0:
                    print(f"[OK] Finished sample {sample_num}", flush=True)
                else:
                    print(f"[WARN] Sample {sample_num} exited code {code}", flush=True)
            except Exception as e:
                print(f"[ERROR] Worker failed: {repr(e)}", file=sys.stderr, flush=True)


for i in file_ID:
    mesh_script_fileID = f'{mesh_script_base}{i}'
    mesh_script_file = f'{mesh_script_fileID}.glf'
    mesh_script_file_updated = f'{mesh_script_fileID}_updated.glf'
    text_2_replace = "sample_blade3"
    loop_seeds(text_2_replace,mesh_script_file,mesh_script_file_updated)


