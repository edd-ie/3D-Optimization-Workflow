import os
cases=list(range(0, 70))
ok_def = []
ok_res = []
failed_def = []
for case in cases:
    def_file = f"sample{case}_infill.def"
    if os.path.exists(def_file):
        print(f"Def file for case {case} exists")
        ok_def.append(case)
    res_file = f"sample{case}_infill_001.res"
    if os.path.exists(res_file):
        print(f"Res file for case {case} exists")
        ok_res.append(case)
    if os.path.exists(def_file) and not os.path.exists(res_file):
        print(f"Def file for case {case} exists but res file does not exist")
        failed_def.append(case)
    
print(ok_def)
print(ok_res)
print(failed_def)
    
    