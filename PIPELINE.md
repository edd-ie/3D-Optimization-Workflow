# Propeller Optimization Workflow — Build Guide

For someone seeing this workflow for the first time. Read section 1 before
anything else; it explains the shape of the thing, which is different from the
2D slat workflow.

Reference case throughout: 5 blades, diameter `D = 1.4 m`.

---

## 1. Read this first: how this workflow is shaped

### It is NOT one script that loops

In the 2D slat workflow, one file (`gp_training_patched.py` / `pso.py`) owns
the whole optimization: it initialises, calls the CFD, loops, and finalises.
**This workflow does not work that way.**

Here the CFD is a 3D RANS simulation on a cluster — hours per design, hundreds
of designs. You cannot hold that inside a Python loop. So the loop is **broken
into stages that you run by hand**, and the "training" is **offline**: the GP
is fitted separately from the CFD, on results that already exist on disk.

```
2D slat workflow                  this workflow
------------------                ------------------------------
one driver script:                7 stages, run by hand:
  init                              1  propose designs   (python)
  loop:                             2  build CAD         (python)
    call CFD  <-- in-process        3  mesh              (Pointwise)
    fit model                       4  CFD               (cluster, hours)
    propose                         5  collect results   (python)
  finalise                          6  build training set(python)
                                    7  fit GP + propose  (python)
                                       ^                   |
                                       +-- you start ------+
                                           the next round
```

**Consequences for you as the builder:**

- No script calls the CFD solver and waits for it. Each stage ends by writing
  files; the next stage starts by reading them.
- **`gp_infill_from_training_data.py` does not call `X_blade` or `X_CAD`.** It
  only proposes new design vectors and writes them to a file.
- The handoff between every stage is **a file on disk**, never a function
  return. That is what makes the stages independently restartable.

### One round, start to finish

```mermaid
flowchart TD
    S1["STAGE 1 — propose designs<br/><b>Initial_sampling.py</b>"]
    S2["STAGE 2 — build CAD<br/><b>Initial_sampling.py</b> again,<br/>INPUT_MODE = 'infill'"]
    S3["STAGE 3 — mesh<br/><b>grid_samples_eg.py</b><br/>(Pointwise machine)"]
    S4["STAGE 4 — CFD<br/><b>generate_def.py → Run_CFD.py</b>"]
    S5["STAGE 5 — collect<br/><b>get_results.py</b>"]
    S6["STAGE 6 — build training set<br/><b>prop_training.py</b>"]
    S7["STAGE 7 — fit GP, propose next<br/><b>gp_infill_from_training_data.py</b>"]

    S1 -->|"control_points.txt"| S2
    S2 -->|"sample_blade&lt;case&gt;.iges"| S3
    S3 -->|"mesh"| S4
    S4 -->|"...__001.res"| S5
    S5 -->|"Results_&lt;tag&gt;.txt"| S6
    S6 -->|"training_data_&lt;tag&gt;.dat"| S7
    S7 -->|"next control_points.txt"| S2

```

Note the loop closes on **stage 2, not stage 1**. Stage 1 (LHS sampling) only
happens once, at the very beginning. Every round after that starts from the
GP's suggestions.

### The one thing that confuses everybody

`Initial_sampling.py` is used **twice, for two different jobs**, controlled by
one variable at the top of the file:

```python
INPUT_MODE = "lhs"      # stage 1: GENERATE new design vectors (first round only)
INPUT_MODE = "infill"   # stage 2: do NOT generate; read the GP's suggestions
                        #          and build CAD for them
```

In `"infill"` mode it generates no new samples. It reads the control points
the GP wrote, then calls `X_blade` and `X_CAD` per case to produce the IGES
files. That is the only place `X_blade` and `X_CAD` are invoked in a normal
round.

Other modes: `"existing"` (read a specific control-points file), `"test"`
(short smoke run, case ids from 1001).

### What each stage costs

| Stage | Runs where | Typical cost |
|---|---|---|
| 1 propose (LHS) | laptop | seconds |
| 2 CAD | laptop, `pyocc310` | **~90 s per design** |
| 3 mesh | Pointwise machine | minutes per design |
| 4 CFD | cluster | **hours per design** |
| 5 collect | laptop | seconds |
| 6 training set | laptop | seconds |
| 7 GP + propose | laptop | minutes |

Stages 3 and 4 are why the loop is manual.

---

## 2. What to run, in order

**First round only:**

```bash
# stage 1 — set INPUT_MODE = "lhs" in Initial_sampling.py
python Initial_sampling.py
```

**Every round (including the first, after stage 1):**

```bash
# stage 2 — set INPUT_MODE = "infill"
conda activate pyocc310
python Initial_sampling.py          # -> sample_blade<case>.iges

# stage 3 — on the Pointwise machine
python grid_samples_eg.py           # -> meshes

# stage 4 — CFD
python generate_def.py              # -> sample<case>_infill.def
python Check_def.py                 # optional validation
python Run_CFD.py                   # -> ...__001.res

# stage 5 — collect
python get_results.py               # -> Results_<tag>.txt

# stage 6 — build the GP training set   <-- easy to forget, do not skip
python prop_training.py             # -> training_data_<tag>.dat

# stage 7 — fit GP, propose next designs
python gp_infill_from_training_data.py
```

Then bump `INFILL_ROUND` in `pipeline_config.py` and go back to stage 2.

`gp_infill_from_training_data_pso.py` is a drop-in alternative to stage 7
(same in and out, PSO instead of ESPSOLS for the acquisition search).

---

## 3. Where state lives

Every stage communicates through files. `pipeline_config.py` defines all of
them, keyed off **`INFILL_ROUND`** — one variable that tags the whole round.

| Stage writes | File | Read by |
|---|---|---|
| 7 (or 1) | `<tag>_control_points.txt` | stage 2 |
| 2 | `geometry/<tag>/sample_blade<case>.iges` | stage 3 |
| 4 | `sample<case>_infill_001.res` | stage 5 |
| 5 | `New_training/Results_<tag>.txt` | stage 6 |
| 6 | `New_training/training_data_<tag>.dat` | stage 7 |

Because each stage only needs the files before it, any stage can be re-run
alone. A crashed CAD batch resumes: cases whose IGES already exists are
skipped.

**Start by reading `pipeline_config.py`.** It is the map of the whole
workflow.

---

## 4. Design vector — the only pipeline input

`pitch_con` (6 floats) + `chord_con` (5 floats) = **11 floats**, all bounded.

| Index | Array | Min | Max |
|---|---|---|---|
| 0 | `pitch_con[0]` | 0.60 | 1.25 |
| 1 | `pitch_con[1]` | 0.00 | 1.00 |
| 2 | `pitch_con[2]` | 0.35 | 0.75 |
| 3 | `pitch_con[3]` | 0.05 | 0.50 |
| 4 | `pitch_con[4]` | 0.05 | 0.50 |
| 5 | `pitch_con[5]` | 0.40 | 0.85 |
| 6 | `chord_con[0]` | 0.15 | 0.30 |
| 7 | `chord_con[1]` | 0.45 | 0.85 |
| 8 | `chord_con[2]` | 0.05 | 0.50 |
| 9 | `chord_con[3]` | 0.25 | 0.70 |
| 10 | `chord_con[4]` | 0.05 | 0.30 |

Defined in `Initial_sampling.py` (`pitch_bounds`, `chord_bounds`). Must match
`gp_infill_from_training_data.py`.

---

## 5. Component reference — geometry

### 5.1 `x_blade_new.py` — function `X_blade()`

```python
X_blade(pitch_con, chord_con, x1, *,
        return_bezier_info=False, return_blade_surface=False,
        bezier_constraint_mode='project', apply_coupled_constraints=None,
        write_dat=True)
```

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pitch_con` | float[6] | — | design vector part 1 |
| `chord_con` | float[5] | — | design vector part 2 |
| `x1` | int | — | case id |
| `return_blade_surface` | bool | `False` | **must be `True`** to feed the next stage |
| `write_dat` | bool | `True` | writes `ORCA<case>.dat` |

**Returns** a tuple:

| Position | Name | Type / size |
|---|---|---|
| 0 | `points` | float[3021, 3] — section point cloud |
| 1 | `min_dis` | float — min blade-to-blade clearance |
| 2 | `constraint_violation` | int — **non-zero ⇒ reject the design** |
| 3 | `chord_con_points` | float[N, 2] |
| 4 | `pitch_con_points` | float[N, 2] |
| 5 | `Pitch` | callable |
| 6 | `ChordLength` | callable |
| −1 | `blade_surface` | `BladeSurface` object *(only if requested)* |

**File out:** `ORCA<case>.dat`, ~103 KB (if `write_dat=True`).
**Time:** 0.08 s.
**Connects to:** `build_drdc_grids(blade_surface)`.

> Check `constraint_violation` before continuing. Infeasible designs must not
> reach the CAD stage.

### 5.2 `tip_surfaces_new.py` — function `build_drdc_grids()`

```python
build_drdc_grids(blade, cfg=None, verbose=True, row_cache=None)
```

**In:** `blade` = the `BladeSurface` from `X_blade`.
`cfg` = `TipConfig()` dataclass (below); `None` uses defaults.

**Out:** `dict` of 5 numpy arrays `(rows, cols, 3)` in **metres**, plus `meta`.

| Key | Shape (defaults) | Points | Memory |
|---|---|---|---|
| `te_strip` | (25, 241, 3) | 6 025 | 141 KB |
| `central_pressure` | (25, 28, 3) | 700 | 16 KB |
| `le_strip` | (25, 241, 3) | 6 025 | 141 KB |
| `central_suction` | (25, 28, 3) | 700 | 16 KB |
| `tip` | (28, 241, 3) | 6 748 | 158 KB |
| `meta` | dict | — | counts, split points, diagnostics |

**Time: ~85 s** — the pipeline's geometry bottleneck.
**Connects to:** `hub_grids(grids)` and `X_CAD(grids, ...)`.

`TipConfig` parameters you may touch:

| Parameter | Default | Effect |
|---|---|---|
| `delta_c` | 0.02 | curve spacing (·D). **Drives grid size and runtime** |
| `n_wrap` | 241 | columns per wrap grid |
| `strip_width` | 0.07 | LE/TE strip width (·D) |
| `eta_top` | 0.64 | radial station of the patch split |
| `tip_cluster` | 0.6 | tip-cut clustering |
| `n_iter` | 0 | smoothing iterations — **leave at 0** |
| `max_curves` | 120 | safety cap on curve count |

Halving `delta_c` roughly doubles grid size and runtime. All other fields have
working defaults; changing them is not required to run the pipeline.

### 5.3 `hub_new.py` — function `hub_grids()`

```python
hub_grids(grids, hub_height=None, hub_center=0.0, n_blades=5,
          n_s=121, n_x=61, n_rho=25, cap_inner_radius=0.0, verbose=True)
```

**In:** `grids` = the dict from `build_drdc_grids` (reads the blade root ring
from it to size the hub).

| Parameter | Default | Notes |
|---|---|---|
| `hub_height` | `None` → 1.0 m | total axial height. **Keep fixed across a batch** |
| `hub_center` | 0.0 | hub midpoint x. **Keep fixed** |
| `n_blades` | 5 | sector spans 360/Z° |
| `cap_inner_radius` | 0.0 | 0 = closed ends; > 0 = shaft bore radius |
| hub **radius** | — | **not a parameter** — measured from the blade |

**Out:** `(dict, info)`

| Key | Shape | Points | Memory |
|---|---|---|---|
| `hub_sector` | (121, 61, 3) | 7 381 | 173 KB |
| `hub_cap_lo` | (25, 121, 3) | 3 025 | 71 KB |
| `hub_cap_hi` | (25, 121, 3) | 3 025 | 71 KB |
| `info` | dict | — | radius, extents, Z, clearances |

Reference values: radius **0.119000 m**, hub x ∈ [−0.500, +0.500], sector 72°.

**Time:** < 0.01 s.
**Connects to:** `X_CAD` (called internally by it — you rarely call this
directly).

**Raises** if the hub can't be built correctly (non-cylindrical root, hub too
short, or blades would overlap at the chosen `n_blades`).

### 5.4 `X_CAD_new.py` — functions `X_CAD()` / `X_CAD_from_design()` *(needs pythonOCC)*

```python
X_CAD(grids, x1, output_dir=None, hub=True, hub_height=None,
      hub_center=0.0, n_blades=5)

X_CAD_from_design(pitch_con, chord_con, x1, output_dir=None,
                  tip_config=None, write_dat=False, verbose=True,
                  hub=True, hub_height=None, hub_center=0.0, n_blades=5)
```

`X_CAD_from_design` is the **one-call path**: design vector → IGES. Use it
unless you need the intermediate grids.

| Parameter | Default | Notes |
|---|---|---|
| `x1` | — | case id, used in the filename |
| `output_dir` | `None` | `None` ⇒ `pipeline_config` default location |
| `hub` | `True` | `False` = blade only |
| `hub_height` / `hub_center` / `n_blades` | 1.0 / 0.0 / 5 | passed to `hub_grids` |
| `tip_config` | `None` | a `TipConfig`, or `None` for defaults |

**Out:**

| Item | Value |
|---|---|
| file | `sample_blade<case>.iges` |
| size | **1.8 – 2.1 MB** |
| surfaces | **8** — 5 blade + hub sector + 2 hub caps |
| units | mm |
| total points | 33 629 |

Return value is the OCC shape; the path comes from
`cad_output_paths(case_id, output_dir)["iges"]`.

```python
# design vector -> IGES, one call
from X_CAD_new import X_CAD_from_design
X_CAD_from_design([1.025, 0.525, 0.55, 0.325, 0.325, 0.55],
                  [0.25, 0.65, 0.325, 0.55, 0.2], 1001)
```

**Connects to:** Pointwise (meshing, external).

> Raises rather than writing bad geometry if any surface fails its
> export check.

### 5.5 `cad_worker.py` — subprocess wrapper (CLI)

```bash
python cad_worker.py <case_id> <geometry_dir> <design_path>
```

Runs `X_CAD_from_design` in a separate process so a hang in pythonOCC can be
killed by timeout. Used by `Initial_sampling.py`; call it directly if your
platform schedules CAD jobs itself.

---

## 6. Component reference — sampling driver

### `Initial_sampling.py` — stages 1 and 2

| | |
|---|---|
| **In** | bounds (11 × 2), `N_GENERATED_BLADES`, `INPUT_MODE` |
| **Out** | `generated_<tag>_control_points.txt`, `rejected_<tag>_control_points.txt`, `input__var_values_<tag>.dat`, one IGES per accepted case |
| **Method** | Latin hypercube (`scipy.stats.qmc`, seed 100) |

Per case it calls `cad_worker.py` with a timeout, so one bad design cannot
stall the batch.

---

## 7. Component reference — CFD chain

| Component | In | Out | Size |
|---|---|---|---|
| `generate_def.py` | mesh + `create_def.pre` | `sample<case>_infill.def` | — |
| `Run_CFD.py` | `.def` | `sample<case>_infill_001.res` | ~100 MB |
| `Check_def.py` | `.def` | validity report | — |
| `get_results.py` | `.res` | `export<N>_<case>.csv` → `Results_<tag>.txt` | small |

`Results_<tag>.txt` columns:

```
case   J   kT   kQ   thrust   torque   drag   total_resistance   CD   CP
```

---

## 8. `prop_training.py` — CFD results → GP training data

The bridge between CFD and the surrogate. **Stage 6. Run after `get_results.py`, before
`gp_infill_from_training_data.py`.**

```bash
python prop_training.py
```

No arguments — all paths come from `pipeline_config.infill_paths()`, keyed off
`INFILL_ROUND`.

| | |
|---|---|
| **In** | `Results_<tag>.txt` (from `get_results.py`) — needs >= 10 columns |
| **In** | `<tag>_control_points_con_points.txt` — the 11 design vars per case |
| **Out** | **`training_data_<tag>.dat`** — 13 columns: 11 design vars + 2 objectives |
| **Out** | `poly_fit_eq_<tag>.txt` — fit equations and R^2 per case |
| **Out** | `intersection_plots_<tag>/` — one plot per case |

For each case it fits thrust(J) and total_resistance(J), finds the
self-propulsion point where they intersect, and writes that operating point as
the objective values.

| Setting | Default | Notes |
|---|---|---|
| `N_DESIGN_VARS` | 11 | must match the design vector |
| `SAVE_TRAINING_DATA` | `True` | set `False` for a dry run |

**Connects to:** `gp_infill_from_training_data.py`, which reads
`training_data_<tag>.dat`.

---

## 9. Component reference — surrogate driver

### `gp_infill_from_training_data.py` — stage 7

| | |
|---|---|
| **In** | `training_data_<tag>.dat` — **13 columns** = 11 design vars + 2 objectives |
| **Out** | next-round control points, `<tag>_predictions.txt`, `poly_fit_eq_<tag>.txt` |
| **Model** | Gaussian Process (Matérn + White) |
| **Acquisition** | `ESPSOLS.py` (default) or `PSO_function.py` |

`Gp_classifier.py` — optional feasibility filter; in: same training data, out:
feasible/infeasible prediction, used to skip designs likely to fail.

**Output feeds back to** `X_blade` as the next round of design vectors.

---

## 10. File and directory contract

| Path | Written by | Read by |
|---|---|---|
| `geometry/<TAG>/ORCA<case>.dat` | `X_blade` | reference only |
| `geometry/<TAG>/sample_blade<case>.iges` | `X_CAD` | Pointwise |
| `input__var_values_<tag>.dat` | `Initial_sampling` | CFD bookkeeping |
| `New_training/training_data_<tag>.dat` | you (assembled) | GP infill |
| `New_training/Results_infill<n>.txt` | `get_results` | GP infill |

Paths resolve via `pipeline_config.py`, relative to its own location:

```python
cad_output_paths(case_id, geometry_dir=None)
# -> {"dir", "orca_dat", "iges", "stl"}
```

`New_training/` and `geometry/` are **data, not code** — not in the repo.
Create them or repoint `DATA_DIR` in `pipeline_config.py`.

---

## 11. Environment

| Component | Requirement |
|---|---|
| `X_CAD_new.py`, `cad_worker.py`, `Initial_sampling.py` | **pythonOCC** — `conda activate pyocc310` |
| everything else | numpy, scipy, sklearn |

---

## 12. Cost per design

| Stage | Time | Output |
|---|---|---|
| `X_blade` | 0.08 s | 3 021 points |
| `build_drdc_grids` | **85 s** | 20 198 points |
| `hub_grids` | < 0.01 s | 13 431 points |
| `X_CAD` → IGES | ~2 s | ~2 MB, 8 surfaces |
| meshing + CFX | external | ~100 MB |

Geometry ≈ 90 s per design, dominated by `build_drdc_grids`. Raise `delta_c`
to trade surface resolution for speed.

---

---

## 13. File index — what each script is for

**Drivers (you run these):**

| File | Role |
|---|---|
| `Initial_sampling.py` | stages 1 & 2: propose designs (`lhs`) and build CAD (`infill`) |
| `grid_samples_eg.py` | stage 3: drives Pointwise meshing (**runs on the Pointwise machine**) |
| `generate_def.py` | build CFX `.def` per case |
| `Run_CFD.py` | submit/solve CFX |
| `Check_def.py` | validate `.def` files |
| `get_results.py` | `.res` → `Results_<tag>.txt` |
| `prop_training.py` | stage 6: results → `training_data_<tag>.dat` |
| `gp_infill_from_training_data.py` | stage 7: GP fit + next designs |
| `gp_infill_from_training_data_pso.py` | same, PSO acquisition |
| `Gp_classifier.py` | feasibility classifier (optional) |
| `job_submission.py` | cluster job submission helper |
| `Main_PSO.py` | standalone PSO driver |

**Libraries (imported, never run directly):**

| File | Provides |
|---|---|
| `x_blade_new.py` | `X_blade()` — design vector → blade surface |
| `blade_surface_new.py` | the blade surface object |
| `tip_surfaces_new.py` | `build_drdc_grids()` — 5 blade patches |
| `blade_cuts_new.py`, `tip_smoothing_new.py` | used by `tip_surfaces_new` |
| `hub_new.py` | `hub_grids()` — hub sector + caps |
| `X_CAD_new.py` | `X_CAD_from_design()` → IGES |
| `cad_worker.py` | CLI subprocess wrapper for CAD |
| `para.py`, `para_control_bez_updated.py` | section + Bezier design control |
| `coupled_constraint_config.py` | design constraint limits |
| `rot_axis.py`, `skew_symmetric_matrix.py` | rotation helpers |
| `pipeline_config.py` | **all file paths and tags** |
| `pipeline_io.py` | table read/write helpers |
| `ESPSOLS.py`, `PSO_function.py` | acquisition optimizers |
| `bound_check.py`, `penalty.py`, `repair.py` | constraint handling |

**Data files:**

| File | Role |
|---|---|
| `airfoil_data_fixed.csv` | airfoil section table (required input) |
| `create_def.pre` | CFX-Pre session template (required input) |

Start with `pipeline_config.py` — it defines every path and the round tag the
other scripts key off.
