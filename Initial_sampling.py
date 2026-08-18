# initial sampling for surrogate model.
# The script supports two input paths:
# 1) read existing control points from file
# 2) generate N new blade designs using optimized Latin hypercube sampling

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import para_control_bez_updated as bezier_constraints
import para
from pipeline_config import infill_paths, initial_sampling_paths
from pipeline_io import append_design_row, clear_file, load_table, save_table, split_case_ids
from X_CAD_new import X_CAD_from_design
from x_blade_new import X_blade


ROOT = Path(__file__).resolve().parent
PATHS = initial_sampling_paths()
INFILL_PATHS = infill_paths()

# Input mode:
# - "existing": read design/control points from EXISTING_CONTROL_POINTS_PATH
# - "lhs": generate N_GENERATED_BLADES control points with optimized Latin hypercube
# - "infill": generate CAD for the current infill round's control points
# - "test": short LHS smoke test of the DRDC CAD path (x_blade_new / X_CAD_new)
INPUT_MODE = "test"

# In "infill"/"test" mode, case ids start at CASE_ID_START so geometry does not
# overwrite earlier cases (e.g. test: 1001, 1002, ...).
CASE_ID_START = 1001

EXISTING_CONTROL_POINTS_PATH = PATHS["existing_input"]
GENERATED_CONTROL_POINTS_PATH = PATHS["generated"]
INFILL_CONTROL_POINTS_PATH = INFILL_PATHS["control_points"]

# Per-case CAD timeout (seconds). pythonOCC hangs on some degenerate geometries
# and cannot be interrupted in-process, so each X_CAD runs in a subprocess that
# is killed if it exceeds this budget. Set to None to disable the guard.
CAD_TIMEOUT_S = 2000

if INPUT_MODE == "infill":
    CONTROL_OUTPUT_PATH = INFILL_PATHS["cad_accepted"]
    REJECTED_CONTROL_POINTS_PATH = INFILL_PATHS["cad_rejected"]
    PLOT_OUTPUT_PATH = INFILL_PATHS["cad_plot"]
    GEOMETRY_DIR = INFILL_PATHS["geometry_dir"]
    CASE_ID_OFFSET = CASE_ID_START
    # Keep prior outputs so a killed/partial batch can be resumed (cases whose
    # IGES already exists are skipped below).
    RESET_OUTPUT_FILES_DEFAULT = False
elif INPUT_MODE == "test":
    CONTROL_OUTPUT_PATH = ROOT / "New_training" / "test_control_points.txt"
    REJECTED_CONTROL_POINTS_PATH = ROOT / "New_training" / "test_rejected_control_points.txt"
    PLOT_OUTPUT_PATH = ROOT / "New_training" / "test_pitch_chord_curves.png"
    GEOMETRY_DIR = ROOT / "geometry" / "test_drdc"
    GENERATED_CONTROL_POINTS_PATH = ROOT / "New_training" / "test_generated_control_points.txt"
    CASE_ID_OFFSET = CASE_ID_START
    RESET_OUTPUT_FILES_DEFAULT = True
else:
    CONTROL_OUTPUT_PATH = PATHS["accepted"]
    REJECTED_CONTROL_POINTS_PATH = PATHS["rejected"]
    PLOT_OUTPUT_PATH = PATHS["plot"]
    GEOMETRY_DIR = PATHS["geometry_dir"]
    CASE_ID_OFFSET = 0
    RESET_OUTPUT_FILES_DEFAULT = False

# When False, skip chord_sum / pitch_diff coupled limits only (no reject or project).
# Variable bounds and blade-clearance rejection always apply.
USE_BEZIER_VIOLATION_FILTER = False
BEZIER_CONSTRAINT_MODE = "reject"
BEZIER_CHORD_SUM_MAX_NORM = 1.49
BEZIER_PITCH_DIFF_MAX_NORM = 0.73
WRITE_CASE_ID_TO_OUTPUT_FILES = True
CONTROL_POINT_FORMAT = "%.6f"

N_GENERATED_BLADES = 5
LHS_SEED = 102
LHS_OPTIMIZATION = "random-cd"
WRITE_GENERATED_CONTROL_POINTS = True
RESET_OUTPUT_FILES = RESET_OUTPUT_FILES_DEFAULT

# Keep in sync with gp_infill_from_training_data.py (expanded after infill5 edge-saturation).
pitch_bounds = np.array([
    [0.60, 1.25],  # p1y
    [0.00, 1.00],  # y4 factor
    [0.35, 0.75],  # p4x
    [0.05, 0.50],  # d1
    [0.05, 0.50],  # d2
    [0.40, 0.85],  # p7y
])

chord_bounds = np.array([
    [0.15, 0.30],  # p1y
    [0.45, 0.85],  # p4x
    [0.05, 0.50],  # d1
    [0.25, 0.70],  # y4
    [0.05, 0.30],  # w56
])

R_FINE = np.linspace(0.17, 0.998, 100)


def load_or_generate_control_points(bounds, d_total):
    if INPUT_MODE == "existing":
        X_data = load_table(EXISTING_CONTROL_POINTS_PATH)
        case_ids, X_phys = split_case_ids(X_data, d_total)
        print(f"Loaded {X_phys.shape[0]} designs from {EXISTING_CONTROL_POINTS_PATH.name}")
        if X_data.shape[1] == d_total + 1:
            print("Using leading case_id column from the input control-point file")
        else:
            print("No case_id column found; falling back to row index for output naming")
        return case_ids, X_phys

    if INPUT_MODE == "infill":
        X_data = load_table(INFILL_CONTROL_POINTS_PATH)
        case_ids, X_phys = split_case_ids(X_data, d_total)
        case_ids = case_ids.astype(int) + CASE_ID_OFFSET
        print(
            f"Loaded {X_phys.shape[0]} infill designs from "
            f"{INFILL_CONTROL_POINTS_PATH.name}; case ids {int(case_ids.min())}.."
            f"{int(case_ids.max())}"
        )
        return case_ids, X_phys

    if INPUT_MODE in ("lhs", "test"):
        sampler = qmc.LatinHypercube(
            d=d_total,
            seed=LHS_SEED,
            optimization=LHS_OPTIMIZATION,
        )
        unit_samples = sampler.random(N_GENERATED_BLADES)
        X_phys = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

        if WRITE_GENERATED_CONTROL_POINTS:
            save_table(GENERATED_CONTROL_POINTS_PATH, X_phys, fmt=CONTROL_POINT_FORMAT)
            print(f"Wrote generated control points to {GENERATED_CONTROL_POINTS_PATH.name}")

        case_ids = CASE_ID_OFFSET + np.arange(X_phys.shape[0], dtype=int)
        mode_label = "test LHS" if INPUT_MODE == "test" else "optimized Latin hypercube"
        print(
            f"Generated {X_phys.shape[0]} designs using {mode_label} "
            f"({LHS_OPTIMIZATION}, seed={LHS_SEED}); "
            f"case ids {int(case_ids.min())}..{int(case_ids.max())}"
        )
        return case_ids, X_phys

    raise ValueError("INPUT_MODE must be 'existing', 'lhs', 'infill', or 'test'")


def initialize_output_files():
    if not RESET_OUTPUT_FILES:
        return
    CONTROL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_CONTROL_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_file(CONTROL_OUTPUT_PATH)
    clear_file(REJECTED_CONTROL_POINTS_PATH)


def generate_cad(pitch_con, chord_con, case_id, geometry_dir):
    """Generate one DRDC blade CAD, guarded by a subprocess timeout if configured.

    Returns True on success, False on timeout/failure so the caller can reject
    the case and continue with the rest of the batch.
    """
    pitch_con = np.asarray(pitch_con, dtype=float).ravel()
    chord_con = np.asarray(chord_con, dtype=float).ravel()
    if not CAD_TIMEOUT_S:
        X_CAD_from_design(pitch_con, chord_con, case_id, output_dir=geometry_dir, hub=True, hub_height=None, n_blades=5)
        return True

    tmp_design = Path(tempfile.gettempdir()) / f"cad_design_{case_id}.npy"
    np.save(tmp_design, np.concatenate([pitch_con, chord_con]))
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "cad_worker.py"),
             str(int(case_id)), str(geometry_dir), str(tmp_design)],
            check=True,
            timeout=CAD_TIMEOUT_S,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] X_CAD exceeded {CAD_TIMEOUT_S}s for case {case_id}; skipping")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] X_CAD worker failed for case {case_id}: rc={exc.returncode}")
        return False
    finally:
        try:
            tmp_design.unlink()
        except FileNotFoundError:
            pass


def append_control_points_row(path, values, case_id=None):
    append_design_row(
        path,
        values,
        case_id=case_id,
        fmt=CONTROL_POINT_FORMAT,
        write_case_id=WRITE_CASE_ID_TO_OUTPUT_FILES,
    )


def main():
    d_pitch = pitch_bounds.shape[0]
    d_chord = chord_bounds.shape[0]
    d_total = d_pitch + d_chord
    bounds = np.vstack([pitch_bounds, chord_bounds])

    initialize_output_files()

    geometry_dir = GEOMETRY_DIR
    geometry_dir.mkdir(parents=True, exist_ok=True)
    para.GEOMETRY_OUTPUT_DIR = geometry_dir
    print(f"Geometry output directory: {geometry_dir}")

    bezier_constraints.APPLY_COUPLED_CONSTRAINTS = USE_BEZIER_VIOLATION_FILTER
    if USE_BEZIER_VIOLATION_FILTER:
        bezier_constraints.CHORD_SUM_MAX_NORM = BEZIER_CHORD_SUM_MAX_NORM
        bezier_constraints.PITCH_DIFF_MAX_NORM = BEZIER_PITCH_DIFF_MAX_NORM
        print(
            "Using Bezier coupled-constraint thresholds: "
            f"b1={bezier_constraints.CHORD_SUM_MAX_NORM:.2f}, "
            f"b2={bezier_constraints.PITCH_DIFF_MAX_NORM:.2f}"
        )
        print(f"Bezier coupled-constraint filter enabled ({BEZIER_CONSTRAINT_MODE})")
    else:
        print(
            "Coupled Bezier limits disabled; "
            "variable bounds and blade-clearance rejection still apply"
        )

    case_ids, X_phys = load_or_generate_control_points(bounds, d_total)
    pitch_phys = X_phys[:, :d_pitch]
    chord_phys = X_phys[:, d_pitch:]
    n_samples = X_phys.shape[0]

    colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
    fig, (ax_pitch, ax_chord) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pitch.set_title("Pitch (all samples)")
    ax_chord.set_title("Chord (all samples)")
    ax_pitch.set_xlabel("radius (R)")
    ax_pitch.set_ylabel("pitch")
    ax_chord.set_xlabel("radius (R)")
    ax_chord.set_ylabel("chord")
    ax_pitch.grid(True)
    ax_chord.grid(True)

    accepted_count = 0
    rejected_count = 0

    for i in range(n_samples):
        case_id = int(case_ids[i])
        pitch_con = pitch_phys[i, :]
        chord_con = chord_phys[i, :]
        blade_count = case_id
        raw_input_values = np.asarray(np.concatenate([pitch_con, chord_con]), dtype=float).ravel()

        # Resumable: skip cases whose CAD was already generated.
        if (geometry_dir / f"sample_blade{case_id}.iges").exists():
            print(f"Case {case_id} already has CAD; skipping")
            accepted_count += 1
            continue

        (
            points,
            min_dis,
            constraint_violation,
            _chord_con_points,
            _pitch_con_points,
            Pitch,
            ChordLength,
            bezier_info,
        ) = X_blade(
            pitch_con,
            chord_con,
            blade_count,
            return_bezier_info=True,
            bezier_constraint_mode=BEZIER_CONSTRAINT_MODE,
            apply_coupled_constraints=USE_BEZIER_VIOLATION_FILTER,
        )

        if USE_BEZIER_VIOLATION_FILTER and bezier_info["rejected"]:
            reasons = ", ".join(bezier_info["reasons"]) or "unknown coupled-constraint violation"
            print(f"Bezier rejection for case {case_id}: {reasons}")
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        if constraint_violation == 1:
            print(f"Blade clearance violation for case {case_id}")
            print(min_dis)
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        if USE_BEZIER_VIOLATION_FILTER and bezier_info["violated"]:
            reasons = ", ".join(bezier_info["reasons"])
            print(f"Projected case {case_id} to satisfy Bezier constraints: {reasons}")

        print(f"Case {case_id} accepted for CAD and data export")

        try:
            cad_ok = generate_cad(pitch_con, chord_con, blade_count, geometry_dir)
        except Exception as e:
            print(f"[WARN] X_CAD failed for case {case_id}: {e}")
            cad_ok = False
        if not cad_ok:
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        # Save the raw design variables, optionally prefixed by case_id.
        append_control_points_row(CONTROL_OUTPUT_PATH, raw_input_values, case_id=case_id)

        c = colors[i % len(colors)]
        ax_pitch.plot(R_FINE, Pitch(R_FINE), color=c, linewidth=1.3, label=f"c{case_id}")
        ax_chord.plot(R_FINE, ChordLength(R_FINE), color=c, linewidth=1.3, label=f"c{case_id}")
        accepted_count += 1

    handles1, labels1 = ax_pitch.get_legend_handles_labels()
    if len(labels1) <= 15:
        ax_pitch.legend(ncol=2, fontsize=8)

    handles2, labels2 = ax_chord.get_legend_handles_labels()
    if len(labels2) <= 15:
        ax_chord.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Accepted {accepted_count} of {n_samples} samples")
    print(f"Rejected {rejected_count} of {n_samples} samples")


if __name__ == "__main__":
    main()


      