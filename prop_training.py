import numpy as np
import matplotlib.pyplot as plt
import os

from pipeline_config import infill_paths
from pipeline_io import load_table, save_table, split_case_ids

PATHS = infill_paths()
INPUT_FILE = str(PATHS["results"])
CONTROL_POINTS_FILE = str(PATHS["training_control_points"])
OUTPUT_FILE = str(PATHS["poly_fit"])
TRAINING_OUTPUT = str(PATHS["training_data"])
PLOT_DIR = str(PATHS["plot_dir"])
SAVE_TRAINING_DATA = True

# Number of design variables per control-point row (6 pitch + 5 chord).
N_DESIGN_VARS = 11


def _load_results_table(path: str) -> np.ndarray:
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError:
        # Some files have a non-comment header row; try skipping it.
        data = np.loadtxt(path, comments="#", skiprows=1)
    except OSError as e:
        raise FileNotFoundError(f"Could not read INPUT_FILE '{path}': {e}") from e

    if data.size == 0:
        raise ValueError(f"INPUT_FILE '{path}' is empty (no numeric rows).")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 10:
        raise ValueError(
            f"INPUT_FILE '{path}' has {data.shape[1]} columns; expected at least 10 "
            "(case_id, J, kT, kQ, thrust, torque, drag, total_resistance, CD, CP)."
        )

    case_col = data[:, 0]
    if not np.all(np.isfinite(case_col)):
        raise ValueError(f"INPUT_FILE '{path}' has non-finite case ids in column 0.")
    if not np.allclose(case_col, np.round(case_col), atol=1e-6):
        raise ValueError(
            f"INPUT_FILE '{path}' column 0 does not look like integer case ids. "
            "Did you point INPUT_FILE at a control-points file (e.g. chord_opt_bez_final.txt)?"
        )

    return data


# Plot styling (match `Python/run_bezier_points_from_file.py` big-format look)
FONT_FAMILY = "Times New Roman"
FIGSIZE = (22, 16)
TITLE_FONTSIZE = 60
LABEL_FONTSIZE = 60
TICK_FONTSIZE = 52
LEGEND_FONTSIZE = 42
CURVE_LINEWIDTH = 5.0
GRID_LINEWIDTH = 2.0
DATA_MARKERSIZE = 18
INTERSECTION_MARKERSIZE = 22
SUBPLOT_ADJUST = dict(left=0.11, right=0.99, bottom=0.13, top=0.92)

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": LABEL_FONTSIZE,
    }
)

data = _load_results_table(INPUT_FILE)

cases = np.unique(data[:, 0]).astype(int)

# Plot controls
SHOW_PLOTS = False  # set True to inspect plots interactively
SAVE_PLOTS = True
if SAVE_PLOTS:
    os.makedirs(PLOT_DIR, exist_ok=True)

# Start fresh each run (older runs may contain outdated results)
with open(OUTPUT_FILE, 'w') as f:
    f.write('Intersection method: quadratic thrust(J) vs quadratic total_resistance(J)\n')
    f.write('Intersection is where thrust_fit(J) == resistance_fit(J)\n\n')

cp_raw = load_table(CONTROL_POINTS_FILE)

# Map each case id to its control-point (design-var) row. Initial-round files
# have no case-id column (row index == case id); infill cad_accepted files carry
# an explicit, possibly offset, case-id column (e.g. 50..67).
cp_case_ids, control_points = split_case_ids(cp_raw, N_DESIGN_VARS)
cp_row_by_case = {int(cid): k for k, cid in enumerate(cp_case_ids)}

num_cases_total = control_points.shape[0]
all_case_ids = np.array(sorted(cp_row_by_case), dtype=int)
missing_cases = sorted(set(all_case_ids.tolist()) - set(cases.tolist()))

# Results keyed by case id (control-point ids need not be contiguous).
J_by_case = {}
torque_by_case = {}
eff_by_case = {}

extrapolation_cases = []
extra_rows_cases = []
linear_fit_cases = []
extrapolation_cases_J = []

extrapolation_cases_count = 0

for case in cases:
    if int(case) not in cp_row_by_case:
        # Results has a case id we don't have control points for
        continue

    case_data = data[data[:, 0].astype(int) == case, :]
    if case_data.shape[0] < 2:
        continue
    if case_data.shape[0] > 3:
        extra_rows_cases.append(case)

    # 2 J points -> linear fit; 3+ points -> quadratic fit
    fit_degree = 1 if case_data.shape[0] == 2 else 2
    fit_label = "linear" if fit_degree == 1 else "quadratic"
    if fit_degree == 1:
        linear_fit_cases.append(case)

    # Sort by J so lines draw left-to-right
    case_data = case_data[np.argsort(case_data[:, 1])]

    J = case_data[:, 1]
    kT = case_data[:, 2]
    kQ = case_data[:, 3] * 10
    thrust = np.abs(case_data[:, 4])
    torque = np.abs(case_data[:, 5])
    drag = np.abs(case_data[:, 6])
    total_resistance = np.abs(case_data[:, 7])
    CD = np.abs(case_data[:, 8])
    CP = np.abs(case_data[:, 9])

    # do a second order polynomial fit to the thrust data and see it matches the target thrust, store the corresposing J and torque values. 
    # for cases where extrapolation is needed note those cases for rerunning the CFD 
    # if the fit is good, store the corresponding J and torque values in a file for each case, also note the polynomial fit equation in the file.

    poly_fit = np.polyfit(J, thrust, fit_degree)
    poly_fit_eq = np.poly1d(poly_fit)

    poly_fit_res = np.polyfit(J, total_resistance, fit_degree)
    poly_fit_res_eq = np.poly1d(poly_fit_res)

    if fit_degree == 2:
        a_t, b_t, c_t = poly_fit
        poly_fit_eq_str = f'{a_t:.6g}*J^2 + {b_t:.6g}*J + {c_t:.6g}'
        a_r, b_r, c_r = poly_fit_res
        poly_fit_res_eq_str = f'{a_r:.6g}*J^2 + {b_r:.6g}*J + {c_r:.6g}'
    else:
        b_t, c_t = poly_fit
        poly_fit_eq_str = f'{b_t:.6g}*J + {c_t:.6g}'
        b_r, c_r = poly_fit_res
        poly_fit_res_eq_str = f'{b_r:.6g}*J + {c_r:.6g}'
   

    # Find where quadratic thrust(J) intersects quadratic total_resistance(J):
    # thrust_fit(J) - resistance_fit(J) = 0
    roots = np.roots(poly_fit_eq - poly_fit_res_eq)
    real_roots = roots[np.isreal(roots)].real

    j_min = float(np.min(J))
    j_max = float(np.max(J))
    j_mid = float(np.mean(J))

    EXTRAP_TOL = 0.05

    if real_roots.size == 0:
        intersection_J = float("nan")
        needs_extrapolation = True
    else:
        in_range = real_roots[(real_roots >= j_min) & (real_roots <= j_max)]
        if in_range.size > 0:
            needs_extrapolation = False
            intersection_J = float(in_range[np.argmin(np.abs(in_range - j_mid))])
        else:
            # Choose the real root closest to [j_min, j_max]
            dist_to_range = np.where(
                real_roots < j_min,
                j_min - real_roots,
                np.where(real_roots > j_max, real_roots - j_max, 0.0),
            )
            intersection_J = float(real_roots[np.argmin(np.abs(dist_to_range))])

            # Only flag extrapolation if we're far outside the sampled J-range
            j_out_of_range_by = (
                (j_min - intersection_J) if intersection_J < j_min else
                (intersection_J - j_max) if intersection_J > j_max else
                0.0
            )
            needs_extrapolation = bool(j_out_of_range_by > EXTRAP_TOL)

    torque_fit = np.poly1d(np.polyfit(J, torque, fit_degree))
    intersection_torque = float(torque_fit(intersection_J)) if np.isfinite(intersection_J) else float("nan")
    intersection_thrust = float(poly_fit_eq(intersection_J)) if np.isfinite(intersection_J) else float("nan")
    intersection_resistance = float(poly_fit_res_eq(intersection_J)) if np.isfinite(intersection_J) else float("nan")
    
    if needs_extrapolation:
        extrapolation_cases.append(case)
        extrapolation_cases_J.append(intersection_J)
        extrapolation_cases_count += 1

    J_by_case[int(case)] = intersection_J
    torque_by_case[int(case)] = intersection_torque
    constant = intersection_thrust * 0.2 / (2 * np.pi) if np.isfinite(intersection_thrust) else float("nan")
    if np.isfinite(intersection_J) and np.isfinite(intersection_torque) and intersection_torque != 0:
        eff_by_case[int(case)] = constant * intersection_J / intersection_torque
   

    # store in text file the polynomial fit equation, R^2, RMSE, MAE, MAPE, MBE
    thrust_pred = poly_fit_eq(J)
    ss_res = np.sum((thrust - thrust_pred) ** 2)
    ss_tot = np.sum((thrust - np.mean(thrust)) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    rmse = float(np.sqrt(np.mean((thrust - thrust_pred) ** 2)))
    mae = float(np.mean(np.abs(thrust - thrust_pred)))
    mape = float(np.mean(np.abs((thrust - thrust_pred) / thrust)) * 100.0)
    mbe = float(np.mean(thrust - thrust_pred))

    with open(OUTPUT_FILE, 'a') as f:
        f.write(f'Case {int(case)}\n')
        f.write(f'Thrust fit ({fit_label}): {poly_fit_eq_str}\n')
        f.write(f'Resistance fit ({fit_label}): {poly_fit_res_eq_str}\n')
        f.write(f'Polynomial fit R^2: {r2}\n')
        f.write(f'Polynomial fit RMSE: {rmse}\n')
        f.write(f'Polynomial fit MAE: {mae}\n')
        f.write(f'Polynomial fit MAPE: {mape}%\n')
        f.write(f'Polynomial fit MBE: {mbe}\n')
        f.write(f'Needs extrapolation: {needs_extrapolation}\n')
        f.write(f'Intersection J: {intersection_J}\n')
        f.write(f'Intersection torque: {intersection_torque}\n')
        f.write(f'Intersection thrust: {intersection_thrust}\n')
        f.write(f'Intersection total_resistance: {intersection_resistance}\n')
        f.write(f'Efficiency: {eff_by_case.get(int(case), float("nan"))}\n')
        f.write('\n')
    
    # plot the thrust & resistance curves + intersection for each sample
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # raw data
    # (markers only; no linear connections)
    ax.plot(J, thrust, linestyle="None", marker="o", markersize=DATA_MARKERSIZE, label="thrust (data)")
    ax.plot(
        J,
        total_resistance,
        linestyle="None",
        marker="s",
        markersize=DATA_MARKERSIZE,
        label="total_resistance (data)",
    )

    # smooth quadratic fits
    j_grid = np.linspace(j_min, j_max, 200)
    ax.plot(j_grid, poly_fit_eq(j_grid), "-", linewidth=CURVE_LINEWIDTH, label="thrust (quadratic fit)")
    ax.plot(
        j_grid,
        poly_fit_res_eq(j_grid),
        "-",
        linewidth=CURVE_LINEWIDTH,
        label="total_resistance (quadratic fit)",
    )

    if np.isfinite(intersection_J):
        ax.plot(
            intersection_J,
            intersection_thrust,
            linestyle="None",
            marker="o",
            markersize=INTERSECTION_MARKERSIZE,
            color="g",
            label="intersection (fit=fit)",
        )
        ax.axvline(intersection_J, color="g", alpha=0.25, linewidth=2)

    # ax.set_title(f"Case {int(case)}", fontname=FONT_FAMILY, fontsize=TITLE_FONTSIZE)
    ax.set_title("Optimized Propeller", fontname=FONT_FAMILY, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("J", fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Force (N)", fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(prop={"family": FONT_FAMILY, "size": LEGEND_FONTSIZE})
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=0.35)
    fig.subplots_adjust(**SUBPLOT_ADJUST)

    if SAVE_PLOTS:
        fig.savefig(os.path.join(PLOT_DIR, f'case_{int(case)}.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()

    # Close the figure to avoid "More than 20 figures have been opened" warning
    plt.close(fig)

    # plt.figure()
    # plt.xlim([0.59, 0.75])
    # plt.plot(J, thrust, '-o', label='thrust')
    # plt.plot(intersection_J, Target_thrust, 'og', label='target thrust')
    # #plt.plot(J, kQ, '-*', label='kQ')
    # plt.title(f'Case {int(case)}')
    # plt.xlabel('J')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

# Keep cases that have a control-point row and a finite J/torque intersection.
valid_cases = [
    int(c) for c in sorted(cp_row_by_case)
    if np.isfinite(J_by_case.get(int(c), np.nan))
    and np.isfinite(torque_by_case.get(int(c), np.nan))
]

training_rows = [
    np.concatenate([
        control_points[cp_row_by_case[c]],
        [J_by_case[c], eff_by_case.get(c, np.nan)],
    ])
    for c in valid_cases
]
training_data = (
    np.vstack(training_rows)
    if training_rows
    else np.empty((0, control_points.shape[1] + 2), dtype=float)
)

if SAVE_TRAINING_DATA:
    save_table(TRAINING_OUTPUT, training_data)
    print(f"Wrote {training_data.shape[0]} training rows to {TRAINING_OUTPUT}")

with open(OUTPUT_FILE, 'a') as f:
    f.write('Summary\n')
    f.write(f'Total control-point cases: {num_cases_total}\n')
    f.write(f'Cases present in results: {len(cases)}\n')
    f.write(f'Missing cases in results: {missing_cases}\n')
    f.write(f'Cases needing extrapolation: {sorted(set(extrapolation_cases))}\n')
    f.write(f'J values for cases needing extrapolation: {extrapolation_cases_J}\n')
    f.write(f'Number of cases needing extrapolation: {extrapolation_cases_count}\n')
    f.write(f'Cases fit with a line (2 J points): {sorted(set(linear_fit_cases))}\n')
    f.write(f'Cases with >3 rows (duplicates/extra points): {sorted(set(extra_rows_cases))}\n')