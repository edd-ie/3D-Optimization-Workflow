"""
Infill suggestion script for Prop_hull (dynamic optimized-acquisition structure).

Trains two scalar Gaussian Processes (sklearn):
- GP_J: predicts J
- GP_EFF: predicts efficiency (optimization objective)

Infill batch selection mirrors Training_dynamic_infill_latest.py (v1.8):
- Build an optimized candidate pool via ESPSOLS niching: EI (explore/exploit balancer),
  MU (pure exploitation), elite block-perturbations (local exploitation).
- Progress signal drives EI exploration increment xi; long stall triggers surge mode
  (global bounds, MaxMSE pool, full batch cap).
- Greedy diversity + marginal-EI filter with a parallel batch floor and backfill.
- Predict J at selected points with GP_J.

Input data format (per row):
  [11 control-point inputs] [J] [efficiency]
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold
from scipy.stats import norm

from ESPSOLS import ESPSOLS
from bound_check import bound_check
from pipeline_config import CONPOINT_TRAINING_NAMES, DATA_DIR, INFILL_ROUND, infill_paths, training_data_files
from pipeline_io import load_table, save_table, split_case_ids


# ============================== I/O ==============================

PATHS = infill_paths()
OUT_CONTROL_POINTS = str(PATHS["control_points"])
OUT_CONTROL_POINTS_CON = str(PATHS["control_points_con"])
OUT_PREDICTIONS = str(PATHS["predictions"])
MAKE_MAX_PRED_MU_PLOT = True
MAKE_GENERATION_HISTORY = True

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_MAX_PRED_MU_HISTORY = os.path.join(_THIS_DIR, "max_pred_mu_history.txt")
OUT_MAX_PRED_MU_PLOT = os.path.join(_THIS_DIR, "max_pred_mu_history.png")
OUT_GENERATION_HISTORY = str(DATA_DIR / "generation_history.txt")
OUT_GENERATION_HISTORY_PLOT = str(DATA_DIR / "generation_history.png")

# ============================== Config (aligned with Training_dynamic_infill_latest.py v1.8) ==============================

GENERATE_INFILL = True
BATCH_CAP = 45
# Minimum number of infill points selected per round (keeps rounds few and large).
BATCH_MIN = 6
BATCH_FLOOR_PROGRESS = 6
BATCH_FLOOR_STALL = max(BATCH_MIN, min(BATCH_CAP, BATCH_CAP // 2))
MARGINAL_UTILITY_TAU = 0.45
MARGINAL_UTILITY_TAU_SURGE = 0.0
PROGRESS_WINDOW = 4
PENALTY_RADIUS_LS = 0.25
PENALTY_RADIUS_BACKFILL = 0.10
SURGE_STALL_GENS = 10
GLOBAL_MINSEP_EI = 0.00005
GLOBAL_MINSEP_MU = 0.00005
GLOBAL_MINSEP_BEST = 0.00005
LS_PREF_BEST_AFTER_LOOP = 10

# Optimized-acquisition pool sizes (niche optima per acquisition).
ACQ_NICHE_EI = 20
ACQ_NICHE_MU = 20
ACQ_NICHE_BEST = 20
ACQ_NICHE_MSE = 5

XI_EXPLORE_MIN = 0.0
XI_EXPLORE_MAX = 0.5
EI_XI = 0.01  # diagnostic EI increment for reporting only

ESPSOLS_SWARM_SIZE = 60
ESPSOLS_NUM_ITER = 100
ESPSOLS_R_SCALE = 0.05

# Global-SD probe (matches Training_dynamic_infill_latest CONV_SD_NREF idea).
GLOBAL_SD_NREF = 512
GLOBAL_SD_SEED = 4242

# ------------------------------ Calibration / robustness knobs ------------------------------
# These help mitigate overconfidence (too-small predicted std) observed in practice.
CALIBRATE_EFF_UNCERTAINTY = True
CALIBRATION_FOLDS = 5
CALIBRATION_SEED = 0

# ------------------------------ RNG / reproducibility knobs ------------------------------
SEED_HISTORY_BASE = 12846
SEED_EI_BASE = 1501
SEED_MU_BASE = 80
SEED_BEST_BASE = 9101
SEED_FILL_BASE = 5601

# Noise floors (raise these if the GP is overconfident / optimistic)
ALPHA_FLOOR = 1e-4
ALPHA_REPLICATE_MIN = 1e-8
WHITE_NOISE_INIT = 1e-5
WHITE_NOISE_LOWER = 1e-5

# ------------------------------ Failure imputation (mirror Training_dynamic_infill_latest.py) ------------------------------
# Failed designs (attempted but no valid CFD result: CAD reject, mesh/solve failure) are
# added back as pessimistic pseudo-observations so the acquisition avoids those regions.
ENABLE_IMPUTATION = True
IMPUTED_ALPHA = 1e-1       # large alpha for pseudo-observations from failed cases
IMPUTE_KAPPA = 4.0         # pessimistic imputation: mu - kappa * sd
FAIL_FLOOR_PAD = 0.25      # downward padding (in std units) below low successful values
FAIL_CAP_PAD = 0.10        # extra downward margin (in std units) on the imputed upper cap
DESIGN_MATCH_DECIMALS = 6  # rounding used to match attempted vs successful design rows

# Expanded after infill5: top designs were pinned at box edges (see analyze_bound_saturation.py).
# Only the saturated edge of each dim is moved; pitch_p4x / chord_p4x left unchanged (bimodal / balanced).
pitch_bounds = np.array([
    [0.60, 1.25],  # p1y     was [0.80, 1.25]; 93% of top-15 at low
    [0.00, 1.00],  # y4frac  was [0.05, 1.00]; 60% of top-15 at low
    [0.35, 0.75],  # p4x     unchanged (bimodal lo/hi among elites)
    [0.05, 0.50],  # d1      was [0.15, 0.50]; 47% of top-15 at low
    [0.05, 0.50],  # d2      was [0.15, 0.50]; 53% of top-15 at low
    [0.40, 0.85],  # p7y     was [0.40, 0.70]; 80% of top-15 at high
])

chord_bounds = np.array([
    [0.15, 0.30],  # p1y     was [0.20, 0.30]; 47% of top-15 at low
    [0.45, 0.85],  # p4x     unchanged (not saturated)
    [0.05, 0.50],  # d1      was [0.15, 0.50]; 73% of top-15 at low
    [0.25, 0.70],  # y4      was [0.40, 0.70]; 87% of top-15 at low
    [0.05, 0.30],  # w56     was [0.10, 0.30]; 53% of top-15 at low
])
bounds_all = np.vstack([pitch_bounds, chord_bounds])

# ============================== Representation ==============================
#
# IMPORTANT:
# - `training_data*.dat` inputs in this repo are con-points (NOT design-vars):
#   pitch: [p1y, y4, p4x, p2x, p5x, p7y]
#   chord: [p1y, p4x, p2x, y4, w56]
# - ESPSOLS bounds below are design-vars:
#   pitch: [p1y, y4_frac, p4x, d1, d2, p7y]
#   chord: [p1y, p4x, d1, y4, w56]
#
# We convert training inputs con-points -> design-vars before fitting the GP and
# before passing evaluated_X to ESPSOLS.

R1_FIXED = 0.17
R7_FIXED = 0.999
MAX_PITCH = 1.4


def conpoints_to_designvars(X_con: np.ndarray) -> np.ndarray:
    """Convert con-points (what training_data stores) -> design-vars (what bounds_all uses)."""
    X = np.asarray(X_con, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, got {X.shape[1]}")

    # pitch con-points
    p_p1y = X[:, 0]
    p_y4 = X[:, 1]
    p_p4x = X[:, 2]
    p_p2x = X[:, 3]
    p_p5x = X[:, 4]
    p_p7y = X[:, 5]

    # chord con-points
    c_p1y = X[:, 6]
    c_p4x = X[:, 7]
    c_p2x = X[:, 8]
    c_y4 = X[:, 9]
    c_w56 = X[:, 10]

    # pitch design-vars
    denom_y4 = np.maximum(MAX_PITCH - p_p1y, 1e-12)
    y4_frac = (p_y4 - p_p1y) / denom_y4
    denom_d1 = np.maximum(p_p4x - R1_FIXED, 1e-12)
    d1_pitch = (p_p4x - p_p2x) / denom_d1
    denom_d2 = np.maximum(R7_FIXED - p_p4x, 1e-12)
    d2_pitch = (p_p5x - p_p4x) / denom_d2

    # chord design-vars
    denom_cd1 = np.maximum(c_p4x - R1_FIXED, 1e-12)
    d1_chord = (c_p4x - c_p2x) / denom_cd1

    X_dv = np.column_stack([
        p_p1y, y4_frac, p_p4x, d1_pitch, d2_pitch, p_p7y,
        c_p1y, c_p4x, d1_chord, c_y4, c_w56,
    ])
    return X_dv


def designvars_to_conpoints(X_dv: np.ndarray) -> np.ndarray:
    """Convert design-vars -> con-points (for optional output and sanity checking)."""
    X = np.asarray(X_dv, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 11:
        raise ValueError(f"Expected 11 columns, got {X.shape[1]}")

    # pitch design-vars
    p_p1y = X[:, 0]
    y4_frac = X[:, 1]
    p_p4x = X[:, 2]
    d1_pitch = X[:, 3]
    d2_pitch = X[:, 4]
    p_p7y = X[:, 5]

    # chord design-vars
    c_p1y = X[:, 6]
    c_p4x = X[:, 7]
    d1_chord = X[:, 8]
    c_y4 = X[:, 9]
    c_w56 = X[:, 10]

    # pitch con-points
    p_y4 = p_p1y + (MAX_PITCH - p_p1y) * y4_frac
    p_p2x = p_p4x - (p_p4x - R1_FIXED) * d1_pitch
    p_p5x = p_p4x + (R7_FIXED - p_p4x) * d2_pitch

    # chord con-points
    c_p2x = c_p4x - (c_p4x - R1_FIXED) * d1_chord

    X_con = np.column_stack([
        p_p1y, p_y4, p_p4x, p_p2x, p_p5x, p_p7y,
        c_p1y, c_p4x, c_p2x, c_y4, c_w56,
    ])
    return X_con


# ============================== Helpers (ported from Training_updated.py) ==============================

def fit_gp(x_train_u: np.ndarray, y_train: np.ndarray, is_imputed=None, init_kernel=None):
    """Fit a stable GP in unit space with per-location noise from replicates.
    Per-sample alpha is computed on normalized y to match normalize_y=True.
    Imputed failed points get a large alpha so they discourage re-sampling without
    dominating the fit (they are pseudo-observations, not real CFD data).
    """
    X = np.asarray(x_train_u, dtype=float)
    y = np.asarray(y_train, dtype=float)

    if is_imputed is None:
        is_imputed = np.zeros(y.shape[0], dtype=bool)
    else:
        is_imputed = np.asarray(is_imputed, dtype=bool).ravel()
    if y.shape[0] != is_imputed.shape[0]:
        raise ValueError("y_train and is_imputed must have the same length.")

    # Normalize y for alpha construction (GPR will normalize internally too)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0)) or 1.0
    y_norm = (y - y_mean) / y_std

    # Per-sample alpha from replicate variance (group identical X rounded in unit space)
    alpha = np.full(y.shape[0], float(ALPHA_FLOOR), dtype=float)
    if X.size > 0:
        keys = [tuple(row) for row in np.round(X, 8)]
        groups = {}
        for idx, k in enumerate(keys):
            groups.setdefault(k, []).append(idx)
        for idxs in groups.values():
            if len(idxs) >= 2:
                v = float(np.var(y_norm[idxs], ddof=1))  # variance in normalized space
                alpha[idxs] = max(v, float(ALPHA_REPLICATE_MIN))

    # Imputed failures are pseudo-observations: give them a large alpha.
    alpha[is_imputed] = np.maximum(alpha[is_imputed], float(IMPUTED_ALPHA))

    d = int(X.shape[1])
    base_kernel = (
        C(1.0, (1e-3, 1e2))
        * Matern(length_scale=[1.0] * d, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        + WhiteKernel(noise_level=float(WHITE_NOISE_INIT), noise_level_bounds=(float(WHITE_NOISE_LOWER), 1e-1))
    )
    kernel = init_kernel if init_kernel is not None else base_kernel

    gp = GPR(kernel=kernel, alpha=alpha, n_restarts_optimizer=25, normalize_y=True, random_state=0)
    gp.fit(X, y)
    return gp


def design_key(row, decimals=DESIGN_MATCH_DECIMALS):
    """Hashable rounded key for matching design-variable rows across files."""
    return tuple(np.round(np.asarray(row, dtype=float).ravel(), decimals))


def collect_failed_designs(successful_X, n_vars, max_round=None):
    """Return attempted design rows that produced no successful CFD result.

    A round is considered "evaluated" only if its training_data file exists, so the
    current (not-yet-run) round's freshly generated CADs are never flagged as failures.
    Failures are inferred as: attempted designs (initial samples for round 0; CAD
    accepted + CAD rejected for infill rounds) whose design vars are absent from the
    successful training set.

    max_round: if set, only consider rounds 0..max_round (inclusive). Otherwise use
    INFILL_ROUND as the upper bound.
    """
    success_keys = {design_key(row) for row in np.asarray(successful_X, dtype=float)}
    failed_by_key = {}
    last_round = int(INFILL_ROUND) if max_round is None else int(max_round)

    for r in range(last_round + 1):
        paths = infill_paths(r)
        if not paths["training_data"].is_file():
            continue  # round has no results yet -> do not treat its CADs as failures

        attempted_files = (
            [paths["control_points"]]
            if r == 0
            else [paths["cad_accepted"], paths["cad_rejected"]]
        )
        for path in attempted_files:
            if not (path.is_file() and path.stat().st_size > 0):
                continue
            _, design_vars = split_case_ids(load_table(path), n_vars)
            for row in np.asarray(design_vars, dtype=float):
                key = design_key(row)
                if key not in success_keys:
                    failed_by_key.setdefault(key, row[:n_vars])

    if not failed_by_key:
        return np.empty((0, n_vars), dtype=float)
    return np.vstack(list(failed_by_key.values()))


def stage_tag_from_training_path(path):
    """Map training_data_initial.dat -> 'initial', training_data_infill3.dat -> 'infill3'."""
    name = Path(path).stem  # training_data_initial / training_data_infill3
    if name == "training_data_initial":
        return "initial"
    if name.startswith("training_data_infill"):
        return "infill" + name.replace("training_data_infill", "", 1)
    return name


def round_num_from_stage_tag(tag):
    """initial -> 0, infill3 -> 3."""
    if tag == "initial":
        return 0
    if tag.startswith("infill"):
        return int(tag.replace("infill", "", 1))
    raise ValueError(f"Unknown stage tag: {tag}")


def count_attempted_designs(max_round, n_vars):
    """Count unique attempted designs through round max_round (inclusive)."""
    keys = set()
    for r in range(int(max_round) + 1):
        paths = infill_paths(r)
        # Only count rounds that already have training results (except we always
        # count initial control points when the initial training file exists).
        if not paths["training_data"].is_file():
            continue
        attempted_files = (
            [paths["control_points"]]
            if r == 0
            else [paths["cad_accepted"], paths["cad_rejected"]]
        )
        for path in attempted_files:
            if not (path.is_file() and path.stat().st_size > 0):
                continue
            _, design_vars = split_case_ids(load_table(path), n_vars)
            for row in np.asarray(design_vars, dtype=float):
                keys.add(design_key(row[:n_vars]))
    return len(keys)


def impute_failed_efficiencies(gp_success, to_unit_fn, failed_X, y_success):
    """Pessimistic imputed efficiency values for failed designs (reference formula)."""
    y_success = np.asarray(y_success, dtype=float).ravel()
    y_best_success = float(np.max(y_success))
    y_std_success = float(np.std(y_success)) or 1.0
    if y_success.size >= 5:
        y_fail_floor = float(np.percentile(y_success, 5) - FAIL_FLOOR_PAD * y_std_success)
        y_fail_cap = float(np.percentile(y_success, 50) - FAIL_CAP_PAD * y_std_success)
    else:
        y_fail_floor = float(np.min(y_success) - FAIL_FLOOR_PAD * y_std_success)
        y_fail_cap = float(np.min(y_success))
    y_fail_cap = min(y_fail_cap, y_best_success - FAIL_FLOOR_PAD * y_std_success)

    f_mu, f_sd = gp_success.predict(to_unit_fn(failed_X), return_std=True)
    pessimistic = np.asarray(f_mu, dtype=float) - IMPUTE_KAPPA * np.asarray(f_sd, dtype=float)
    imputed_vals = np.minimum(np.maximum(pessimistic, y_fail_floor), y_fail_cap)
    imputed_vals = np.minimum(imputed_vals, y_best_success - FAIL_FLOOR_PAD * y_std_success)
    return imputed_vals, y_fail_floor, y_fail_cap


class CalibratedStdWrapper:
    """Wrap a sklearn GP so return_std is scaled (mu unchanged)."""

    def __init__(self, gp, sd_scale: float):
        self.gp = gp
        self.sd_scale = float(sd_scale)
        # Proxy kernel_ for length-scale extraction, etc.
        self.kernel_ = getattr(gp, "kernel_", None)

    def predict(self, X, return_std=False):
        if return_std:
            mu, sd = self.gp.predict(X, return_std=True)
            return mu, np.asarray(sd, dtype=float) * self.sd_scale
        return self.gp.predict(X, return_std=False)


def calibrate_sd_scale_cv(X_train_u: np.ndarray, y: np.ndarray, folds: int, seed: int) -> float:
    """Estimate a multiplicative scale for predicted std via CV residuals."""
    X = np.asarray(X_train_u, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = int(X.shape[0])
    if n < max(5, int(folds) * 2):
        return 1.0

    kf = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    abs_z = []
    rmse_terms = []
    sig_terms = []

    for tr_idx, te_idx in kf.split(X):
        gp = fit_gp(X[tr_idx], y[tr_idx], init_kernel=None)
        mu, sd = gp.predict(X[te_idx], return_std=True)
        sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
        r = (y[te_idx] - np.asarray(mu, dtype=float))
        abs_z.append(np.abs(r) / sd)
        rmse_terms.append(np.mean(r ** 2))
        sig_terms.append(np.mean(sd ** 2))

    abs_z = np.concatenate(abs_z) if abs_z else np.array([1.0])
    q95 = float(np.quantile(abs_z, 0.95)) if abs_z.size else 2.0
    # scale so that ~95% of points fall within ±2σ: q95 / s ≈ 2  => s ≈ q95/2
    s_cov2 = max(q95 / 2.0, 1e-6)

    rmse = float(np.sqrt(np.mean(rmse_terms))) if rmse_terms else 0.0
    sig = float(np.sqrt(np.mean(sig_terms))) if sig_terms else 0.0
    s_rmse = (rmse / sig) if sig > 0 else 1.0

    # Conservative: choose the larger multiplier.
    s = float(max(s_cov2, s_rmse, 1.0))
    return s


def extract_length_scale(kernel):
    """Recursively extract a length_scale array from a composite kernel."""
    try:
        if hasattr(kernel, "length_scale"):
            return np.asarray(kernel.length_scale, dtype=float)
        for attr in ("k1", "k2"):
            if hasattr(kernel, attr):
                ls = extract_length_scale(getattr(kernel, attr))
                if ls is not None:
                    return ls
    except Exception:
        pass
    return None


def closeness(existing_pts_orig, new_pts_orig, ls, to_unit, minsep=0):
    existing_pts_orig = np.asarray(existing_pts_orig)
    new_pts_orig = np.asarray(new_pts_orig)
    if existing_pts_orig.size == 0 or new_pts_orig.size == 0:
        return new_pts_orig

    ep_u = to_unit(existing_pts_orig)
    np_u = to_unit(new_pts_orig)
    ep_ls = ep_u / ls
    keep = []
    for i in range(np_u.shape[0]):
        d = np.linalg.norm(ep_ls - (np_u[i] / ls), axis=1)
        if np.all(d >= minsep):
            keep.append(i)
    if not keep:
        return np.empty((0, existing_pts_orig.shape[1]), dtype=float)
    return new_pts_orig[np.array(keep, dtype=int)]


def espsols_opt(gp, search_bounds, y_best, opt, swarm_size, num_iter, r, run_seed=None, to_unit_fn=None):
    """Optimize a GP acquisition using ESPSOLS; returns (X_phys, scores)."""
    if run_seed is not None:
        np.random.seed(int(run_seed))
    try:
        X_phys, scores, _ = ESPSOLS(
            gp=gp,
            num_var=search_bounds.shape[0],
            bounds=search_bounds,
            y_best=float(y_best),
            swarm_size=int(swarm_size),
            num_iter=int(num_iter),
            r=r,
            opt=str(opt),
            type_of_Problem="max",
        )
        return np.asarray(X_phys, dtype=float), np.asarray(scores, dtype=float).ravel()
    except Exception as _es_err:
        print(f"[WARN] ESPSOLS failed (opt={opt}): {_es_err}. Falling back to random scoring.")
        d = int(search_bounds.shape[0])
        n_fallback = max(int(swarm_size), 2 * d, 20)
        if run_seed is not None:
            np.random.seed(int(run_seed))
        batch = bound_check(10000 + (run_seed or 0), search_bounds, d, n_fallback)
        if batch.size == 0:
            return np.empty((0, d), dtype=float), np.empty((0,), dtype=float)
        unit_fn = to_unit_fn if to_unit_fn is not None else (lambda X: X)
        mu_fb, sd_fb = gp.predict(unit_fn(batch), return_std=True)
        opt_key = str(opt).lower()
        if opt_key == "ei":
            scores = expected_improvement(mu_fb, sd_fb, y_best)
        elif opt_key == "maxmse":
            scores = np.asarray(sd_fb, dtype=float).ravel() ** 2
        else:
            scores = np.asarray(mu_fb, dtype=float).ravel()
        return batch, np.asarray(scores, dtype=float).ravel()


def collect_espsols_topk(
    gp,
    search_bounds,
    y_best,
    opt,
    k_needed,
    swarm_size,
    num_iter,
    r,
    evaluated_X,
    ls,
    to_unit_fn,
    minsep=0,
    max_tries=50,
    stall_limit=15,
    seed_base=0,
):
    """
    Collect >=k_needed unique candidate points by calling ESPSOLS repeatedly and
    using ALL niche results from each run (ordered by score). Keep running ESPSOLS
    until k_needed is met (or until max_tries is reached as a safety cap).

    For opt="EI" and opt="MaxMSE", larger score is better.
    """
    chosen = np.empty((0, search_bounds.shape[0]), dtype=float)
    tries = 0
    stalls = 0
    max_tries = int(max_tries)
    stall_limit = int(stall_limit)

    while chosen.shape[0] < k_needed and tries < max_tries:
        X_niche, scores = espsols_opt(
            gp, search_bounds, y_best=y_best, opt=opt,
            swarm_size=swarm_size, num_iter=num_iter, r=r,
            run_seed=int(seed_base + tries), to_unit_fn=to_unit_fn,
        )
        if X_niche.size == 0:
            tries += 1
            continue

        order = np.argsort(scores)[::-1]
        X_sorted = X_niche[order]

        added = 0
        for x in X_sorted:
            if chosen.shape[0] >= k_needed:
                break
            cand = x.reshape(1, -1)
            tmp_eval = np.vstack([evaluated_X, chosen]) if chosen.size else evaluated_X
            cand = closeness(tmp_eval, cand, ls=ls, to_unit=to_unit_fn, minsep=float(minsep))
            if cand.size == 0:
                continue
            chosen = np.vstack([chosen, cand])
            added += 1

        if added == 0:
            stalls += 1
            if stalls >= stall_limit:
                break
        else:
            stalls = 0
        tries += 1

    return chosen[:k_needed]


def load_control_points_file(path: str, d: int) -> np.ndarray:
    try:
        arr = load_table(path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not read '{path}'.") from e
    if arr.shape[1] != d:
        raise RuntimeError(f"{path} must have {d} columns, got {arr.shape[1]}")
    return arr


def fill_with_random_minsep(existing_X, need, bounds, ls, to_unit_fn, minsep):
    """Fallback filler: random LHS points filtered by minsep vs existing."""
    if need <= 0:
        return np.empty((0, bounds.shape[0]), dtype=float)
    d = int(bounds.shape[0])
    chosen = np.empty((0, d), dtype=float)
    tries = 0
    max_tries = 200
    while chosen.shape[0] < need and tries < max_tries:
        batch = bound_check(10000 + tries, bounds, d, max(need * 5, 20))
        tmp_eval = np.vstack([existing_X, chosen]) if chosen.size else existing_X
        batch = closeness(tmp_eval, batch, ls=ls, to_unit=to_unit_fn, minsep=float(minsep))
        if batch.size:
            chosen = np.vstack([chosen, batch])
        tries += 1
    return chosen[:need]


def expected_improvement(mu, sd, y_best, xi=EI_XI):
    """Expected improvement for maximizing efficiency."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (mu - float(y_best) - float(xi)) / sigma
    return (mu - float(y_best) - float(xi)) * norm.cdf(z) + sigma * norm.pdf(z)


# ============================== Dynamic infill engine (Training_dynamic_infill_latest v1.8) ==============================


def progress_signal(y_best_history, counter_stall, window=4, y_scale=1.0):
    w = int(max(1, int(window)))
    stall_p = float(np.exp(-float(max(0, int(counter_stall))) / float(w)))
    hist = np.asarray(y_best_history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size >= 2:
        k = int(max(1, min(w, hist.size - 1)))
        gain = float(hist[-1] - hist[-1 - k])
        scale = float(y_scale) if (np.isfinite(y_scale) and y_scale > 0.0) else 1.0
        imp_p = float(np.clip(gain / (0.05 * scale + 1e-12), 0.0, 1.0))
    else:
        imp_p = 1.0
    return float(np.clip(0.5 * stall_p + 0.5 * imp_p, 0.0, 1.0))


def exploration_xi(p, xi_min, xi_max, y_scale):
    p = float(np.clip(p, 0.0, 1.0))
    scale = float(y_scale) if (np.isfinite(y_scale) and y_scale > 0.0) else 1.0
    return float(xi_min + (xi_max - xi_min) * (1.0 - p)) * scale


def ei_scores(gp, X, y_ref, to_unit_fn, xi=0.0):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.empty((0,), dtype=float)
    mu, sd = gp.predict(to_unit_fn(X), return_std=True)
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.maximum(np.asarray(sd, dtype=float).ravel(), 1e-12)
    imp = mu - float(y_ref) - float(xi)
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def ls_normalized(X, to_unit_fn, ls, d):
    lsa = np.asarray(ls, dtype=float).ravel()
    if lsa.size == 1:
        lsa = np.full(d, float(lsa[0]), dtype=float)
    lsa = np.where(lsa > 1e-12, lsa, 1e-12)
    return np.asarray(to_unit_fn(X), dtype=float).reshape(-1, d) / lsa


def min_dist_to_set(Z, Zset):
    if Zset.shape[0] == 0:
        return np.full(Z.shape[0], np.inf, dtype=float)
    z2 = np.sum(Z * Z, axis=1)[:, None]
    s2 = np.sum(Zset * Zset, axis=1)[None, :]
    d2 = np.maximum(z2 + s2 - 2.0 * (Z @ Zset.T), 0.0)
    return np.sqrt(d2.min(axis=1))


def greedy_penalized_batch(pool, scores, origins, ls, to_unit_fn, evaluated_X, y_ref,
                           cap, tau, radius_ls, minsep, batch_min=1, batch_floor=1):
    pool = np.asarray(pool, dtype=float)
    d = int(pool.shape[1]) if pool.ndim == 2 else 0
    n = int(pool.shape[0])
    if n == 0:
        return np.empty((0, d), dtype=float), np.empty((0,), dtype=object), 0

    scores = np.asarray(scores, dtype=float).ravel()
    origins = np.asarray(origins, dtype=object).ravel()
    cap = int(max(int(batch_min), int(cap)))
    batch_floor = int(max(int(batch_min), int(batch_floor)))
    tau = float(tau)

    Z = ls_normalized(pool, to_unit_fn, ls, d)
    if evaluated_X is not None and np.asarray(evaluated_X).size:
        Ze = ls_normalized(evaluated_X, to_unit_fn, ls, d)
        available = min_dist_to_set(Z, Ze) >= float(minsep)
    else:
        available = np.ones(n, dtype=bool)

    sel = []
    u_first = None
    while len(sel) < cap and bool(np.any(available)):
        masked = np.where(available, scores, -np.inf)
        j = int(np.argmax(masked))
        if not np.isfinite(masked[j]):
            break
        u_j = float(scores[j] - y_ref)
        if u_first is None:
            u_first = u_j
        elif len(sel) >= batch_floor:
            if tau <= 0.0:
                pass
            elif u_first <= 1e-12 or u_j < tau * u_first:
                break
        sel.append(j)
        available[j] = False
        too_close = np.linalg.norm(Z - Z[j], axis=1) < float(radius_ls)
        available[too_close] = False

    if not sel:
        return np.empty((0, d), dtype=float), np.empty((0,), dtype=object), 0
    sel = np.asarray(sel, dtype=int)
    return pool[sel], origins[sel], int(sel.size)


def point_is_diverse(x_row, selected_X, evaluated_X, ls, to_unit_fn, radius_ls, minsep):
    d = int(np.asarray(x_row).shape[-1])
    z = ls_normalized(np.asarray(x_row, dtype=float).reshape(1, -1), to_unit_fn, ls, d)
    if selected_X is not None and np.asarray(selected_X).size:
        zsel = ls_normalized(selected_X, to_unit_fn, ls, d)
        if float(min_dist_to_set(z, zsel)[0]) < float(radius_ls):
            return False
    if evaluated_X is not None and np.asarray(evaluated_X).size:
        ze = ls_normalized(evaluated_X, to_unit_fn, ls, d)
        if float(min_dist_to_set(z, ze)[0]) < float(minsep):
            return False
    return True


def backfill_batch_to_floor(selected_X, selected_origins, floor, cap, candidate_blocks,
                            ls, to_unit_fn, evaluated_X, radius_ls, minsep):
    selected_X = np.asarray(selected_X, dtype=float)
    if selected_X.ndim == 1 and selected_X.size:
        selected_X = selected_X.reshape(1, -1)
    elif selected_X.size == 0:
        selected_X = np.empty((0, int(ls.size) if np.asarray(ls).size else 0), dtype=float)
    selected_origins = np.asarray(selected_origins, dtype=object).ravel()
    floor = int(max(1, floor))
    cap = int(max(floor, cap))

    for X_block, tag in candidate_blocks:
        X_block = np.asarray(X_block, dtype=float)
        if X_block.size == 0:
            continue
        if X_block.ndim == 1:
            X_block = X_block.reshape(1, -1)
        for i in range(X_block.shape[0]):
            if selected_X.shape[0] >= floor:
                return selected_X, selected_origins
            if selected_X.shape[0] >= cap:
                return selected_X, selected_origins
            cand = X_block[i:i + 1]
            if not point_is_diverse(cand, selected_X, evaluated_X, ls, to_unit_fn, radius_ls, minsep):
                continue
            selected_X = np.vstack([selected_X, cand]) if selected_X.size else cand
            selected_origins = np.concatenate(
                [selected_origins, np.array([str(tag)], dtype=object)]
            )
    return selected_X, selected_origins


def elite_local_bounds(X_success, y_success, global_bounds, top_n=25, pad_frac=0.25, min_width_frac=0.05):
    X_success = np.asarray(X_success, dtype=float)
    y_success = np.asarray(y_success, dtype=float).ravel()
    global_bounds = np.asarray(global_bounds, dtype=float)
    d = int(global_bounds.shape[0])
    lo_g = global_bounds[:, 0]
    hi_g = global_bounds[:, 1]
    span_g = hi_g - lo_g
    span_g[span_g == 0.0] = 1.0
    if X_success.ndim != 2 or X_success.shape[0] == 0 or y_success.size == 0:
        return global_bounds.copy()
    top_n = int(max(1, min(int(top_n), X_success.shape[0], y_success.size)))
    elite = X_success[np.argsort(y_success)[::-1][:top_n]]
    lo_l = np.percentile(elite, 15.0, axis=0)
    hi_l = np.percentile(elite, 85.0, axis=0)
    width = np.maximum(hi_l - lo_l, float(min_width_frac) * span_g)
    center = 0.5 * (lo_l + hi_l)
    lo_new = np.maximum(center - 0.5 * width * (1.0 + float(pad_frac)), lo_g)
    hi_new = np.minimum(center + 0.5 * width * (1.0 + float(pad_frac)), hi_g)
    lo_new = np.minimum(lo_new, hi_new)
    return np.column_stack([lo_new, hi_new]).reshape(d, 2)


def local_bounds_are_usable(local_bounds, global_bounds):
    local_bounds = np.asarray(local_bounds, dtype=float)
    global_bounds = np.asarray(global_bounds, dtype=float)
    if local_bounds.shape != global_bounds.shape:
        return False
    local_w = local_bounds[:, 1] - local_bounds[:, 0]
    global_w = np.maximum(global_bounds[:, 1] - global_bounds[:, 0], 1e-12)
    if np.any(local_w <= 1e-12) or not np.all(np.isfinite(local_w)):
        return False
    width_ratio = local_w / global_w
    if float(np.sum(np.log10(np.clip(width_ratio, 1e-12, None)))) < -35.0:
        return False
    if float(np.mean(width_ratio < 0.02)) > 0.50:
        return False
    return True


def adaptive_elite_local_bounds(X_success, y_success, global_bounds, top_n=25):
    for pad_frac, min_w in [(0.20, 0.04), (0.35, 0.08), (0.60, 0.12), (1.00, 0.20)]:
        local = elite_local_bounds(
            X_success, y_success, global_bounds, top_n=top_n,
            pad_frac=pad_frac, min_width_frac=min_w,
        )
        if local_bounds_are_usable(local, global_bounds):
            return local, ("local", float(pad_frac), float(min_w))
    return np.asarray(global_bounds, dtype=float).copy(), ("global", None, None)


def topk_block_perturbed_from_ytrain(
    x_train_orig, y_train, k_needed, bounds, seed, ls=None,
    perturb_frac=0.02, min_dims=2, max_dims=5, N_loops=None,
):
    d = int(bounds.shape[0])
    if k_needed <= 0:
        return np.empty((0, d), dtype=float)
    X = np.asarray(x_train_orig, dtype=float).reshape(-1, d)
    y = np.asarray(y_train, dtype=float).ravel()
    if X.shape[0] == 0 or y.size == 0:
        return np.empty((0, d), dtype=float)

    lo_b = np.asarray(bounds[:, 0], dtype=float)
    hi_b = np.asarray(bounds[:, 1], dtype=float)
    span_b = hi_b - lo_b
    span_b[span_b == 0.0] = 1.0
    X_u = (X - lo_b) / span_b

    elite_pool = min(10 if N_loops is not None and int(N_loops) >= 10 else 20, X.shape[0], y.size)
    elite_u = X_u[np.argsort(y)[::-1][:elite_pool]]
    rng = np.random.RandomState(int(seed))

    if ls is None:
        ls_arr = None
        step_u = np.full(d, float(perturb_frac), dtype=float)
    else:
        ls_arr = np.asarray(ls, dtype=float).ravel()
        if ls_arr.size == 1:
            ls_arr = np.full(d, float(ls_arr[0]), dtype=float)
        step_u = np.maximum(float(perturb_frac) * np.clip(ls_arr, 0.10, 1.00), 0.003)

    pts_u = np.empty((0, d), dtype=float)
    min_active = max(1, min(int(min_dims), d))
    max_active = max(min_active, min(int(max_dims), d))

    while pts_u.shape[0] < int(k_needed):
        base_u = elite_u[pts_u.shape[0] % elite_u.shape[0]].copy()
        n_active = int(rng.randint(min_active, max_active + 1))
        if (
            N_loops is not None
            and int(N_loops) >= int(LS_PREF_BEST_AFTER_LOOP)
            and ls_arr is not None
        ):
            importance = 1.0 / np.clip(ls_arr, 0.025, 5.0)
            prob = 0.75 * (importance / np.sum(importance)) + 0.25 * (1.0 / d)
            prob = prob / np.sum(prob)
            active_dims = rng.choice(d, size=n_active, replace=False, p=prob)
        else:
            active_dims = rng.choice(d, size=n_active, replace=False)
        jitter = np.zeros(d, dtype=float)
        jitter[active_dims] = rng.normal(0.0, step_u[active_dims])
        pts_u = np.vstack([pts_u, np.clip(base_u + jitter, 0.0, 1.0).reshape(1, -1)])

    return lo_b + pts_u * span_b


def stall_state_from_history(y_blocks):
    """Derive y_best_history and counter_stall from cumulative training blocks."""
    y_best_history = []
    counter_stall = 0
    y_best_prev = float("-inf")
    tol = 1e-8
    for i in range(len(y_blocks)):
        cum = np.concatenate([
            np.asarray(y_blocks[j], dtype=float).ravel() for j in range(i + 1)
        ])
        y_best = float(np.max(cum))
        y_best_history.append(y_best)
        if y_best > y_best_prev + tol:
            counter_stall = 0
            y_best_prev = y_best
        else:
            counter_stall += 1
    return y_best_history, counter_stall


def select_dynamic_infill_batch(
    gp_eff_cal,
    gp_train_X,
    gp_train_y,
    global_bounds,
    to_unit_fn,
    ls,
    y_best,
    y_best_history,
    counter_stall,
    n_loops,
    seed_ei,
    seed_mu,
    seed_best,
    seed_fill,
):
    """Build the next infill batch using the optimized-acquisition dynamic engine."""
    y_scale = float(np.std(gp_train_y)) or 1.0
    progress_p = progress_signal(y_best_history, counter_stall, PROGRESS_WINDOW, y_scale)
    xi_explore = exploration_xi(progress_p, XI_EXPLORE_MIN, XI_EXPLORE_MAX, y_scale)
    surge_active = bool(counter_stall >= int(SURGE_STALL_GENS))
    r_vec = ESPSOLS_R_SCALE * ls
    evaluated_X = np.asarray(gp_train_X, dtype=float)
    d = int(global_bounds.shape[0])

    if surge_active:
        search_bounds = global_bounds
        batch_floor_run = int(BATCH_CAP)
        tau_run = float(MARGINAL_UTILITY_TAU_SURGE)
        print(f"[surge] counter_stall={counter_stall} -> batch_floor={BATCH_CAP}, global bounds")
    elif counter_stall >= 3:
        search_bounds, local_info = adaptive_elite_local_bounds(
            evaluated_X, gp_train_y, global_bounds, top_n=25,
        )
        batch_floor_run = int(BATCH_FLOOR_PROGRESS if progress_p > 0.5 else BATCH_FLOOR_STALL)
        tau_run = float(MARGINAL_UTILITY_TAU)
        src, pad_used, minw_used = local_info
        if src == "local":
            print(f"[local-bounds] stall={counter_stall} pad={pad_used:.2f} min_w={minw_used:.2f}")
    else:
        search_bounds = global_bounds
        batch_floor_run = int(BATCH_FLOOR_PROGRESS if progress_p > 0.5 else BATCH_FLOOR_STALL)
        tau_run = float(MARGINAL_UTILITY_TAU)

    X_ei_pts = collect_espsols_topk(
        gp=gp_eff_cal, search_bounds=search_bounds, y_best=y_best + xi_explore, opt="EI",
        k_needed=ACQ_NICHE_EI, swarm_size=ESPSOLS_SWARM_SIZE, num_iter=ESPSOLS_NUM_ITER, r=r_vec,
        evaluated_X=evaluated_X, ls=ls, to_unit_fn=to_unit_fn, minsep=GLOBAL_MINSEP_EI,
        seed_base=int(seed_ei),
    )
    eval_for_mu = np.vstack([evaluated_X, X_ei_pts]) if X_ei_pts.size else evaluated_X
    X_mu_pts = collect_espsols_topk(
        gp=gp_eff_cal, search_bounds=search_bounds, y_best=y_best, opt="mu",
        k_needed=ACQ_NICHE_MU, swarm_size=ESPSOLS_SWARM_SIZE, num_iter=ESPSOLS_NUM_ITER, r=r_vec,
        evaluated_X=eval_for_mu, ls=ls, to_unit_fn=to_unit_fn, minsep=GLOBAL_MINSEP_MU,
        seed_base=int(seed_mu),
    )
    X_best_pts = topk_block_perturbed_from_ytrain(
        x_train_orig=evaluated_X, y_train=gp_train_y, k_needed=ACQ_NICHE_BEST,
        bounds=search_bounds, seed=int(seed_best), ls=ls,
        perturb_frac=0.02, min_dims=2, max_dims=5, N_loops=n_loops,
    )

    X_mse_pts = np.empty((0, d), dtype=float)
    if surge_active:
        eval_for_mse = evaluated_X
        if X_ei_pts.size:
            eval_for_mse = np.vstack([eval_for_mse, X_ei_pts])
        if X_mu_pts.size:
            eval_for_mse = np.vstack([eval_for_mse, X_mu_pts]) if eval_for_mse.size else X_mu_pts
        X_mse_pts = collect_espsols_topk(
            gp=gp_eff_cal, search_bounds=global_bounds, y_best=y_best, opt="MaxMSE",
            k_needed=ACQ_NICHE_MSE, swarm_size=ESPSOLS_SWARM_SIZE, num_iter=ESPSOLS_NUM_ITER, r=r_vec,
            evaluated_X=eval_for_mse, ls=ls, to_unit_fn=to_unit_fn, minsep=GLOBAL_MINSEP_EI,
            seed_base=int(seed_fill),
        )

    pool_blocks = []
    origin_blocks = []
    for Xb, tag in (
        (X_ei_pts, "EI"), (X_mu_pts, "MU"), (X_best_pts, "ELITE"), (X_mse_pts, "SPACE"),
    ):
        Xb = np.asarray(Xb, dtype=float)
        if Xb.size:
            pool_blocks.append(Xb)
            origin_blocks.append(np.full(Xb.shape[0], tag, dtype=object))
    pool = np.vstack(pool_blocks) if pool_blocks else np.empty((0, d), dtype=float)
    pool_origin = np.concatenate(origin_blocks).astype(object) if pool_blocks else np.empty((0,), dtype=object)

    if pool.shape[0] == 0:
        filler = bound_check(int(seed_fill) + 9000, global_bounds, d, BATCH_CAP)
        return filler, np.full(filler.shape[0], "FILL", dtype=object), {
            "progress_p": progress_p, "xi_explore": xi_explore, "surge_active": surge_active,
            "batch_floor_run": batch_floor_run, "pool_size": 0,
        }

    pool_ei = ei_scores(gp_eff_cal, pool, y_best + xi_explore, to_unit_fn)
    ei_order = np.argsort(pool_ei)[::-1]
    pool_ei_sorted = pool[ei_order]
    if X_mu_pts.size:
        mu_vals = gp_eff_cal.predict(to_unit_fn(X_mu_pts))
        mu_sorted = np.asarray(X_mu_pts, dtype=float)[np.argsort(np.asarray(mu_vals).ravel())[::-1]]
    else:
        mu_sorted = np.empty((0, d), dtype=float)

    training_pts, source_labels, k_emergent = greedy_penalized_batch(
        pool, pool_ei, pool_origin, ls=ls, to_unit_fn=to_unit_fn, evaluated_X=evaluated_X,
        y_ref=0.0, cap=BATCH_CAP, tau=tau_run, radius_ls=PENALTY_RADIUS_LS,
        minsep=GLOBAL_MINSEP_EI, batch_min=BATCH_MIN, batch_floor=batch_floor_run,
    )

    if training_pts.shape[0] < batch_floor_run:
        backfill_blocks = [(pool_ei_sorted, "EI"), (mu_sorted, "MU"), (X_best_pts, "ELITE")]
        if X_mse_pts.size:
            backfill_blocks.append((X_mse_pts, "SPACE"))
        training_pts, source_labels = backfill_batch_to_floor(
            training_pts, source_labels, batch_floor_run, BATCH_CAP, backfill_blocks,
            ls=ls, to_unit_fn=to_unit_fn, evaluated_X=evaluated_X,
            radius_ls=PENALTY_RADIUS_BACKFILL, minsep=GLOBAL_MINSEP_EI,
        )
        k_emergent = int(training_pts.shape[0])

    if training_pts.shape[0] < batch_floor_run:
        n_lhs = int(batch_floor_run - training_pts.shape[0])
        lhs_fill = bound_check(int(seed_fill) + 3000, global_bounds, d, max(n_lhs * 3, 20))
        training_pts, source_labels = backfill_batch_to_floor(
            training_pts, source_labels, batch_floor_run, BATCH_CAP, [(lhs_fill, "FILL")],
            ls=ls, to_unit_fn=to_unit_fn, evaluated_X=evaluated_X,
            radius_ls=PENALTY_RADIUS_BACKFILL, minsep=GLOBAL_MINSEP_EI,
        )
        k_emergent = int(training_pts.shape[0])

    print(
        f"[infill] p={progress_p:.2f} xi={xi_explore:.3f} floor={batch_floor_run} "
        f"surge={int(surge_active)} pool={pool.shape[0]} k={k_emergent}/{BATCH_CAP}"
    )
    meta = {
        "progress_p": progress_p,
        "xi_explore": xi_explore,
        "surge_active": surge_active,
        "batch_floor_run": batch_floor_run,
        "pool_size": int(pool.shape[0]),
        "k_emergent": k_emergent,
    }
    return training_pts, source_labels, meta


# ============================== Main ==============================

def main():
    def prep_dataset(arr: np.ndarray, inputs_are_conpoints: bool):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 13:
            raise RuntimeError("training_data must have 13 columns: 11 inputs + J + efficiency")

        X_raw = np.asarray(arr[:, :11], dtype=float)
        yJ_blk = np.asarray(arr[:, 11], dtype=float).ravel()
        yEff_blk = np.asarray(arr[:, 12], dtype=float).ravel()
        X_blk = conpoints_to_designvars(X_raw) if inputs_are_conpoints else X_raw
        return X_blk, yJ_blk, yEff_blk

    training_files = training_data_files()
    if not training_files:
        raise FileNotFoundError(f"No training_data*.dat files found under {DATA_DIR}")

    X_blocks = []
    yJ_blocks = []
    yE_blocks = []
    stage_tags = []
    for path in training_files:
        block = load_table(path)
        use_conpoints = path.name in CONPOINT_TRAINING_NAMES
        X_blk, yJ_blk, yE_blk = prep_dataset(block, use_conpoints)
        tag = stage_tag_from_training_path(path)
        X_blocks.append(X_blk)
        yJ_blocks.append(yJ_blk)
        yE_blocks.append(yE_blk)
        stage_tags.append(tag)
        print(
            f"Loaded {path.name}: {X_blk.shape[0]} successful rows "
            f"(stage={tag}, {'con-points' if use_conpoints else 'design-vars'})"
        )
    print(f"Stage order: {stage_tags}")

    # Full training data (all blocks)
    X_train = np.vstack(X_blocks)
    yJ = np.concatenate(yJ_blocks).ravel()
    yEff = np.concatenate(yE_blocks).ravel()

    eff_max = np.max(yEff)
    eff_max_index = np.argmax(yEff)
    print(f"Efficiency max measured so far: {eff_max}")
    print(f"Efficiency max index: {eff_max_index}")

    # unit-cube scaling bounds from known control bounds (per-dimension)
    lo = bounds_all[:, 0].reshape(1, -1)
    hi = bounds_all[:, 1].reshape(1, -1)
    span = (hi - lo).reshape(1, -1)
    span[span == 0.0] = 1.0

    def to_unit(X):
        return (np.asarray(X, dtype=float) - lo) / span

    X_train_u = to_unit(X_train)

    # # Sanity check: if training conversion is wildly out of bounds, warn early.
    # if TRAINING_INPUTS_ARE_CONPOINTS and X_train.shape[0] > 0:
    #     mins = X_train.min(axis=0)
    #     maxs = X_train.max(axis=0)
    #     if np.any(mins < bounds_all[:, 0] - 1e-6) or np.any(maxs > bounds_all[:, 1] + 1e-6):
    #         print("[WARN] Converted training inputs exceed design-var bounds_all.")
    #         print("       min per dim:", mins)
    #         print("       max per dim:", maxs)

    # Train two separate scalar GPs (same training structure as Training_updated.py).
    # Fit on successful data first; gp_eff here doubles as the imputation model below.
    gp_eff = fit_gp(X_train_u, yEff, init_kernel=None)
    gp_J = fit_gp(X_train_u, yJ, init_kernel=None)

    # Calibrate uncertainty for EI/MaxMSE (scale predicted std). This affects EI behavior directly.
    # Calibrate on successful data only, before adding imputed pseudo-observations.
    eff_sd_scale = 1.0
    if CALIBRATE_EFF_UNCERTAINTY:
        eff_sd_scale = calibrate_sd_scale_cv(X_train_u, yEff, folds=CALIBRATION_FOLDS, seed=CALIBRATION_SEED)
        print(f"[CAL] eff_pred_sd scale factor: {eff_sd_scale:.6g} (CV folds={CALIBRATION_FOLDS})")

    # ------------------------------ Failure imputation ------------------------------
    # Add failed designs (attempted but no valid CFD result) back as pessimistic
    # pseudo-observations so EI/MU steer away from failure-prone regions.
    is_imputed_train = np.zeros(X_train.shape[0], dtype=bool)
    if ENABLE_IMPUTATION:
        failed_X = collect_failed_designs(X_train, X_train.shape[1])
        if failed_X.shape[0] > 0:
            y_success = np.asarray(yEff, dtype=float)
            y_best_success = float(np.max(y_success))
            y_std_success = float(np.std(y_success)) or 1.0

            # Conservative floor and cap so imputed values are clearly below successes
            # (percentile + std-scaled pads mirror Training_dynamic_infill_latest.py).
            if y_success.size >= 5:
                y_fail_floor = float(np.percentile(y_success, 5) - FAIL_FLOOR_PAD * y_std_success)
                y_fail_cap = float(np.percentile(y_success, 50) - FAIL_CAP_PAD * y_std_success)
            else:
                y_fail_floor = float(np.min(y_success) - FAIL_FLOOR_PAD * y_std_success)
                y_fail_cap = float(np.min(y_success))
            y_fail_cap = min(y_fail_cap, y_best_success - FAIL_FLOOR_PAD * y_std_success)

            # Pessimistic imputation from the success-only GP: mu - kappa * sd.
            f_mu, f_sd = gp_eff.predict(to_unit(failed_X), return_std=True)
            pessimistic = np.asarray(f_mu, dtype=float) - IMPUTE_KAPPA * np.asarray(f_sd, dtype=float)
            imputed_vals = np.minimum(np.maximum(pessimistic, y_fail_floor), y_fail_cap)
            # Hard safety: an imputed failure must never beat an observed success.
            imputed_vals = np.minimum(imputed_vals, y_best_success - FAIL_FLOOR_PAD * y_std_success)

            X_train = np.vstack([X_train, failed_X])
            yEff = np.concatenate([y_success, imputed_vals])
            is_imputed_train = np.concatenate(
                [is_imputed_train, np.ones(failed_X.shape[0], dtype=bool)]
            )
            X_train_u = to_unit(X_train)
            gp_eff = fit_gp(X_train_u, yEff, is_imputed=is_imputed_train, init_kernel=None)
            print(
                f"[impute] {failed_X.shape[0]} failed design(s) added as pseudo-observations; "
                f"imputed eff in [{float(imputed_vals.min()):.4f}, {float(imputed_vals.max()):.4f}] "
                f"(floor={y_fail_floor:.4f}, cap={y_fail_cap:.4f}, best_success={y_best_success:.4f})"
            )
        else:
            print("[impute] no failed designs detected among evaluated rounds.")

    gp_eff_cal = CalibratedStdWrapper(gp_eff, sd_scale=eff_sd_scale)

    # ============================== Per-generation history ==============================
    # Generations follow the campaign order:
    #   gen 1 = after initial sampling (50 attempted designs)
    #   gen 2 = after infill1, ...
    # Each stage fits a GP on cumulative successful CFD + stage-scoped failure imputation.
    # Reported counts:
    #   n_success / n_imputed / n_gp (= success+imputed) / n_attempted
    max_mu_hist = []
    best_ei_hist = []
    generation_hist = []
    rng_state = np.random.get_state()
    try:
        # Fixed probe set for global-SD (same points every generation).
        np.random.seed(int(GLOBAL_SD_SEED))
        lo_b = bounds_all[:, 0]
        hi_b = bounds_all[:, 1]
        conv_ref_X = lo_b + np.random.rand(int(GLOBAL_SD_NREF), bounds_all.shape[0]) * (hi_b - lo_b)

        for it in range(len(X_blocks)):
            stage = stage_tags[it]
            round_num = round_num_from_stage_tag(stage)
            X_success = np.vstack(X_blocks[: it + 1])
            yE_success = np.concatenate(yE_blocks[: it + 1]).ravel()
            n_success = int(X_success.shape[0])
            y_best_it = float(np.max(yE_success))

            # Success-only GP, then (optionally) impute failures known through this stage.
            gp_success = fit_gp(to_unit(X_success), yE_success, init_kernel=None)
            X_fit = X_success
            yE_fit = yE_success
            is_imputed_it = np.zeros(n_success, dtype=bool)
            n_imputed = 0
            if ENABLE_IMPUTATION:
                failed_X = collect_failed_designs(
                    X_success, X_success.shape[1], max_round=round_num
                )
                n_imputed = int(failed_X.shape[0])
                if n_imputed > 0:
                    imputed_vals, _, _ = impute_failed_efficiencies(
                        gp_success, to_unit, failed_X, yE_success
                    )
                    X_fit = np.vstack([X_success, failed_X])
                    yE_fit = np.concatenate([yE_success, imputed_vals])
                    is_imputed_it = np.concatenate(
                        [is_imputed_it, np.ones(n_imputed, dtype=bool)]
                    )
                    gp_eff_it = fit_gp(
                        to_unit(X_fit), yE_fit, is_imputed=is_imputed_it, init_kernel=None
                    )
                else:
                    gp_eff_it = gp_success
            else:
                gp_eff_it = gp_success

            n_attempted = count_attempted_designs(round_num, X_success.shape[1])
            # Fallback if control-point files are missing: success + imputed.
            if n_attempted < n_success + n_imputed:
                n_attempted = n_success + n_imputed

            gp_eff_it_cal = CalibratedStdWrapper(gp_eff_it, sd_scale=eff_sd_scale)

            ls_it = extract_length_scale(gp_eff_it.kernel_)
            if ls_it is None:
                ls_it = np.ones(X_fit.shape[1], dtype=float)
            else:
                ls_it = np.asarray(ls_it, dtype=float).ravel()
                if ls_it.size == 1:
                    ls_it = np.full(X_fit.shape[1], float(ls_it[0]), dtype=float)

            r_vec_it = ESPSOLS_R_SCALE * ls_it
            y_scale_it = float(np.std(yE_success)) or 1.0

            # --- max predicted mean + SD at that point ---
            np.random.seed(int(SEED_HISTORY_BASE) + int(it))
            X_mu, scores_mu = espsols_opt(
                gp_eff_it_cal,
                bounds_all,
                y_best=y_best_it,
                opt="mu",
                swarm_size=ESPSOLS_SWARM_SIZE,
                num_iter=ESPSOLS_NUM_ITER,
                r=r_vec_it,
            )

            if X_mu.size == 0:
                cand = bound_check(9 + it, bounds_all, bounds_all.shape[0], 5000)
                mu_c, sd_c = gp_eff_it_cal.predict(to_unit(cand), return_std=True)
                j_best = int(np.argmax(mu_c))
                x_mu_best = np.asarray(cand[j_best], dtype=float)
                mu_best = float(mu_c[j_best])
                sd_best = float(sd_c[j_best])
            else:
                j_best = int(np.argmax(scores_mu))
                x_mu_best = np.asarray(X_mu[j_best], dtype=float).ravel()
                mu_best_arr, sd_best_arr = gp_eff_it_cal.predict(
                    to_unit(x_mu_best.reshape(1, -1)), return_std=True
                )
                mu_best = float(mu_best_arr[0])
                sd_best = float(sd_best_arr[0])

            max_mu_hist.append((it + 1, n_success + n_imputed, mu_best, sd_best))

            # --- max EI ---
            np.random.seed(int(SEED_HISTORY_BASE) + 1000 + int(it))
            X_ei, scores_ei = espsols_opt(
                gp_eff_it_cal,
                bounds_all,
                y_best=y_best_it,
                opt="EI",
                swarm_size=ESPSOLS_SWARM_SIZE,
                num_iter=ESPSOLS_NUM_ITER,
                r=r_vec_it,
            )

            if X_ei.size == 0:
                cand = bound_check(9009 + it, bounds_all, bounds_all.shape[0], 5000)
                mu_c, sd_c = gp_eff_it_cal.predict(to_unit(cand), return_std=True)
                ei_c = expected_improvement(mu_c, sd_c, y_best_it)
                j_ei = int(np.argmax(ei_c))
                x_ei_best = np.asarray(cand[j_ei], dtype=float)
                ei_best = float(ei_c[j_ei])
                mu_ei_best = float(mu_c[j_ei])
                sd_ei_best = float(sd_c[j_ei])
            else:
                j_ei = int(np.argmax(scores_ei))
                x_ei_best = np.asarray(X_ei[j_ei], dtype=float).ravel()
                mu_ei_arr, sd_ei_arr = gp_eff_it_cal.predict(
                    to_unit(x_ei_best.reshape(1, -1)), return_std=True
                )
                mu_ei_best = float(mu_ei_arr[0])
                sd_ei_best = float(sd_ei_arr[0])
                ei_best = float(scores_ei[j_ei])

            best_ei_hist.append(
                {
                    "iter": int(it + 1),
                    "stage": stage,
                    "n_train": n_success + n_imputed,
                    "x": x_ei_best,
                    "eff_pred_mu": mu_ei_best,
                    "eff_pred_sd": sd_ei_best,
                    "ei": ei_best,
                }
            )

            # --- global SD: mean predictive SD on fixed reference set ---
            _, sd_ref = gp_eff_it_cal.predict(to_unit(conv_ref_X), return_std=True)
            global_sd = float(np.mean(np.asarray(sd_ref, dtype=float).ravel()))
            global_sd_norm = global_sd / y_scale_it

            # SD of GP at the best *measured* design so far
            i_meas = int(np.argmax(yE_success))
            _, sd_at_meas = gp_eff_it_cal.predict(
                to_unit(X_success[i_meas].reshape(1, -1)), return_std=True
            )
            sd_at_meas = float(np.asarray(sd_at_meas).ravel()[0])

            generation_hist.append(
                {
                    "gen": int(it + 1),
                    "stage": stage,
                    "n_attempted": int(n_attempted),
                    "n_success": n_success,
                    "n_imputed": n_imputed,
                    "n_gp": n_success + n_imputed,
                    "y_best": y_best_it,
                    "max_pred_mu": mu_best,
                    "best_pt_sd": sd_best,
                    "sd_at_measured_best": sd_at_meas,
                    "ei_max": ei_best,
                    "global_sd": global_sd,
                    "global_sd_norm": global_sd_norm,
                }
            )
            print(
                f"[hist] gen {it + 1} ({stage}): attempted={n_attempted}, "
                f"success={n_success}, imputed={n_imputed}, y_best={y_best_it:.6f}"
            )
    finally:
        np.random.set_state(rng_state)

    hist_arr = np.asarray(max_mu_hist, dtype=float)
    out_arr = np.column_stack(
        [
            hist_arr[:, 0],
            hist_arr[:, 1],
            hist_arr[:, 2],
            hist_arr[:, 3],
            hist_arr[:, 2] - 2.0 * hist_arr[:, 3],
            hist_arr[:, 2] + 2.0 * hist_arr[:, 3],
        ]
    )
    np.savetxt(
        OUT_MAX_PRED_MU_HISTORY,
        out_arr,
        fmt="%.0f\t%.0f\t%.10g\t%.10g\t%.10g\t%.10g",
        header="iter\tn_gp\tmax_pred_mu\tpred_sd\tmu_minus_2sd\tmu_plus_2sd",
        comments="",
    )

    # Consolidated generation history table (text, with stage tags)
    os.makedirs(os.path.dirname(OUT_GENERATION_HISTORY), exist_ok=True)
    with open(OUT_GENERATION_HISTORY, "w", encoding="utf-8") as hist_file:
        hist_file.write(
            "gen\tstage\tn_attempted\tn_success\tn_imputed\tn_gp\t"
            "y_best\tmax_pred_mu\tbest_pt_sd\tsd_at_measured_best\t"
            "ei_max\tglobal_sd\tglobal_sd_norm\n"
        )
        for r in generation_hist:
            hist_file.write(
                f"{r['gen']}\t{r['stage']}\t{r['n_attempted']}\t{r['n_success']}\t"
                f"{r['n_imputed']}\t{r['n_gp']}\t"
                f"{r['y_best']:.10g}\t{r['max_pred_mu']:.10g}\t{r['best_pt_sd']:.10g}\t"
                f"{r['sd_at_measured_best']:.10g}\t{r['ei_max']:.10g}\t"
                f"{r['global_sd']:.10g}\t{r['global_sd_norm']:.10g}\n"
            )

    print("\nPer-generation history (initial = 50 attempted, then each infill round):")
    print(
        f"  {'gen':>3} {'stage':>10} {'att':>4} {'ok':>4} {'imp':>4} "
        f"{'y_best':>10} {'max_mu':>10} {'sd*':>10} {'EI_max':>10} {'gSD_n':>8}"
    )
    for r in generation_hist:
        print(
            f"  {r['gen']:3d} {r['stage']:>10} {r['n_attempted']:4d} {r['n_success']:4d} "
            f"{r['n_imputed']:4d} {r['y_best']:10.6f} {r['max_pred_mu']:10.6f} "
            f"{r['best_pt_sd']:10.6f} {r['ei_max']:10.6f} {r['global_sd_norm']:8.4f}"
        )
    print(f"[WROTE] {OUT_GENERATION_HISTORY}")

    print("\nMax predicted mean history (mu ± 2sd):")
    for it, ntr, mu_b, sd_b in max_mu_hist:
        print(f"  iter {int(it)} (n_gp={int(ntr)}): mu={mu_b:.10g}, sd={sd_b:.10g}")

    print("\nBest EI point history:")
    for row in best_ei_hist:
        x_str = np.array2string(row["x"], precision=10, separator=", ")
        print(
            f"  iter {row['iter']} ({row['stage']}, n_gp={row['n_train']}): "
            f"EI={row['ei']:.10g}, sd={row['eff_pred_sd']:.10g}, "
            f"mu={row['eff_pred_mu']:.10g}, x={x_str}"
        )

    if MAKE_MAX_PRED_MU_PLOT:
        iters = out_arr[:, 0]
        mu_b = out_arr[:, 2]
        sd_b = out_arr[:, 3]

        fig_h, ax_h = plt.subplots(figsize=(10, 6))
        ax_h.plot(iters, mu_b, "ko-", linewidth=2, label=r"$\mu$")
        ax_h.fill_between(iters, mu_b - 2 * sd_b, mu_b + 2 * sd_b, color="k", alpha=0.15, label=r"$\mu \pm 2\sigma$")
        ax_h.set_xlabel("Training iteration")
        ax_h.set_ylabel("max predicted mean efficiency")
        ax_h.set_title(r"Max predicted mean ($\mu$) with $\pm 2\sigma$ vs iteration")
        ax_h.grid(True, alpha=0.3)
        ax_h.legend()
        fig_h.tight_layout()
        fig_h.savefig(OUT_MAX_PRED_MU_PLOT, dpi=200)
        print(f"[WROTE] {OUT_MAX_PRED_MU_PLOT}")

    if MAKE_GENERATION_HISTORY and generation_hist:
        gens = np.array([r["gen"] for r in generation_hist], dtype=float)
        y_best_arr = np.array([r["y_best"] for r in generation_hist], dtype=float)
        max_mu_arr = np.array([r["max_pred_mu"] for r in generation_hist], dtype=float)
        sd_star = np.array([r["best_pt_sd"] for r in generation_hist], dtype=float)
        sd_meas = np.array([r["sd_at_measured_best"] for r in generation_hist], dtype=float)
        ei_arr = np.array([r["ei_max"] for r in generation_hist], dtype=float)
        gsd_arr = np.array([r["global_sd"] for r in generation_hist], dtype=float)
        gsdn_arr = np.array([r["global_sd_norm"] for r in generation_hist], dtype=float)
        stage_labels = [r["stage"] for r in generation_hist]

        fig_g, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        axes[0, 0].plot(gens, y_best_arr, "ko-", lw=2, label=r"$y_{best}$ (measured)")
        axes[0, 0].plot(gens, max_mu_arr, "rs--", lw=1.5, label=r"max $\mu$ (GP)")
        axes[0, 0].set_ylabel("Efficiency")
        axes[0, 0].set_title("Best discovered vs max predicted μ")
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(gens, sd_star, "bo-", lw=2, label=r"SD at max-$\mu$")
        axes[0, 1].plot(gens, sd_meas, "g^--", lw=1.5, label="SD at measured best")
        axes[0, 1].set_ylabel("Predictive SD")
        axes[0, 1].set_title("Best-point SD")
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(gens, ei_arr, "mo-", lw=2)
        axes[1, 0].set_xlabel("Generation")
        axes[1, 0].set_ylabel("EI max")
        axes[1, 0].set_title("Maximum Expected Improvement")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(gens, gsd_arr, "co-", lw=2, label="global SD")
        axes[1, 1].plot(gens, gsdn_arr, "k^--", lw=1.5, label="global SD / y_std")
        axes[1, 1].set_xlabel("Generation")
        axes[1, 1].set_ylabel("Global SD")
        axes[1, 1].set_title("Global predictive SD (LHS probe)")
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        for ax in (axes[1, 0], axes[1, 1]):
            ax.set_xticks(gens)
            ax.set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=8)

        fig_g.suptitle(
            "Per-generation history (gen1=initial/50 attempted, then infills)",
            fontsize=13,
            y=0.98,
        )
        fig_g.tight_layout(rect=[0, 0, 1, 0.96])
        fig_g.savefig(OUT_GENERATION_HISTORY_PLOT, dpi=200)
        print(f"[WROTE] {OUT_GENERATION_HISTORY_PLOT}")
        plt.close(fig_g)

    # Use efficiency GP for acquisition with ESPSOLS
    ls = extract_length_scale(gp_eff.kernel_)
    if ls is None:
        ls = np.ones(X_train.shape[1], dtype=float)
    else:
        ls = np.asarray(ls, dtype=float).ravel()
        if ls.size == 1:
            ls = np.full(X_train.shape[1], float(ls[0]), dtype=float)

    y_best = float(np.max(yEff))
    y_best_history, counter_stall = stall_state_from_history(yE_blocks)
    n_loops = len(X_blocks)

    if GENERATE_INFILL:
        X_new, source_labels, infill_meta = select_dynamic_infill_batch(
            gp_eff_cal=gp_eff_cal,
            gp_train_X=X_train,
            gp_train_y=yEff,
            global_bounds=bounds_all,
            to_unit_fn=to_unit,
            ls=ls,
            y_best=y_best,
            y_best_history=y_best_history,
            counter_stall=counter_stall,
            n_loops=n_loops,
            seed_ei=SEED_EI_BASE + 100 * n_loops,
            seed_mu=SEED_MU_BASE + 100 * n_loops,
            seed_best=SEED_BEST_BASE + 100 * n_loops,
            seed_fill=SEED_FILL_BASE + 100 * n_loops,
        )
        X_new = closeness(X_train, X_new, ls=ls, to_unit=to_unit, minsep=GLOBAL_MINSEP_EI)
        X_new_u = to_unit(X_new)

        J_mu, _ = gp_J.predict(X_new_u, return_std=True)
        eff_mu, eff_sd = gp_eff_cal.predict(X_new_u, return_std=True)
        sigma_sel = np.maximum(eff_sd, 1e-12)
        z_sel = (eff_mu - y_best - EI_XI) / sigma_sel
        ei_sel = (eff_mu - y_best - EI_XI) * norm.cdf(z_sel) + sigma_sel * norm.pdf(z_sel)

        if X_new.shape[0] > 0:
            i_best = int(np.argmax(eff_mu))
            print("Predicted best point (max eff_pred_mu):")
            print(f"  index: {i_best}")
            print(f"  x*: {X_new[i_best]}")
            print(f"  source: {source_labels[i_best]}")
            print(f"  eff_pred_mu: {eff_mu[i_best]:.16g}")
            print(f"  eff_pred_sd: {eff_sd[i_best]:.16g}")
            print(f"  J_pred_mu: {J_mu[i_best]:.16g}")

        save_table(
            OUT_CONTROL_POINTS,
            X_new,
            header="cp1 cp2 cp3 cp4 cp5 cp6 cp7 cp8 cp9 cp10 cp11",
        )
        try:
            X_new_con = designvars_to_conpoints(X_new)
            save_table(
                OUT_CONTROL_POINTS_CON,
                X_new_con,
                header="p_p1y p_y4 p_p4x p_p2x p_p5x p_p7y c_p1y c_p4x c_p2x c_y4 c_w56",
            )
            print(f"Suggested control points (con-points): {OUT_CONTROL_POINTS_CON}")
        except Exception as e:
            print(f"[WARN] Could not write con-point suggestions: {e}")

        save_table(
            OUT_PREDICTIONS,
            np.column_stack([J_mu, eff_mu, eff_sd, ei_sel]),
            fmt="%.16g",
            header="J_pred_mu eff_pred_mu eff_pred_sd EI_eff",
        )

        print(f"Loaded training rows: {X_train.shape[0]}")
        print(f"Suggested control points: {OUT_CONTROL_POINTS}")
        print(f"Suggested predictions: {OUT_PREDICTIONS}")
        print(
            f"Suggested count: {X_new.shape[0]} (cap={BATCH_CAP}, floor={infill_meta['batch_floor_run']}, "
            f"pool={infill_meta['pool_size']})"
        )
        src_counts = {}
        for s in source_labels:
            src_counts[str(s)] = src_counts.get(str(s), 0) + 1
        print(f"Source mix: {src_counts}")

    print("\nFinal GP kernel parameters:")
    print(f"  GP_EFF kernel_: {gp_eff.kernel_}")
    print(f"  GP_EFF log_marginal_likelihood: {gp_eff.log_marginal_likelihood_value_:.10g}")
    print(f"  GP_J kernel_: {gp_J.kernel_}")
    print(f"  GP_J log_marginal_likelihood: {gp_J.log_marginal_likelihood_value_:.10g}")


if __name__ == "__main__":
    main()

