import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from coupled_constraint_config import (
    CHORD_P1Y_BOUNDS,
    CHORD_SUM_MAX_NORM,
    CHORD_Y4_BOUNDS,
    PITCH_DIFF_MAX_NORM,
    PITCH_P4X_BOUNDS,
    PITCH_P7Y_BOUNDS,
)


def normalize(val, lo, hi):
    return (val - lo) / (hi - lo)


def denormalize(val_n, lo, hi):
    return lo + val_n * (hi - lo)


# Toggle
# This optimization campaign does NOT use the coupled chord_sum / pitch_diff
# constraints. Keep this False so no caller silently projects/rejects designs
# onto them; only per-variable bounds and blade-clearance rejection apply.
APPLY_COUPLED_CONSTRAINTS = False

# Selected coupled constraints in normalized space
# 1) chord_p1y_norm + chord_y4_norm <= CHORD_SUM_MAX_NORM
# 2) pitch_p4x_norm - pitch_p7y_norm <= PITCH_DIFF_MAX_NORM


def para_control_bez_updated(
    Pitch,
    ChordLength,
    R_values,
    para_control,
    pitch_con,
    chord_con,
    case_id=None,
    reject_log=None,
    constraint_mode="reject",   # "reject" or "project"
    return_reject_info=False,
    apply_coupled_constraints=None,
):
    """
    Parametric control using rational Bézier curves with optional coupled-constraint handling.

    Parameters
    ----------
    Pitch, ChordLength : callable
        Original pitch and chord functions.
    R_values : array_like
        Fixed normalized radial positions.
    para_control : int
        2 = chord only, 3 = pitch only, 7 = both.
    pitch_con, chord_con : array_like
        Design/control variables.
    case_id : any, optional
        Identifier for the design/case being evaluated.
    reject_log : list, optional
        If provided, rejected or violated cases will be appended here as dictionaries.
    constraint_mode : str, optional
        "reject"  -> keep track of rejected cases and return original geometry unchanged
        "project" -> project violating variables back to nearest feasible value and log violation
    return_reject_info : bool, optional
        If True, return a fifth output: reject_info dictionary.
    apply_coupled_constraints : bool or None, optional
        When True, enforce chord_sum and pitch_diff coupled limits.
        When False, only per-variable bounds apply (no projection or rejection).
        When None, use module-level APPLY_COUPLED_CONSTRAINTS.

    Returns
    -------
    Pitch, ChordLength, chord_con_points, pitch_con_points
    or
    Pitch, ChordLength, chord_con_points, pitch_con_points, reject_info
    """

    if constraint_mode not in {"reject", "project"}:
        raise ValueError("constraint_mode must be 'reject' or 'project'")

    use_coupled = (
        APPLY_COUPLED_CONSTRAINTS
        if apply_coupled_constraints is None
        else bool(apply_coupled_constraints)
    )

    # Store original functions
    Pitch_origin = Pitch
    ChordLength_origin = ChordLength
    chord_con_points = np.zeros(5)
    pitch_con_points = np.zeros(6)

    reject_info = {
        "case_id": case_id,
        "constraint_mode": constraint_mode,
        "violated": False,
        "rejected": False,
        "reasons": [],
        "details": {},
    }

    def register_violation(name, detail_dict):
        reject_info["violated"] = True
        reject_info["reasons"].append(name)
        reject_info["details"][name] = detail_dict
        if constraint_mode == "reject":
            reject_info["rejected"] = True

    # -------------------------
    # Chord Control
    # -------------------------
    if para_control == 2 or para_control == 7:
        chord_R1 = R_values[0]
        chord_p1x = R_values[0]  # chord P1.x

        # Variables (chord)
        chord_p1y = chord_con[0]
        chord_p4x = chord_con[1]
        chord_d1 = chord_con[2]
        chord_y4 = chord_con[3]  # chord P4.y

        chord_w23 = 0.4
        chord_w56 = chord_con[4]  # weight at P5=P6 (chord)

        # Coupled constraint:
        # chord_p1y_norm + chord_y4_norm <= CHORD_SUM_MAX_NORM
        if use_coupled:
            chord_p1y_n = normalize(chord_p1y, *CHORD_P1Y_BOUNDS)
            chord_y4_n = normalize(chord_y4, *CHORD_Y4_BOUNDS)
            chord_sum_n = chord_p1y_n + chord_y4_n

            if chord_sum_n > CHORD_SUM_MAX_NORM:
                detail = {
                    "original_chord_p1y": float(chord_p1y),
                    "original_chord_y4": float(chord_y4),
                    "original_chord_p1y_norm": float(chord_p1y_n),
                    "original_chord_y4_norm": float(chord_y4_n),
                    "original_sum_norm": float(chord_sum_n),
                    "limit": float(CHORD_SUM_MAX_NORM),
                }

                if constraint_mode == "project":
                    chord_y4_n_new = max(0.0, min(1.0, CHORD_SUM_MAX_NORM - chord_p1y_n))
                    chord_y4_new = denormalize(chord_y4_n_new, *CHORD_Y4_BOUNDS)
                    detail["projected_chord_y4_norm"] = float(chord_y4_n_new)
                    detail["projected_chord_y4"] = float(chord_y4_new)
                    chord_y4 = chord_y4_new

                register_violation("chord_sum_constraint", detail)

        # Only build updated chord geometry if not rejected
        if not reject_info["rejected"]:
            # Build the seven Bézier anchors (with the required coincidences)
            chord_p2x = chord_p4x - (chord_p4x - chord_R1) * chord_d1  # P2.x = P3.x
            chord_P1 = np.array([chord_p1x, chord_p1y])
            chord_P2 = np.array([chord_p2x, chord_y4])
            chord_P3 = chord_P2  # coincide
            chord_P4 = np.array([chord_p4x, chord_y4])
            chord_P5 = np.array([R_values[-1], chord_y4])
            chord_P6 = chord_P5  # coincide
            chord_P7 = np.array([R_values[-1], ChordLength_origin(R_values[-1])])

            chord_con_points = np.array([chord_p1y, chord_p4x, chord_d1, chord_y4, chord_w56])

            # Homogeneous weights for each 4-point segment
            w_seg1 = np.array([1, chord_w23, chord_w23, 1])
            w_seg2 = np.array([1, chord_w56, chord_w56, 1])

            # Sample each rational Bézier
            u = np.linspace(0, 1, 600)
            R1_s, C1_s = eval_rational_bezier(
                np.column_stack([chord_P1, chord_P2, chord_P3, chord_P4]),
                w_seg1, u,
            )
            R2_s, C2_s = eval_rational_bezier(
                np.column_stack([chord_P4, chord_P5, chord_P6, chord_P7]),
                w_seg2, u,
            )

            # Stitch & interpolate
            chord_R_curve = np.concatenate([R1_s, R2_s[1:]])
            chord_curve = np.concatenate([C1_s, C2_s[1:]])
            chord_interp = interp1d(
                chord_R_curve,
                chord_curve,
                kind="cubic",
                bounds_error=False,
                fill_value=(chord_curve[0], chord_curve[-1]),
            )
            ChordLength = lambda x, _f=chord_interp: _f(x)

    # -------------------------
    # Pitch Control
    # -------------------------
    if para_control == 3 or para_control == 7:
        pitch_R1 = R_values[0]
        pitch_R7 = R_values[-1]
        max_pitch = 1.4

        # Variables (pitch)
        pitch_p1y = pitch_con[0]
        pitch_y4 = pitch_p1y + (max_pitch - pitch_p1y) * pitch_con[1]
        pitch_p4x = pitch_con[2]
        pitch_d1 = pitch_con[3]
        pitch_d2 = pitch_con[4]
        pitch_w23 = 0.4
        pitch_w56 = 0.4
        pitch_p7y = pitch_con[5]

        # Coupled constraint:
        # pitch_p4x_norm - pitch_p7y_norm <= PITCH_DIFF_MAX_NORM
        if use_coupled:
            pitch_p4x_n = normalize(pitch_p4x, *PITCH_P4X_BOUNDS)
            pitch_p7y_n = normalize(pitch_p7y, *PITCH_P7Y_BOUNDS)
            pitch_diff_n = pitch_p4x_n - pitch_p7y_n

            if pitch_diff_n > PITCH_DIFF_MAX_NORM:
                detail = {
                    "original_pitch_p4x": float(pitch_p4x),
                    "original_pitch_p7y": float(pitch_p7y),
                    "original_pitch_p4x_norm": float(pitch_p4x_n),
                    "original_pitch_p7y_norm": float(pitch_p7y_n),
                    "original_diff_norm": float(pitch_diff_n),
                    "limit": float(PITCH_DIFF_MAX_NORM),
                }

                if constraint_mode == "project":
                    pitch_p7y_n_new = max(0.0, min(1.0, pitch_p4x_n - PITCH_DIFF_MAX_NORM))
                    pitch_p7y_new = denormalize(pitch_p7y_n_new, *PITCH_P7Y_BOUNDS)
                    detail["projected_pitch_p7y_norm"] = float(pitch_p7y_n_new)
                    detail["projected_pitch_p7y"] = float(pitch_p7y_new)
                    pitch_p7y = pitch_p7y_new

                register_violation("pitch_diff_constraint", detail)

        # Only build updated pitch geometry if not rejected
        if not reject_info["rejected"]:
            pitch_p2x = pitch_p4x - (pitch_p4x - pitch_R1) * pitch_d1
            pitch_p5x = pitch_p4x + (pitch_R7 - pitch_p4x) * pitch_d2
            pitch_P1 = np.array([pitch_R1, pitch_p1y])
            pitch_P2 = np.array([pitch_p2x, pitch_y4])
            pitch_P3 = pitch_P2
            pitch_P4 = np.array([pitch_p4x, pitch_y4])
            pitch_P5 = np.array([pitch_p5x, pitch_y4])
            pitch_P6 = pitch_P5
            pitch_P7 = np.array([pitch_R7, pitch_p7y])

            pitch_w1 = np.array([1, pitch_w23, pitch_w23, 1])
            pitch_w2 = np.array([1, pitch_w56, pitch_w56, 1])

            pitch_con_points = np.array([
                pitch_p1y, pitch_con[1], pitch_p4x, pitch_d1, pitch_d2, pitch_p7y
            ])

            u = np.linspace(0, 1, 600)
            R1_s, Y1_s = eval_rational_bezier(
                np.column_stack([pitch_P1, pitch_P2, pitch_P3, pitch_P4]),
                pitch_w1, u,
            )
            R2_s, Y2_s = eval_rational_bezier(
                np.column_stack([pitch_P4, pitch_P5, pitch_P6, pitch_P7]),
                pitch_w2, u,
            )

            pitch_R_curve = np.concatenate([R1_s, R2_s[1:]])
            pitch_curve = np.concatenate([Y1_s, Y2_s[1:]])

            pitch_interp = interp1d(
                pitch_R_curve,
                pitch_curve,
                kind="cubic",
                bounds_error=False,
                fill_value=(pitch_curve[0], pitch_curve[-1]),
            )
            Pitch = lambda x, _f=pitch_interp: _f(x)

    # If rejected, keep original functions unchanged
    if reject_info["rejected"]:
        Pitch = Pitch_origin
        ChordLength = ChordLength_origin

    # Optional logging
    if reject_log is not None and (reject_info["rejected"] or reject_info["violated"]):
        reject_log.append(reject_info.copy())

    if return_reject_info:
        return Pitch, ChordLength, chord_con_points, pitch_con_points, reject_info

    return Pitch, ChordLength, chord_con_points, pitch_con_points


def eval_rational_bezier(CP, w, u):
    """
    Helper function for one 4-point rational cubic Bézier
    """
    B0 = (1 - u) ** 3
    B1 = 3 * (1 - u) ** 2 * u
    B2 = 3 * (1 - u) * u ** 2
    B3 = u ** 3

    num = (
        CP[:, 0:1] * (w[0] * B0)
        + CP[:, 1:2] * (w[1] * B1)
        + CP[:, 2:3] * (w[2] * B2)
        + CP[:, 3:4] * (w[3] * B3)
    )

    den = w[0] * B0 + w[1] * B1 + w[2] * B2 + w[3] * B3

    X = num[0, :] / den
    Y = num[1, :] / den

    return X, Y