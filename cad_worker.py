"""
Isolated single-case CAD generation worker.

Run as a subprocess so a hang or crash inside pythonOCC (which cannot be
interrupted from Python because it is C code) only affects one case and can be
killed via a subprocess timeout instead of blocking the whole batch.

Usage:
    python cad_worker.py <case_id> <geometry_dir> <design_npy>

design_npy is a 1-D array of length 11: 6 pitch CPs + 5 chord CPs
(DRDC path via X_CAD_from_design).
"""

import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 4:
        print("usage: python cad_worker.py <case_id> <geometry_dir> <design_npy>")
        return 2

    case_id = int(sys.argv[1])
    geometry_dir = Path(sys.argv[2])
    design_path = Path(sys.argv[3])

    design = np.asarray(np.load(design_path), dtype=float).ravel()
    if design.size != 11:
        print(f"expected design length 11 (6 pitch + 5 chord), got {design.size}")
        return 2

    pitch_con = design[:6]
    chord_con = design[6:]

    import para
    para.GEOMETRY_OUTPUT_DIR = geometry_dir
    from X_CAD_new import X_CAD_from_design

    X_CAD_from_design(pitch_con, chord_con, case_id, output_dir=geometry_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
