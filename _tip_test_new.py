"""
_tip_test_new.py

Test driver for the DRDC five-surface tip pipeline (the *_new modules).

Stage 1 (no pythonOCC needed): builds the BladeSurface for a benign midpoint
design, verifies the geometric invariants, builds the five surface grids and
checks that all shared edges are identical point sets.

Stage 2 (requires pythonOCC, i.e. your normal CAD environment): converts the
grids to B-spline faces, sews, writes IGES and reports topology counts.

Run:  python _tip_test_new.py [--coarse] [--no-cad]
"""

import sys
import time
import numpy as np

COARSE = "--coarse" in sys.argv
NO_CAD = "--no-cad" in sys.argv

from x_blade_new import X_blade
from tip_surfaces_new import build_drdc_grids, TipConfig

# Midpoints of the Initial_sampling bounds (a benign, non-rejected design).
pitch_con = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
chord_con = np.array([0.25, 0.65, 0.325, 0.55, 0.2])
CASE_ID = 99999


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    all_ok = True

    print("== Stage 1a: blade surface ==")
    res = X_blade(pitch_con, chord_con, CASE_ID,
                  return_blade_surface=True, write_dat=False)
    points, min_dis, cv = res[0], res[1], res[2]
    blade = res[-1]
    print(f"  aero points {points.shape}, min_dis={min_dis:.4f}, cv={cv}")
    all_ok &= check("design accepted", cv == 0 and points.size > 0)

    eta = np.linspace(blade.eta_min, 1.0, 40)
    te_gap = np.linalg.norm(blade.b(0.0, eta) - blade.b(1.0, eta),
                            axis=1).max()
    all_ok &= check("TE closed (periodic)", te_gap < 1e-12,
                    f"max gap {te_gap:.2e} m")

    xi = np.linspace(0.0, 0.5, 40)
    fold_gap = np.linalg.norm(blade.b(xi, 1.0) - blade.b(1.0 - xi, 1.0),
                              axis=1).max()
    all_ok &= check("tip closed (fold identification)", fold_gap < 1e-12,
                    f"max gap {fold_gap:.2e} m")

    n = blade.normal(np.array([0.1, 0.3, 0.6, 0.9]),
                     np.array([0.3, 0.6, 0.9, 0.99]))
    all_ok &= check("normals finite", bool(np.all(np.isfinite(n))))

    # legacy agreement at a mid-blade station (away from TE cap / tip zone)
    r_chk = 0.656
    i_sec = int(round((r_chk - 0.17) / 0.018))
    sec = points[i_sec * 53:(i_sec + 1) * 53]
    from scipy.spatial import cKDTree
    xi_d = np.linspace(0, 1, 4000)
    surf = blade.b(xi_d, np.full_like(xi_d, blade.eta_of_r(r_chk)))
    dmax = cKDTree(surf).query(sec)[0].max()
    all_ok &= check("matches legacy section (r=0.656)", dmax < 5e-4,
                    f"max {dmax*1000:.3f} mm")

    print("== Stage 1b: five-surface grids ==")
    cfg = (TipConfig(delta_c=0.05, n_wrap=61, n_iter=20,
                     n_outline=1500, max_curves=16)
           if COARSE else TipConfig())
    t0 = time.time()
    grids = build_drdc_grids(blade, cfg)
    print(f"  built in {time.time()-t0:.1f} s")
    te, le, tip = grids["te_strip"], grids["le_strip"], grids["tip"]
    cp, cs = grids["central_pressure"], grids["central_suction"]

    for name, g in (("te_strip", te), ("le_strip", le), ("tip", tip),
                    ("central_pressure", cp), ("central_suction", cs)):
        all_ok &= check(f"{name} finite {g.shape}",
                        bool(np.all(np.isfinite(g))))

    def emax(a, b):
        return np.abs(a - b).max()

    pairs = [
        ("cp|te", cp[:, 0], te[:, 0]), ("cp|le", cp[:, -1], le[:, 0]),
        ("cs|te", cs[:, 0], te[:, -1]), ("cs|le", cs[:, -1], le[:, -1]),
        ("cp|tip", cp[-1, :], tip[:, 0]), ("cs|tip", cs[-1, :], tip[:, -1]),
        ("te|tip", te[-1, :], tip[0, :]), ("le|tip", le[-1, :], tip[-1, :]),
    ]
    worst = max(emax(a, b) for _, a, b in pairs)
    all_ok &= check("all shared edges identical", worst == 0.0,
                    f"worst {worst:.2e} m")

    # no folds/crossings in any grid
    rev = grids["meta"]["reversals"]
    all_ok &= check("no row-direction reversals",
                    all(v == 0 for v in rev.values()), str(rev))

    # no wildly oversized cut rows (the failure that produced spurious lobes)
    for name, g in (("te_strip", te), ("le_strip", le), ("tip", tip)):
        L = np.sum(np.linalg.norm(np.diff(g, axis=1), axis=-1), axis=1)
        jump = np.max(np.abs(np.diff(L)) / np.maximum(L[:-1], 1e-12))
        all_ok &= check(f"{name} row lengths smooth", jump < 0.35,
                        f"max row-to-row change {jump*100:.1f}%")

    # root ring lies at the root radius
    root_pts = np.vstack([te[0], le[0], cp[0], cs[0]])
    rr = np.sqrt(root_pts[:, 1] ** 2 + root_pts[:, 2] ** 2)
    all_ok &= check("root ring at r_root", np.allclose(rr, 0.17 * 0.7,
                                                       atol=1e-6),
                    f"radius range [{rr.min():.6f}, {rr.max():.6f}]")

    if NO_CAD:
        print("== Stage 2 skipped (--no-cad) ==")
    else:
        print("== Stage 2: OCC faces + IGES ==")
        try:
            from X_CAD_new import X_CAD
        except ImportError as exc:
            print(f"  pythonOCC not available here ({exc}); "
                  "run stage 2 in the CAD environment.")
        else:
            shape = X_CAD(grids, CASE_ID)
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            nf, ex = 0, TopExp_Explorer(shape, TopAbs_FACE)
            while ex.More():
                nf += 1
                ex.Next()
            all_ok &= check("five faces", nf == 5, f"got {nf}")
            all_ok &= check("BRepCheck valid",
                            BRepCheck_Analyzer(shape).IsValid())

    print("== RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", "==")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
