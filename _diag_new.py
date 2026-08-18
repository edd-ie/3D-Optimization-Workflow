"""
_diag_new.py  --  one run, one report. Tells us where the defect is.

    conda activate pyocc310
    python _diag_new.py > diag.txt

Stage A builds the five grids and checks them on their own terms.
Stage B pushes them through X_CAD_new to IGES.
Stage C reads that IGES BACK with OCC and compares it against the grids.

If A is clean and C is not, the bug is in the OCC/IGES conversion.
If A is dirty, the bug is upstream in the grids and the CAD layer is innocent.
Paste the whole output; every number below is something I can act on.
"""

import sys
import numpy as np

np.set_printoptions(precision=6, suppress=True)

PITCH_CON = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
CHORD_CON = np.array([0.25, 0.65, 0.325, 0.55, 0.2])
CASE_ID = 99999
NAMES = ["te_strip", "central_pressure", "le_strip", "central_suction", "tip"]


def rule(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# --------------------------------------------------------------- STAGE A
rule("STAGE A -- grids")

from x_blade_new import X_blade
from tip_surfaces_new import build_drdc_grids, TipConfig

blade = X_blade(PITCH_CON, CHORD_CON, CASE_ID,
                return_blade_surface=True, write_dat=False)[-1]

cfg = TipConfig()
_mode = getattr(cfg, "split_mode", None)
if _mode is None:
    _mode = "<<< MISSING: this is the OLD tip_surfaces_new.py >>>"
_width = getattr(cfg, "strip_width", "<<< MISSING >>>")
print("split_mode  = %s" % _mode)
print("strip_width = %s" % _width)
print("tip_extent = %s, trans_le = %s, trans_te = %s, delta_c = %s"
      % (cfg.tip_extent, cfg.trans_le, cfg.trans_te, cfg.delta_c))

grids = build_drdc_grids(blade, cfg)
meta = grids["meta"]

print("\n-- split points and Eq. 48 --")
for k in ("s_tip", "s_te", "s_le", "s_trans_te", "s_trans_le",
          "len_te", "len_le", "margin_te", "margin_le", "d_t",
          "N", "N_t", "j_te", "j_le"):
    if k in meta:
        v = meta[k]
        print(f"  {k:12s} {v:.6f}" if isinstance(v, float) else
              f"  {k:12s} {v}")
print(f"  reversals    {meta.get('reversals')}")

print("\n-- per patch --")
for n in NAMES:
    g = np.asarray(grids[n])
    lo, hi = g.reshape(-1, 3).min(0), g.reshape(-1, 3).max(0)
    drow = np.linalg.norm(np.diff(g, axis=0), axis=-1).sum(axis=0)
    dcol = np.linalg.norm(np.diff(g, axis=1), axis=-1).sum(axis=1)
    # cell sizes, to catch a pinched or collapsed region
    du = np.linalg.norm(np.diff(g, axis=0), axis=-1)
    dv = np.linalg.norm(np.diff(g, axis=1), axis=-1)
    print(f"  {n:18s} {str(g.shape):16s} finite={np.isfinite(g).all()}")
    print(f"      bbox  x[{lo[0]:8.4f},{hi[0]:8.4f}] "
          f"y[{lo[1]:8.4f},{hi[1]:8.4f}] z[{lo[2]:8.4f},{hi[2]:8.4f}]")
    print(f"      row span min {drow.min():.6f}  col span min {dcol.min():.6f}")
    print(f"      cell du [{du.min():.2e}, {du.max():.2e}]  "
          f"dv [{dv.min():.2e}, {dv.max():.2e}]")
    if drow.min() < 1e-9 or dcol.min() < 1e-9:
        print("      *** DEGENERATE row or column ***")

print("\n-- shared edges (must be ~0) --")


def edge_gap(a, b, label):
    d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    flag = "" if d < 1e-9 else "   *** MISMATCH ***"
    print(f"  {label:44s} {d:.3e}{flag}")
    return d


# These are the pairings used by _tip_test_new.py, which are the correct
# adjacencies. (An earlier version of this file guessed them and reported
# false mismatches of up to 0.5 m on a grid set that is in fact exact.)
te, cp, le, cs, tp = (np.asarray(grids[n]) for n in NAMES)
worst = 0.0
for lab, a, b in (
        ("central_pressure col 0   / te_strip col 0", cp[:, 0], te[:, 0]),
        ("central_pressure col -1  / le_strip col 0", cp[:, -1], le[:, 0]),
        ("central_suction  col 0   / te_strip col -1", cs[:, 0], te[:, -1]),
        ("central_suction  col -1  / le_strip col -1", cs[:, -1], le[:, -1]),
        ("central_pressure row -1  / tip col 0", cp[-1, :], tp[:, 0]),
        ("central_suction  row -1  / tip col -1", cs[-1, :], tp[:, -1]),
        ("te_strip row -1          / tip row 0", te[-1, :], tp[0, :]),
        ("le_strip row -1          / tip row -1", le[-1, :], tp[-1, :])):
    worst = max(worst, edge_gap(a, b, lab))
print(f"  worst shared-edge gap {worst:.3e} m")

# ---- folding screen ---------------------------------------------------
# NOT against a global mean normal: te_strip, le_strip and tip each wrap
# around an edge from the pressure side to the suction side, so their normal
# genuinely swings through ~180 deg and any global reference reports a fold
# that is not there. Compare each cell to its NEIGHBOUR instead, which is a
# real continuity test.
print("\n-- folding screen (neighbour-to-neighbour normal continuity) --")
for n in NAMES:
    g = np.asarray(grids[n])
    du = np.diff(g, axis=0)[:, :-1, :]
    dv = np.diff(g, axis=1)[:-1, :, :]
    nrm = np.cross(du, dv)
    ln = np.linalg.norm(nrm, axis=-1)
    u = nrm / np.maximum(ln, 1e-300)[..., None]
    # A wrap patch turns through ~180 deg AT the apex (the rounded LE/TE nose,
    # or the tip fold where b(xi,1) = b(1-xi,1)). That is real geometry. Only
    # a large turn AWAY from the apex column is a defect, so report where the
    # worst turn is and judge it by that.
    ncol = u.shape[1]
    apex = ncol // 2
    worst_turn, where, bad_away = 0.0, None, 0
    for ax in (0, 1):
        a = u.take(np.arange(u.shape[ax] - 1), axis=ax)
        b = u.take(np.arange(1, u.shape[ax]), axis=ax)
        ang = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))
        if ang.max() > worst_turn:
            worst_turn = float(ang.max())
            where = np.unravel_index(int(np.argmax(ang)), ang.shape)
        big = ang > 90.0
        cols = np.broadcast_to(np.arange(ang.shape[1]), ang.shape)
        near = np.abs(cols - apex) <= 0.08 * ncol
        if n == "tip":
            # the tip patch has a different topology from the strips: its
            # rows run TE -> LE, so its sharp edges are the FIRST and LAST
            # rows (the blunt TE and the LE nose), not a middle column.
            rows = np.broadcast_to(np.arange(ang.shape[0])[:, None],
                                   ang.shape)
            nr = ang.shape[0]
            near = near | (rows <= 0.08 * nr) | (rows >= nr - 1 - 0.08 * nr)
        bad_away += int((big & ~near).sum())
    tag = ("   *** FOLD away from the apex ***" if bad_away else
           "   ok (turn is at the apex)")
    print(f"  {n:18s} max turn {worst_turn:6.1f} deg at row/col {where}, "
          f"apex col ~{apex}; >90 deg away from apex: {bad_away}{tag}")

# --------------------------------------------------------------- STAGE B
rule("STAGE B -- X_CAD_new -> IGES")
try:
    from X_CAD_new import X_CAD
    from hub_new import hub_grids
    hg, hinfo = hub_grids(grids, verbose=True)
    all_grids = dict(grids)
    all_grids.update(hg)
    ALL_NAMES = NAMES + list(hg.keys())
    X_CAD(grids, CASE_ID)
    from pipeline_config import cad_output_paths
    iges_path = cad_output_paths(CASE_ID)["iges"]
    print(f"wrote: {iges_path}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("\nSTAGE B FAILED -- stopping here.")
    sys.exit(1)

# --------------------------------------------------------------- STAGE C
rule("STAGE C -- read the IGES back with OCC and compare")
try:
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add

    r = IGESControl_Reader()
    st = r.ReadFile(str(iges_path))
    print(f"ReadFile status {st}")
    r.TransferRoots()
    shape = r.OneShape()

    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        faces.append(topods.Face(ex.Current()))
        ex.Next()
    print(f"faces found in the file: {len(faces)}  (expected {len(ALL_NAMES)}: 5 blade + {len(hg)} hub)")

    for i, f in enumerate(faces):
        p = GProp_GProps()
        brepgprop_SurfaceProperties(f, p)
        bb = Bnd_Box()
        brepbndlib_Add(f, bb)
        xm, ym, zm, xM, yM, zM = bb.Get()
        srf = BRep_Tool.Surface(f)
        u0, u1, v0, v1 = srf.Bounds()
        print(f"  face {i}: area {p.Mass():10.6f}  "
              f"uv [{u0:.3f},{u1:.3f}]x[{v0:.3f},{v1:.3f}]")
        print(f"          bbox x[{xm:8.4f},{xM:8.4f}] "
              f"y[{ym:8.4f},{yM:8.4f}] z[{zm:8.4f},{zM:8.4f}]")

    # compare each face against the grid it should be
    print("\n-- max distance from grid points to the surface read back --")
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    for i, f in enumerate(faces):
        srf = BRep_Tool.Surface(f)
        best = None
        for n in ALL_NAMES:
            g = np.asarray(all_grids[n]).reshape(-1, 3)
            idx = np.linspace(0, len(g) - 1, 60).astype(int)
            worst_d = 0.0
            for k in idx:
                pr = GeomAPI_ProjectPointOnSurf(
                    gp_Pnt(*[float(c) for c in g[k]]), srf)
                if pr.NbPoints() > 0:
                    worst_d = max(worst_d, pr.LowerDistance())
                else:
                    worst_d = float("inf")
                    break
            if best is None or worst_d < best[1]:
                best = (n, worst_d)
        flag = "" if best[1] < 1e-6 else "   *** does not match ***"
        print(f"  face {i} best match {best[0]:18s} {best[1]:.3e} m{flag}")

except Exception:
    import traceback
    traceback.print_exc()

rule("END")
