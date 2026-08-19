"""
X_CAD_new.py

DRDC-method CAD export, replacing the loft-and-cap approach of the original
X_CAD.py (which this file supersedes; the original X_CAD.py is untouched).

Two entry points:
  X_CAD(grids, x1)                       -> grids to IGES (main function)
  X_CAD_from_design(pitch_con, chord_con, x1)
                                         -> design vector to IGES (wrapper:
                                            X_blade -> build_drdc_grids ->
                                            X_CAD); drop-in for cad_worker.

Input to X_CAD is the five structured surface grids produced by
tip_surfaces_new.build_drdc_grids(). Each grid is interpolated by a
tensor-product cubic B-spline (scipy, s=0 so the grid points are ON the
surface), converted to an OCC Geom_BSplineSurface, made into a face, sewed,
and written to IGES. Because adjacent grids share identical boundary
point rows/columns, the faces match along their common edges and the sewing
merges them; there is no filling solver, no loft, no cap, and no degenerate
tip ring anywhere.

Surface naming follows TM 2013-180 Annex B:
  Blade 1 = trailing edge surface   ('te_strip')
  Blade 2 = central pressure side   ('central_pressure')
  Blade 3 = leading edge surface    ('le_strip')
  Blade 4 = central suction side    ('central_suction')
  Blade 5 = tip surface             ('tip')
Orientation flips (to satisfy the outward-normal and parameter-direction
conventions of Annex B) are collected in ORIENTATIONS below so they can be
adjusted in one place once checked in the target CAD/mesh tool.
"""

import os, sys
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(os.path.join(sys.prefix, "Library", "bin"))
    except (FileNotFoundError, OSError):
        pass

import numpy as np
from scipy.interpolate import RectBivariateSpline

from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Extend.DataExchange import write_iges_file

from pipeline_config import cad_output_paths

MM_PER_M = 1000.0
SEW_TOL = 1.0e-3          # mm; grids share boundary points exactly

# (flip_rows, flip_cols, transpose) applied to each grid before splining.
# Adjust here if the meshing tool needs the Annex B parameter directions.
ORIENTATIONS = {
    "te_strip": (False, False, False),
    "central_pressure": (False, False, False),
    "le_strip": (False, False, False),
    "central_suction": (False, False, False),
    "tip": (False, False, False),
}

ANNEX_B_ORDER = [
    ("Blade 1", "te_strip"),
    ("Blade 2", "central_pressure"),
    ("Blade 3", "le_strip"),
    ("Blade 4", "central_suction"),
    ("Blade 5", "tip"),
]


def _knots_to_occ_arrays(t):
    """Full clamped knot vector -> (TColStd knots, TColStd mults)."""
    uniq, mult = [], []
    for kn in t:
        if uniq and abs(kn - uniq[-1]) < 1e-12:
            mult[-1] += 1
        else:
            uniq.append(float(kn))
            mult.append(1)
    kn_arr = TColStd_Array1OfReal(1, len(uniq))
    ml_arr = TColStd_Array1OfInteger(1, len(mult))
    for i, (kn, ml) in enumerate(zip(uniq, mult), start=1):
        kn_arr.SetValue(i, kn)
        ml_arr.SetValue(i, ml)
    return kn_arr, ml_arr


def _occ_surface_from_tck(tu, tv, coeffs, ku, kv, transpose=False):
    """Build a Geom_BSplineSurface from scipy tck data.

    coeffs : (n_cu, n_cv, 3). transpose=True swaps the role of the two
    parameter directions (used by the runtime self-check below).
    """
    if transpose:
        coeffs = np.transpose(coeffs, (1, 0, 2))
        tu, tv = tv, tu
        ku, kv = kv, ku
    n_cu, n_cv, _ = coeffs.shape
    poles = TColgp_Array2OfPnt(1, n_cu, 1, n_cv)
    for i in range(n_cu):
        for j in range(n_cv):
            poles.SetValue(i + 1, j + 1,
                           gp_Pnt(float(coeffs[i, j, 0]),
                                  float(coeffs[i, j, 1]),
                                  float(coeffs[i, j, 2])))
    uk, um = _knots_to_occ_arrays(tu)
    vk, vm = _knots_to_occ_arrays(tv)
    return Geom_BSplineSurface(poles, uk, vk, um, vm, ku, kv, False, False)


def grid_to_bspline_surface(grid, u_params=None, v_params=None,
                            check_tol_mm=1.0e-3):
    """Interpolating bicubic B-spline surface through a structured grid,
    SELF-VERIFIED at runtime.

    The spline is fitted with scipy using OUR parameter values (u_params /
    v_params; this matters because the wrap grids are clustered 100:1 near
    the edges and any interpolator that picks its own parameterization
    overshoots into large lobes there). The scipy spline is converted to an
    OCC Geom_BSplineSurface, then the OCC surface is EVALUATED at every grid
    parameter and compared against the grid points. If the deviation exceeds
    check_tol_mm the conversion is retried transposed; if it still fails, a
    RuntimeError is raised rather than exporting garbage geometry.
    """
    grid = np.asarray(grid, dtype=float)
    nu, nv, _ = grid.shape
    u = np.linspace(0.0, 1.0, nu) if u_params is None else np.asarray(u_params, dtype=float)
    v = np.linspace(0.0, 1.0, nv) if v_params is None else np.asarray(v_params, dtype=float)

    ku = min(3, nu - 1)
    kv = min(3, nv - 1)
    splines = [RectBivariateSpline(u, v, grid[:, :, c] * MM_PER_M,
                                   kx=ku, ky=kv, s=0) for c in range(3)]
    tu, tv = splines[0].tck[0], splines[0].tck[1]
    n_cu = len(tu) - ku - 1
    n_cv = len(tv) - kv - 1
    coeffs = np.stack([s.get_coeffs().reshape(n_cu, n_cv) for s in splines],
                      axis=-1)

    # sanity: the scipy spline itself must interpolate the grid
    chk = np.stack([splines[c](u, v) for c in range(3)], axis=-1)
    scipy_err = np.abs(chk - grid * MM_PER_M).max()
    if scipy_err > check_tol_mm:
        raise RuntimeError(
            f"scipy surface fit failed to interpolate (err {scipy_err:.3g} mm)")

    for transpose in (False, True):
        srf = _occ_surface_from_tck(tu, tv, coeffs, ku, kv, transpose)
        worst = 0.0
        for i in range(nu):
            for j in range(nv):
                uu, vv = (u[i], v[j]) if not transpose else (v[j], u[i])
                p = srf.Value(float(uu), float(vv))
                d = max(abs(p.X() - grid[i, j, 0] * MM_PER_M),
                        abs(p.Y() - grid[i, j, 1] * MM_PER_M),
                        abs(p.Z() - grid[i, j, 2] * MM_PER_M))
                worst = max(worst, d)
            if worst > check_tol_mm:
                break
        if worst <= check_tol_mm:
            if transpose:
                print("grid_to_bspline_surface: NOTE transposed conversion "
                      "was required.")
            return srf
    raise RuntimeError(
        f"OCC B-spline conversion failed self-check (worst dev {worst:.3g} "
        "mm); refusing to export bad geometry.")


def _apply_orientation(grid, name):
    fr, fc, tr = ORIENTATIONS.get(name, (False, False, False))
    g = grid
    if fr:
        g = g[::-1, :, :]
    if fc:
        g = g[:, ::-1, :]
    if tr:
        g = np.transpose(g, (1, 0, 2))
    return np.ascontiguousarray(g)


# Root skirt depth (m). The blade root ring lies ON the hub surface to
# machine precision at the shared sample points, but BETWEEN samples both
# surfaces are cubic B-splines that deviate from the exact cylinder by up to
# ~0.3 um, each on its own side. The contact is therefore tangency within a
# band of about +-1 um: the worst case for a mesher's intersection logic, and
# the reason blade and hub failed to assemble in Pointwise. The robust
# practice is to make the surfaces clearly INTERSECT: the blade root section
# is copied radially inward and the band between the copy and the root ring
# is exported as extra "skirt" faces. The mesher trims the blade at the hub
# and discards the skirt, which lies entirely inside the hub.
#
# The skirt is deliberately SEPARATE faces rather than extra rows prepended
# to the blade grids: prepending rows bends the global interpolating spline
# and was measured to deform the wetted blade by 1.4-2.5 mm in the first
# radial band. As separate ruled faces the wetted blade is byte-identical to
# the validated geometry, and the tangent kink between blade and skirt lies
# exactly on the hub surface, where the trim removes it.
ROOT_SKIRT_DEPTH = 0.003

# skirt faces sewn to their parent patches (parent key -> face label)
SKIRT_PARENTS = [
    ("te_strip", "Blade 1 skirt"),
    ("central_pressure", "Blade 2 skirt"),
    ("le_strip", "Blade 3 skirt"),
    ("central_suction", "Blade 4 skirt"),
]


def build_root_skirts(grids, depth=ROOT_SKIRT_DEPTH, n_rows=4):
    """Ruled skirt grids below the root ring: exact radial copies inward.

    For each root-touching patch, every point p of its root row is scaled in
    (y, z) toward the axis so the copies sit at radius r0 - depth ... r0.
    The LAST skirt row is the parent's root row itself, so the shared edge
    coincides point-for-point (and, fitted with the same column parameters,
    curve-for-curve), which lets the sewing merge skirt and blade exactly.
    Rows are colinear per column, so the fitted surface is exactly ruled.
    """
    out = {}
    for key, _ in SKIRT_PARENTS:
        g = np.asarray(grids[key], dtype=float)
        p0 = g[0]
        r0 = np.hypot(p0[:, 1], p0[:, 2])
        rows = []
        for k in range(n_rows):
            scale = (r0 - depth * (1.0 - k / (n_rows - 1.0))) / r0
            q = p0.copy()
            q[:, 1] *= scale
            q[:, 2] *= scale
            rows.append(q)
        rows[-1] = p0                      # shared edge, exact
        out[key] = np.stack(rows, axis=0)
    return out


def X_CAD(grids, x1, output_dir=None, hub=True, hub_height=None,
          hub_center=0.0, n_blades=5, root_skirt=ROOT_SKIRT_DEPTH):
    """Build the five-surface DRDC blade (plus hub) and write it to IGES.

    grids      : dict from tip_surfaces_new.build_drdc_grids()
    x1         : case id (used for the output filename via pipeline_config)
    hub        : include the Sec. 10 hub sector (default True)
    hub_height : total hub height in metres; None = 2x the root ring's
                 axial span, centred on the ring. The radius is measured
                 from the blade root ring and takes no input.
    n_blades   : Z; the sector spans exactly 2 pi / Z and Z rotated copies
                 reproduce the full hub (TM 2013-178 Sec. 10, Fig. 9).
    root_skirt : depth (m) of the radial root-skirt faces sunk below the
                 hub surface so blade and hub clearly intersect for the
                 mesher's trim. 0 or None disables. The wetted blade faces
                 are identical with or without the skirt.

    The hub goes through the SAME grid -> self-verified-B-spline path as
    the blade patches, so it arrives in the IGES as bounded entity-128
    surfaces. Deliberately NOT an analytic Geom_CylindricalSurface: the
    Faces-mode IGES writer discards trimming, and an analytic cylinder then
    arrives unbounded (the earlier "Model Size adjusted to 100000" and
    stack-of-rings failures).
    """
    t_common = grids.get("meta", {}).get("t_common")

    if hub:
        from hub_new import hub_grids
        hgrids, hinfo = hub_grids(grids, hub_height=hub_height,
                                  hub_center=hub_center, n_blades=n_blades)

    sewing = BRepBuilderAPI_Sewing(SEW_TOL)
    faces = []
    for label, key in ANNEX_B_ORDER:
        grid = _apply_orientation(np.asarray(grids[key], dtype=float), key)
        # the wrap grids are sampled at the clustered t_common columns; the
        # spline MUST use those same parameter values
        v_params = None
        if key in ("te_strip", "le_strip", "tip") and t_common is not None:
            v_params = t_common
        srf = grid_to_bspline_surface(grid, v_params=v_params)
        face = BRepBuilderAPI_MakeFace(srf, 1.0e-6).Face()
        faces.append((label, face))
        sewing.Add(face)

    if root_skirt and root_skirt > 0:
        skirts = build_root_skirts(grids, depth=root_skirt)
        for key, label in SKIRT_PARENTS:
            # same column parameters as the parent patch, so the shared root
            # edge is the SAME 1D spline in both fits: they coincide exactly,
            # between nodes included
            v_params = None
            if key in ("te_strip", "le_strip") and t_common is not None:
                v_params = t_common
            srf = grid_to_bspline_surface(skirts[key], v_params=v_params)
            face = BRepBuilderAPI_MakeFace(srf, 1.0e-6).Face()
            faces.append((label, face))
            sewing.Add(face)
        print(f"[X_CAD] root skirt: 4 ruled faces sunk {root_skirt*1000:.1f} "
              f"mm below the hub surface (mesher trims at the hub); wetted "
              f"blade unchanged")

    if hub:
        for label, key in (("Hub", "hub_sector"),
                           ("Hub cap lo", "hub_cap_lo"),
                           ("Hub cap hi", "hub_cap_hi")):
            srf = grid_to_bspline_surface(hgrids[key])
            face = BRepBuilderAPI_MakeFace(srf, 1.0e-6).Face()
            faces.append((label, face))
            sewing.Add(face)

    sewing.Perform()
    sewed = sewing.SewedShape()
    fixer = ShapeFix_Shape(sewed)
    fixer.Perform()
    shape = fixer.Shape()

    paths = cad_output_paths(x1, output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    iges_path = paths["iges"]
    write_iges_file(shape, str(iges_path))
    n_srf = len(faces)
    print(f"IGES file written ({n_srf} faces: 5 blade"
          + (" + 4 root skirt" if root_skirt and root_skirt > 0 else "")
          + (" + hub sector + 2 caps" if hub else "") + f"): {iges_path}")
    return shape


def X_CAD_from_design(pitch_con, chord_con, x1, output_dir=None,
                      tip_config=None, write_dat=False, verbose=True,
                      hub=True, hub_height=None, hub_center=0.0,
                      n_blades=5, root_skirt=ROOT_SKIRT_DEPTH):
    """Convenience wrapper: design vector -> DRDC grids -> IGES.

    Drop-in for workers that previously called
    X_CAD(points, case_id): call X_CAD_from_design(pitch_con, chord_con,
    case_id) instead.
    """
    from x_blade_new import X_blade
    from tip_surfaces_new import build_drdc_grids, TipConfig

    res = X_blade(pitch_con, chord_con, x1, return_blade_surface=True,
                  write_dat=write_dat)
    blade_surface = res[-1]
    constraint_violation = res[2]
    if blade_surface is None or constraint_violation:
        raise RuntimeError(
            f"Case {x1}: design rejected (violation={constraint_violation}); "
            "no CAD generated.")
    grids = build_drdc_grids(blade_surface, tip_config or TipConfig(),
                             verbose=verbose)
    return X_CAD(grids, x1, output_dir=output_dir, hub=hub,
                 hub_height=hub_height, hub_center=hub_center,
                 n_blades=n_blades, root_skirt=root_skirt)
