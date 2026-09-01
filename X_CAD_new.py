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
# for the root trim experiment
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Ax2, gp_Dir
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.Interface import Interface_Static

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


def _count_faces(shape):
    n, ex = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n += 1
        ex.Next()
    return n


def _trim_blade_at_hub(blade_shape, hub_radius_m, ring_x_mm):
    """Cut away the blade portion INSIDE the hub cylinder.

    The tool is an analytic cylinder SOLID at the design hub radius along +x
    (tool only; it is never exported). BRepAlgoAPI_Cut(blade, cyl) keeps the
    part of the blade outside the cylinder, so the root-extended band below
    the hub surface is removed and the new root edge is OCC's intersection
    curve of blade and hub cylinder. Units: mm (the faces are built in mm).
    """
    R = hub_radius_m * MM_PER_M
    x_lo = float(ring_x_mm[0]) - 200.0
    x_hi = float(ring_x_mm[1]) + 200.0
    ax = gp_Ax2(gp_Pnt(x_lo, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    tool = BRepPrimAPI_MakeCylinder(ax, R, x_hi - x_lo).Solid()
    cut = BRepAlgoAPI_Cut(blade_shape, tool)
    cut.Build()
    if not cut.IsDone():
        raise RuntimeError("root trim failed: BRepAlgoAPI_Cut did not build")
    out = cut.Shape()
    n = _count_faces(out)
    if n < 5:
        raise RuntimeError(f"root trim produced only {n} faces; refusing")
    return out


def _write_iges_brep(shape, path):
    """IGES writer in BRep mode (1).

    REQUIRED whenever faces carry trim loops: the default Faces-mode writer
    exports only each face's underlying surface and DISCARDS trimming (the
    documented unbounded-cylinder failure earlier in this project). BRep
    mode writes trimmed faces as MSBO entities, which carry the loops.
    """
    Interface_Static.SetCVal("write.iges.unit", "MM")
    w = IGESControl_Writer("MM", 1)
    w.AddShape(shape)
    w.ComputeModel()
    if not w.Write(str(path)):
        raise RuntimeError(f"IGES write failed: {path}")


def X_CAD(grids, x1, output_dir=None, hub=True, hub_height=None,
          hub_center=0.0, n_blades=5, trim_root_at_hub=True):
    """Build the five-surface DRDC blade (plus hub) and write it to IGES.

    grids      : dict from tip_surfaces_new.build_drdc_grids()
    x1         : case id (used for the output filename via pipeline_config)
    hub        : include the Sec. 10 hub sector (default True)
    hub_height : total hub height in metres; None = 2x the root ring's
                 axial span, centred on the ring. The radius is measured
                 from the blade root ring and takes no input.
    n_blades   : Z; the sector spans exactly 2 pi / Z and Z rotated copies
                 reproduce the full hub (TM 2013-178 Sec. 10, Fig. 9).

    The five blade patches are ROOT-EXTENDED: the blade surface starts
    ROOT_EXTENSION (blade_surface_new, default 3 mm) below the design root
    radius, while the hub keeps the design radius (meta['hub_radius']), so
    blade and hub clearly intersect. Nothing is trimmed in the file; the
    blade-at-hub trim is done manually in Pointwise.

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

    if trim_root_at_hub:
        hub_radius = grids.get("meta", {}).get("hub_radius")
        if hub_radius is None:
            raise RuntimeError("trim_root_at_hub needs meta['hub_radius'] "
                               "(regenerate grids with the current "
                               "tip_surfaces_new)")
        # blade faces only: separate the blade shell from the already-added
        # hub faces by rebuilding the blade shell alone
        blade_sew = BRepBuilderAPI_Sewing(SEW_TOL)
        for label, f in faces:
            if label.startswith("Blade"):
                blade_sew.Add(f)
        blade_sew.Perform()
        blade_fix = ShapeFix_Shape(blade_sew.SewedShape())
        blade_fix.Perform()
        blade_shape = blade_fix.Shape()

        ring = np.vstack([np.asarray(grids[k], dtype=float)[0, :, :]
                          for k in ("te_strip", "central_pressure",
                                    "le_strip", "central_suction")])
        ring_x_mm = (ring[:, 0].min() * MM_PER_M,
                     ring[:, 0].max() * MM_PER_M)
        blade_trimmed = _trim_blade_at_hub(blade_shape, hub_radius, ring_x_mm)

        comp = TopoDS_Compound()
        bb = BRep_Builder()
        bb.MakeCompound(comp)
        bb.Add(comp, blade_trimmed)
        if hub:
            hub_sew = BRepBuilderAPI_Sewing(SEW_TOL)
            for label, f in faces:
                if label.startswith("Hub"):
                    hub_sew.Add(f)
            hub_sew.Perform()
            bb.Add(comp, hub_sew.SewedShape())
        shape = comp

        _write_iges_brep(shape, iges_path)
        print(f"[X_CAD] blade root TRIMMED at the hub cylinder "
              f"(R = {hub_radius:.6f} m); trimmed blade faces: "
              f"{_count_faces(blade_trimmed)}. Written in IGES BRep mode "
              f"(trim loops preserved).")
        print(f"IGES file written (trimmed blade"
              + (" + hub sector + 2 caps" if hub else "") + f"): {iges_path}")
        return shape

    write_iges_file(shape, str(iges_path))
    n_srf = len(faces)
    print(f"IGES file written ({n_srf} faces: 5 blade (root-extended)"
          + (" + hub sector + 2 caps" if hub else "") + f"): {iges_path}")
    return shape


def X_CAD_from_design(pitch_con, chord_con, x1, output_dir=None,
                      tip_config=None, write_dat=False, verbose=True,
                      hub=True, hub_height=None, hub_center=0.0,
                      n_blades=5, trim_root_at_hub=True):
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
                 n_blades=n_blades, trim_root_at_hub=trim_root_at_hub)
