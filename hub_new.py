"""
hub_new.py

Per-blade hub SECTOR for the five-surface DRDC blade, following TM 2013-178
Sec. 10 (Eqs. 51-58, Figs. 9-10): the periodic 2*pi/Z wedge of the hub that
belongs to the reference blade, which rotated Z times reproduces the full
hub, exactly as the report's Fig. 9 shows "the portion saved in the IGES
file".

Hub parameter space (the report's (xi, theta), our axis being +x):
    xi    = x                     axial position
    theta = atan2(y, z) - theta_ref   circumferential, centred on the blade

Outer boundary, Sec. 10 steps 1-6:
  1.  Hub parameters of the blade root at the leading and trailing edges,
      (xi_le, th_le) and (xi_te, th_te), taken from the ROOT ROW of the
      le_strip / te_strip grids at the edge column (t = 1/2, which the cut
      construction pins to the LE/TE).
  2.  m = (th_te - th_le) / (xi_te - xi_le)                        (Eq. 53)
      th_lo = th_le - m/2 (xi_le - xi_lo)                          (Eq. 51)
      th_hi = th_te + m/2 (xi_hi - xi_te)                          (Eq. 52)
      (the report also clamps th to [-pi, pi] for its IGES surface of
      revolution; our representation has no branch cut at the sector, but a
      guard below still verifies the sector never wraps past +-pi.)
  3.  Hermite spline p_h(xi) through the four points, slope 0 at the hub
      ends and m at the LE/TE points: the centre curve, meeting the hub
      ends orthogonally.
  4-6. Edges: p_h1 = p_h + pi/Z, p_h3 = p_h - pi/Z reversed, joined by the
      straight caps p_h2, p_h4 at xi_hi and xi_lo.

Realisation, adapted to this pipeline (the deliberate deviation):
  The report stores the sector as a TRIMMED surface of revolution (entities
  144/120/142/102). That trimming layer is exactly what failed repeatedly
  here before (unbounded cylinder through the Faces-mode writer, invalid
  loop orientations). Instead note that the region between p_h - pi/Z and
  p_h + pi/Z is a RECTANGLE under

      theta(s, xi) = p_h(xi) + (2 s - 1) pi / Z,    s in [0, 1],

  whose four grid edges ARE the report's four curves c_h1..c_h4. So the
  sector is built as ONE structured grid and exported through the same
  self-verified B-spline path as the blade patches: a bounded entity-128
  surface, no trimming anywhere. The blade footprint is likewise not
  trimmed away: the root ring lies on the sector surface to machine
  precision (the hub radius is MEASURED from the root ring, which para.py
  places exactly on a cylinder) and the mesher imprints it.

User controls: hub_height (m, default DEFAULT_HUB_HEIGHT, FIXED so every
design in a batch gets identical hub extents), hub_center (default 0.0,
fixed) and n_blades (Z, sector width 2 pi / Z). The radius takes no input.

Numpy/scipy only; unit-testable without pythonOCC.
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline

RING_ROWS = ("te_strip", "central_pressure", "le_strip", "central_suction")
CYL_SPREAD_TOL = 1.0e-9      # m; the root ring must be this cylindrical
DEFAULT_N_BLADES = 5

# Fixed batch geometry. hub_height=None resolves to DEFAULT_HUB_HEIGHT and
# the hub is centred at DEFAULT_HUB_CENTER, so EVERY design in a sampling
# batch gets the SAME hub extents. (Previously None meant "2x this design's
# root-ring span, centred on this design's ring": each case then had its own
# hub height and position, which is wrong for a batch. Measured over 20 LHS
# designs across the production bounds, the root ring stays within
# x in [-0.19, +0.19] m, so 0.55 m about x = 0 clears every design by >20%;
# the enclosure guard below still rejects any outlier loudly.)
DEFAULT_HUB_HEIGHT = 1.0    # m
DEFAULT_HUB_CENTER = 0.0     # m (x of the hub midpoint); None = per-design
DEFAULT_N_S = 121            # samples across the sector
DEFAULT_N_X = 61             # samples along the axis (p_h(x) is curved)
FOOT_MARGIN_DEG = 1.0        # required clearance footprint <-> sector edge


def root_ring_from_grids(grids):
    """Measure the blade root ring from the patch grids.

    Returns dict(radius, spread, x_lo, x_hi, span, theta_ref).
    """
    ring = np.vstack([np.asarray(grids[k], dtype=float)[0, :, :]
                      for k in RING_ROWS])
    R = np.hypot(ring[:, 1], ring[:, 2])
    radius = float(R.mean())
    spread = float(R.max() - R.min())
    if spread > CYL_SPREAD_TOL:
        raise RuntimeError(
            f"blade root ring is not cylindrical (radius spread "
            f"{spread:.3e} m); a revolution hub cannot match it exactly")
    theta = np.arctan2(ring[:, 1], ring[:, 2])
    theta_ref = float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))
    x_lo, x_hi = float(ring[:, 0].min()), float(ring[:, 0].max())
    return dict(radius=radius, spread=spread, x_lo=x_lo, x_hi=x_hi,
                span=x_hi - x_lo, theta_ref=theta_ref, ring=ring)


def _edge_root_point(grids, key, theta_ref):
    """Hub parameters (xi, theta) of the root LE or TE point.

    The strips' root row is the root cut; its edge column (t = 1/2, the
    middle of the symmetric t_common sampling) is pinned to the LE/TE.
    """
    g = np.asarray(grids[key], dtype=float)
    p = g[0, g.shape[1] // 2, :]
    th = np.arctan2(p[1], p[2]) - theta_ref
    th = float(np.arctan2(np.sin(th), np.cos(th)))
    return float(p[0]), th


def hub_grids(grids, hub_height=None, hub_center=DEFAULT_HUB_CENTER,
              n_blades=DEFAULT_N_BLADES,
              n_s=DEFAULT_N_S, n_x=DEFAULT_N_X, n_rho=25,
              cap_inner_radius=0.0, verbose=True):
    """The Sec. 10 hub sector plus its two end caps (metres).

    hub_height : total axial height in metres (user's knob);
                 None = DEFAULT_HUB_HEIGHT (fixed, batch-consistent).
    hub_center : x of the hub midpoint; None = this design's ring midpoint
                 (per-design). Default DEFAULT_HUB_CENTER = 0.0, fixed.
    n_blades   : Z; the sector spans exactly 2 pi / Z.
    n_rho      : radial samples on each end cap.
    cap_inner_radius : caps run from the hub radius down to this radius.
                 0.0 (default) closes the hub ends completely; set > 0 to
                 leave a shaft bore instead.

    Returns ({"hub_sector": G, "hub_cap_lo": C1, "hub_cap_hi": C2}, info).

    The caps are flat pie wedges at x = xi_lo and x = xi_hi spanning the
    SAME 2 pi / Z as the sector, built over the same s sampling so their
    outer arc coincides with the lateral grid's end column point-for-point.
    Z rotated copies of (sector + caps) therefore close the hub completely:
    barrel and both end disks, no hole. At cap_inner_radius = 0 the inner
    edge collapses onto the axis point (a flat polar cap; the degenerate
    edge lies in the plane of the cap, which meshes as an ordinary disk
    centre, unlike the out-of-plane degenerate tip ring this pipeline
    eliminated).
    """
    log = print if verbose else (lambda *a, **k: None)
    Z = int(n_blades)
    if Z < 2:
        raise ValueError("n_blades must be >= 2")
    info = root_ring_from_grids(grids)
    R, span, th_ref = info["radius"], info["span"], info["theta_ref"]
    x_c = (0.5 * (info["x_lo"] + info["x_hi"]) if hub_center is None
           else float(hub_center))

    H = DEFAULT_HUB_HEIGHT if hub_height is None else float(hub_height)
    margin = 0.02 * span
    xi_lo, xi_hi = x_c - 0.5 * H, x_c + 0.5 * H
    if not (xi_lo < info["x_lo"] - margin and info["x_hi"] + margin < xi_hi):
        raise ValueError(
            f"hub [{xi_lo:.4f}, {xi_hi:.4f}] m (height {H:.4f}, centre "
            f"{x_c:.4f}) does not enclose the blade root ring "
            f"[{info['x_lo']:.4f}, {info['x_hi']:.4f}] m with margin "
            f"{margin:.4f}; increase hub_height or adjust hub_center")

    # --- Sec. 10 steps 1-3: the centre curve p_h(xi) ----------------------
    xi_le, th_le = _edge_root_point(grids, "le_strip", th_ref)
    xi_te, th_te = _edge_root_point(grids, "te_strip", th_ref)
    if abs(xi_te - xi_le) < 1e-9:
        m = 0.0
    else:
        m = (th_te - th_le) / (xi_te - xi_le)                       # Eq. 53
    th_lo = th_le - 0.5 * m * (xi_le - xi_lo)                       # Eq. 51
    th_hi = th_te + 0.5 * m * (xi_hi - xi_te)                       # Eq. 52

    nodes = sorted([(xi_lo, th_lo, 0.0), (xi_le, th_le, m),
                    (xi_te, th_te, m), (xi_hi, th_hi, 0.0)])
    xs = np.array([n[0] for n in nodes])
    ts = np.array([n[1] for n in nodes])
    ds = np.array([n[2] for n in nodes])
    if np.any(np.diff(xs) <= 0):
        raise RuntimeError("hub centre-curve nodes are not strictly "
                           "increasing in x; check the root LE/TE points")
    p_h = CubicHermiteSpline(xs, ts, ds)

    half = np.pi / Z

    # sector must not wrap past +-pi (the report's Eq. 51/52 clamp regime)
    xf = np.linspace(xi_lo, xi_hi, 400)
    if np.max(np.abs(p_h(xf))) + half > np.pi:
        raise RuntimeError(
            "hub sector wraps past +-180 deg; reduce hub_height or check "
            "the blade skew")

    # footprint must sit inside its own sector (else the blades overlap)
    ring = info["ring"]
    th_ring = np.arctan2(ring[:, 1], ring[:, 2]) - th_ref
    th_ring = np.arctan2(np.sin(th_ring), np.cos(th_ring))
    gap = half - np.abs(th_ring - p_h(ring[:, 0]))
    clearance_deg = float(np.degrees(gap.min()))
    if clearance_deg < FOOT_MARGIN_DEG:
        raise RuntimeError(
            f"blade root footprint clears its sector edge by only "
            f"{clearance_deg:.2f} deg (need {FOOT_MARGIN_DEG:.1f}); with "
            f"Z = {Z} the blades would overlap at the root")

    # --- Sec. 10 steps 4-6 as one rectangular grid ------------------------
    # theta(s, xi) = p_h(xi) + (2s - 1) * pi/Z. Grid edges:
    #   s = 1        -> p_h + pi/Z  = c_h1
    #   s = 0        -> p_h - pi/Z  = c_h3 (orientation aside)
    #   xi = xi_hi   -> straight cap = c_h2
    #   xi = xi_lo   -> straight cap = c_h4
    s = np.linspace(0.0, 1.0, int(n_s))
    x = np.linspace(xi_lo, xi_hi, int(n_x))
    TH = th_ref + p_h(x)[None, :] + (2.0 * s - 1.0)[:, None] * half
    G = np.empty((len(s), len(x), 3))
    G[:, :, 0] = x[None, :]
    G[:, :, 1] = R * np.sin(TH)
    G[:, :, 2] = R * np.cos(TH)

    # --- end caps: flat pie wedges closing the hub ends -------------------
    r_in = float(cap_inner_radius)
    if not (0.0 <= r_in < R):
        raise ValueError(f"cap_inner_radius must be in [0, {R:.4f})")
    rho = np.linspace(R, r_in, int(n_rho))          # outer arc -> inner edge
    caps = {}
    for name, xe in (("hub_cap_lo", xi_lo), ("hub_cap_hi", xi_hi)):
        th_e = th_ref + p_h(xe) + (2.0 * s - 1.0) * half
        C = np.empty((len(rho), len(s), 3))
        C[:, :, 0] = xe
        C[:, :, 1] = rho[:, None] * np.sin(th_e)[None, :]
        C[:, :, 2] = rho[:, None] * np.cos(th_e)[None, :]
        caps[name] = C
    # each cap's outer arc must coincide with the sector's end column
    cap_seam = max(
        float(np.abs(caps["hub_cap_lo"][0] - G[:, 0, :]).max()),
        float(np.abs(caps["hub_cap_hi"][0] - G[:, -1, :]).max()))
    if cap_seam > 1e-12:
        raise RuntimeError("hub caps do not share the sector's end columns")

    # --- exactness checks -------------------------------------------------
    # Z rotated copies must tile: edge s=1 rotated by -2 pi/Z == edge s=0
    def _tile(A_last, A_first):
        rot = -2.0 * half
        y_r = A_last[..., 1] * np.cos(rot) + A_last[..., 2] * np.sin(rot)
        z_r = -A_last[..., 1] * np.sin(rot) + A_last[..., 2] * np.cos(rot)
        return float(max(np.abs(y_r - A_first[..., 1]).max(),
                         np.abs(z_r - A_first[..., 2]).max()))

    tile_err = _tile(G[-1, :, :], G[0, :, :])
    tile_err = max(tile_err,
                   _tile(caps["hub_cap_lo"][:, -1, :],
                         caps["hub_cap_lo"][:, 0, :]),
                   _tile(caps["hub_cap_hi"][:, -1, :],
                         caps["hub_cap_hi"][:, 0, :]))
    ring_err = float(np.abs(np.hypot(ring[:, 1], ring[:, 2]) - R).max())

    info.update(hub_height=H, hub_x_lo=xi_lo, hub_x_hi=xi_hi,
                n_blades=Z, sector_deg=float(np.degrees(2 * half)),
                slope_m=float(m), clearance_deg=clearance_deg,
                tile_err=tile_err, ring_on_surface=ring_err,
                cap_seam=cap_seam, cap_inner_radius=r_in,
                xi_le=xi_le, th_le=th_le, xi_te=xi_te, th_te=th_te)

    log(f"[hub] Sec. 10 sector: Z = {Z}, width {np.degrees(2*half):.2f} deg, "
        f"radius {R:.6f} m (measured, spread {info['spread']:.1e} m)")
    log(f"[hub] height {H:.4f} m, axial [{xi_lo:.4f}, {xi_hi:.4f}] m "
        f"(root ring spans [{info['x_lo']:.4f}, {info['x_hi']:.4f}])")
    log(f"[hub] centre curve: LE ({xi_le:.4f}, {np.degrees(th_le):.2f} deg) "
        f"TE ({xi_te:.4f}, {np.degrees(th_te):.2f} deg), "
        f"slope m {m:.4f} rad/m (Eqs. 51-53)")
    log(f"[hub] footprint clearance {clearance_deg:.2f} deg; "
        f"tiling error over Z copies {tile_err:.2e} m; "
        f"root ring on surface to {ring_err:.2e} m")
    log(f"[hub] end caps: pie wedges to "
        + (f"radius {r_in:.4f} m (shaft bore)" if r_in > 0 else "the axis")
        + f", outer arc on the sector to {cap_seam:.1e} m")
    out = {"hub_sector": G}
    out.update(caps)
    return out, info
