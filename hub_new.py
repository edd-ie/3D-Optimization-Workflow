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
      DEVIATION (center_on_footprint=True, the default): the LE and TE are
      the ends of the section's camber line, so a curve through them runs
      as a near-straight chord while the foil body bulges to one side of
      it, leaving the sector's two side edges at unequal distances from
      the root section's two surfaces. p_h is therefore laid instead on
      the MID-LINE of the root footprint, (theta_max + theta_min)/2 at
      each axial station, which is exactly the equal-gap condition. The
      Eq. 51-52 extrapolation and the zero-slope hub-end conditions are
      unchanged; center_on_footprint=False restores the four-point form.
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
from scipy.interpolate import (CubicHermiteSpline, CubicSpline,
                               LSQUnivariateSpline)

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
# Sector sampling. The budget is deliberately shifted from s to x.
#   s: the sector is a 72 deg circular arc at constant radius, which a
#      cubic reproduces almost exactly -- at 41 points (1.8 deg) the
#      interpolation error is ~1e-6 mm, so 121 was spending points on a
#      direction that does not need them.
#   x: this is the direction that has to follow the centred p_h. Measured
#      with 61 columns (~8 mm across the footprint) the exported edge lay
#      1.01 mm off p_h even though p_h itself was within 0.018 mm of the
#      mid-line: the whole remaining error was axial resolution. Cubic
#      error falls as h^4, so 321 columns (~3.1 mm) should leave ~0.02 mm.
# Net grid 41 x 321 vs the old 121 x 61: 1.8x the points, not 5x.
DEFAULT_N_S = 41             # samples across the sector
DEFAULT_N_X = 321            # samples along the axis (p_h(x) is curved)
FOOT_MARGIN_DEG = 1.0        # required clearance footprint <-> sector edge
# Centre-curve fitting. The tuning history, because the answer is not the
# obvious one:
#   41 uniform nodes, interpolating     -> 1.26 mm off-centre
#   61 cosine-clustered, interpolating  -> 0.018 mm, but the strip edge's
#                                          curvature rose 140x, from 857 to
#                                          119926 deg/m^2 (a ~4 mm radius
#                                          turn on a 119 mm hub)
# That spike was an artifact of node PLACEMENT, not of the geometry: with
# an inset of 1e-6 of the span, the clustered end nodes sat microns from
# the footprint's axial extremes, where the section turns around,
# consecutive footprint samples differ in x by microns, and the envelope's
# crossing interpolation is ill-conditioned. Those nodes were interpolating
# numerical noise. Hence two controls:
#   CENTRE_INSET        keeps nodes clear of that degenerate zone;
#   CENTRE_SMOOTH_KNOTS fits the mid-line by least squares through a small
#                       number of knots instead of interpolating it, so p_h
#                       is smooth by construction and cannot chase noise.
# None restores interpolation (sharper, and only sensible with a healthy
# inset). Equal gaps are worth little if they cost the mesher a curvature
# spike, so the default errs toward smooth.
#
# Measured sweep at inset 1%, one design, Z = 5 (max asymmetry as arc on
# the hub / max strip-edge curvature):
#       doc Eqs. 51-53    23.652 mm      857 deg/m^2   1.0x
#       interpolating      1.097 mm    12484           14.6x
#       24 knots           1.348 mm     7145            8.3x
#       16 knots           2.075 mm    10825           12.6x
#       10 knots           1.615 mm     5929            6.9x
#        6 knots           0.672 mm     3924            4.6x   <- default
# The sweep is NOT monotonic in knot count, which is the whole point: a
# stiffer fit comes out both smoother AND better centred, so the extra
# freedom was being spent on noise, not on geometry. 6 knots is the best
# point on both axes at once. Note also that essentially all of the
# residual sits at the two axial extremes, where the edge gaps are ~35 deg
# and 0.7 mm is under 1% of them; at the TIGHTEST station, which is what a
# viewer sees, the two gaps agree to about 0.01 deg.
CENTRE_NODES = 61
CENTRE_SMOOTH_KNOTS = 6
CENTRE_INSET = 0.01          # fraction of footprint span left at each end


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


def _theta_envelope(x_fp, th_fp, x_query):
    """Extreme theta of the closed footprint polyline at each x station.

    Exact for a polyline: every segment that straddles the station
    contributes its linearly interpolated theta, and the min / max over
    those crossings are the two surfaces the strip edges must clear. Using
    crossings rather than a monotone branch split means a section that
    doubles back in x (heavy skew) needs no special case.

    Stations with no crossing come back NaN.
    """
    x0, x1 = x_fp, np.roll(x_fp, -1)
    t0, t1 = th_fp, np.roll(th_fp, -1)
    dx = x1 - x0
    lo = np.full(len(x_query), np.nan)
    hi = np.full(len(x_query), np.nan)
    for k, xq in enumerate(x_query):
        hit = ((x0 - xq) * (x1 - xq) <= 0.0) & (dx != 0.0)
        if not np.any(hit):
            continue
        th = t0[hit] + (xq - x0[hit]) / dx[hit] * (t1[hit] - t0[hit])
        lo[k], hi[k] = th.min(), th.max()
    return lo, hi


def _footprint_theta(fp, th_ref):
    """Footprint points -> (x, theta) with theta measured from th_ref."""
    fp = np.asarray(fp, dtype=float)
    th = np.arctan2(fp[:, 1], fp[:, 2]) - th_ref
    return fp[:, 0], np.arctan2(np.sin(th), np.cos(th))


def _centred_centre_curve(fp, th_ref, xi_lo, xi_hi, n_nodes=CENTRE_NODES,
                          smooth_knots=CENTRE_SMOOTH_KNOTS,
                          inset=CENTRE_INSET):
    """Centre curve p_h(x) laid on the MID-LINE of the blade footprint.

    TM 2013-178 Eqs. 51-53 run p_h through the root LE and TE points. Those
    two points are the ends of the section's camber line, so between them
    p_h is a near-straight chord while the foil body bulges to one side of
    it; the strip edges p_h +- pi/Z then sit closer to one surface of the
    root section than to the other. Here p_h is instead the mid-line of the
    footprint,

        p_h(x) = (theta_max(x) + theta_min(x)) / 2,

    which is precisely the condition that equalises the two edge clearances
    at every axial station (the edges are offset from p_h by the same
    +-pi/Z, so equal gaps <=> p_h halfway between the two surfaces).

    The doc's END conditions are kept unchanged: beyond the footprint the
    curve extrapolates with half the mid-line's chord slope (Eqs. 51-52)
    and reaches both hub ends with zero slope, so the sector still meets
    the end caps orthogonally and Z rotated copies still tile.

    Returns (p_h, x_nodes, mid, m).
    """
    x_fp, th_fp = _footprint_theta(fp, th_ref)
    xa, xb = float(x_fp.min()), float(x_fp.max())
    span = xb - xa
    if not (span > 0.0):
        raise RuntimeError("hub centring: root footprint has no axial extent")
    # Stay out of the degenerate zone at the two axial extremes (see the
    # CENTRE_INSET note above).
    a = xa + float(inset) * span
    b = xb - float(inset) * span

    if smooth_knots:
        # Least-squares fit through a few uniform knots: smooth by
        # construction, so p_h cannot follow either the mid-line's hardest
        # turns or the envelope's noise near the ends.
        xs_d = np.linspace(a, b, 401)
        lo_d, hi_d = _theta_envelope(x_fp, th_fp, xs_d)
        good = np.isfinite(lo_d)
        if good.sum() < 20:
            raise RuntimeError("hub centring: too few usable stations on the "
                               "root footprint")
        xs_d, mid_d = xs_d[good], 0.5 * (lo_d[good] + hi_d[good])
        n_int = max(1, int(smooth_knots))
        t_int = np.linspace(xs_d[0], xs_d[-1], n_int + 2)[1:-1]
        fit = LSQUnivariateSpline(xs_d, mid_d, t_int, k=3)
        x_nodes = np.linspace(a, b, int(n_nodes))
        mid = fit(x_nodes)
    else:
        # Interpolate the mid-line, cosine-clustered toward both ends.
        u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n_nodes))))
        x_nodes = a + (b - a) * u
        lo, hi = _theta_envelope(x_fp, th_fp, x_nodes)
        if not np.all(np.isfinite(lo)):
            raise RuntimeError("hub centring: root footprint has no crossing "
                               "at some axial station; check the root "
                               "section")
        mid = 0.5 * (lo + hi)

    m = (mid[-1] - mid[0]) / (x_nodes[-1] - x_nodes[0])             # Eq. 53
    th_lo = mid[0] - 0.5 * m * (x_nodes[0] - xi_lo)                 # Eq. 51
    th_hi = mid[-1] + 0.5 * m * (xi_hi - x_nodes[-1])               # Eq. 52

    xs = np.concatenate([[xi_lo], x_nodes, [xi_hi]])
    ts = np.concatenate([[th_lo], mid, [th_hi]])
    if np.any(np.diff(xs) <= 0):
        raise RuntimeError("hub centring: centre-curve nodes are not strictly "
                           "increasing in x; the hub is too short for this "
                           "root section")
    # C2 spline with zero end slope: the doc's orthogonal meeting at the
    # hub ends, but following the mid-line in between.
    p_h = CubicSpline(xs, ts, bc_type=((1, 0.0), (1, 0.0)))
    return p_h, x_nodes, mid, float(m)


def hub_grids(grids, hub_height=None, hub_center=DEFAULT_HUB_CENTER,
              hub_radius=None, center_on_footprint=True,
              centre_nodes=CENTRE_NODES,
              centre_smooth=CENTRE_SMOOTH_KNOTS,
              n_blades=DEFAULT_N_BLADES,
              n_s=DEFAULT_N_S, n_x=DEFAULT_N_X, n_rho=25,
              cap_inner_radius=0.0, verbose=True):
    """The Sec. 10 hub sector plus its two end caps (metres).

    hub_height : total axial height in metres (user's knob);
                 None = DEFAULT_HUB_HEIGHT (fixed, batch-consistent).
    hub_center : x of the hub midpoint; None = this design's ring midpoint
                 (per-design). Default DEFAULT_HUB_CENTER = 0.0, fixed.
    center_on_footprint : place the centre curve on the mid-line of the
                 blade root footprint so the sector's two side edges are
                 equidistant from the root section's two surfaces at every
                 axial station. Needs meta['root_footprint']; falls back to
                 the doc's LE/TE construction when it is absent. False
                 restores TM 2013-178 Eqs. 51-53 verbatim.
    centre_nodes : mid-line samples used to build the centred curve.
    centre_smooth : least-squares knots for the mid-line fit (None to
                 interpolate instead); see the CENTRE_SMOOTH_KNOTS note.
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
    # With the root-extended blade the measured ring sits BELOW the hub
    # surface on purpose; the hub must keep the DESIGN root radius.
    if hub_radius is None:
        hub_radius = grids.get("meta", {}).get("hub_radius")
    if hub_radius is not None:
        info["ring_immersion"] = float(hub_radius) - info["radius"]
        info["radius"] = float(hub_radius)
    else:
        info["ring_immersion"] = 0.0
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
    fp = grids.get("meta", {}).get("root_footprint")
    centred = bool(center_on_footprint) and fp is not None
    if centred:
        # p_h on the footprint mid-line: equal edge gaps at every station
        p_h, x_nodes, mid, m = _centred_centre_curve(
            fp, th_ref, xi_lo, xi_hi, centre_nodes, centre_smooth)
    else:
        # TM 2013-178 Eqs. 51-53 verbatim: p_h through the root LE/TE
        if abs(xi_te - xi_le) < 1e-9:
            m = 0.0
        else:
            m = (th_te - th_le) / (xi_te - xi_le)                   # Eq. 53
        th_lo = th_le - 0.5 * m * (xi_le - xi_lo)                   # Eq. 51
        th_hi = th_te + 0.5 * m * (xi_hi - xi_te)                   # Eq. 52

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

    # --- centring diagnostic ----------------------------------------------
    # The two strip edges sit at p_h +- half. At each axial station the gap
    # to the blade is (p_h + half) - theta_max above and theta_min -
    # (p_h - half) below; centring is exactly the statement that these are
    # equal. Reported as an arc length on the hub surface, which is what is
    # visible in CAD.
    if fp is not None:
        x_fp, th_fp = _footprint_theta(fp, th_ref)
        xq = np.linspace(float(x_fp.min()), float(x_fp.max()), 401)[1:-1]
        lo_e, hi_e = _theta_envelope(x_fp, th_fp, xq)
        ok = np.isfinite(lo_e)
        ph_q = p_h(xq[ok])
        d_up = (ph_q + half) - hi_e[ok]
        d_dn = lo_e[ok] - (ph_q - half)
        asym = np.abs(d_up - d_dn)
        offcentre_deg = float(np.degrees(asym.max()))
        offcentre_mm = float(asym.max() * R * 1000.0)
        min_gap_deg = float(np.degrees(min(d_up.min(), d_dn.min())))
        if min_gap_deg <= 0.0:
            raise RuntimeError(
                f"hub strip edge cuts into the blade root footprint "
                f"(gap {min_gap_deg:.3f} deg); with Z = {Z} the sector is "
                f"too narrow for this root section")
    else:
        offcentre_deg = offcentre_mm = min_gap_deg = float("nan")

    # --- Sec. 10 steps 4-6 as one rectangular grid ------------------------
    # theta(s, xi) = p_h(xi) + (2s - 1) * pi/Z. Grid edges:
    #   s = 1        -> p_h + pi/Z  = c_h1
    #   s = 0        -> p_h - pi/Z  = c_h3 (orientation aside)
    #   xi = xi_hi   -> straight cap = c_h2
    #   xi = xi_lo   -> straight cap = c_h4
    s = np.linspace(0.0, 1.0, int(n_s))
    # Axial columns stay UNIFORM, deliberately. The centred p_h curves most
    # near the two ends of the footprint, so concentrating columns there is
    # the obvious way to resolve it -- but X_CAD splines the hub with
    # grid_to_bspline_surface(hgrids[key]), i.e. with the interpolator's
    # own UNIFORM parameterization and no v_params, and unequal spacing
    # under a uniform parameterization is precisely what overshoots into
    # lobes on the wrap patches (see that function's docstring). Resolution
    # is bought with a larger n_x instead, paid for by a smaller n_s.
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
                xi_le=xi_le, th_le=th_le, xi_te=xi_te, th_te=th_te,
                centred=centred, offcentre_deg=offcentre_deg,
                offcentre_mm=offcentre_mm, min_edge_gap_deg=min_gap_deg)

    log(f"[hub] Sec. 10 sector: Z = {Z}, width {np.degrees(2*half):.2f} deg, "
        f"radius {R:.6f} m"
        + (f" (design root; blade ring sunk {info['ring_immersion']*1000:.1f}"
           f" mm below the hub surface for the manual trim)"
           if info['ring_immersion'] > 1e-9 else " (measured from ring)"))
    log(f"[hub] height {H:.4f} m, axial [{xi_lo:.4f}, {xi_hi:.4f}] m "
        f"(root ring spans [{info['x_lo']:.4f}, {info['x_hi']:.4f}])")
    if centred:
        log(f"[hub] centre curve CENTRED on the root-footprint mid-line "
            f"({len(x_nodes)} nodes), slope m {m:.4f} rad/m; hub ends still "
            f"met orthogonally (Eqs. 51-52 extrapolation)")
    else:
        log(f"[hub] centre curve: LE ({xi_le:.4f}, {np.degrees(th_le):.2f} "
            f"deg) TE ({xi_te:.4f}, {np.degrees(th_te):.2f} deg), "
            f"slope m {m:.4f} rad/m (Eqs. 51-53, NOT centred)")
    if np.isfinite(offcentre_deg):
        log(f"[hub] edge-gap asymmetry {offcentre_deg:.4f} deg "
            f"({offcentre_mm:.3f} mm of arc); smallest edge gap "
            f"{min_gap_deg:.2f} deg")
    log(f"[hub] footprint clearance {clearance_deg:.2f} deg; "
        f"tiling error over Z copies {tile_err:.2e} m; "
        f"root ring on surface to {ring_err:.2e} m")
    log(f"[hub] end caps: pie wedges to "
        + (f"radius {r_in:.4f} m (shaft bore)" if r_in > 0 else "the axis")
        + f", outer arc on the sector to {cap_seam:.1e} m")
    out = {"hub_sector": G}
    out.update(caps)
    return out, info
