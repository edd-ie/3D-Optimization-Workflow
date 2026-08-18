"""
tip_surfaces_new.py

Five-surface decomposition of the blade per DRDC TM 2013-178:

  - the blade surface is split into central-pressure, central-suction,
    leading-edge strip, trailing-edge strip and tip surfaces (Secs. 3-5);
  - the wrap surfaces are families of blade cuts (Sec. 6) smoothed near the
    tip (Sec. 7);
  - curves below a transition point are built by trans-finite interpolation
    in parameter space so they reach the (non-planar) root ring (Sec. 8);
  - all five surfaces are returned as structured point grids whose shared
    edges are IDENTICAL point rows/columns, so the downstream spline
    surfaces (X_CAD_new.py) match watertight.

The blade root here is the open ring at r = r_root (this pipeline has no
hub), which plays the role of the blade-hub intersection in the report.

Coordinates follow blade_surface_new.BladeSurface: xi in [0,1] periodic
(0/1 = TE, 0.5 = LE; xi < 0.5 is the para.py "lower" side, called pressure
here), eta in [eta_min, 1]. Where a curve wraps the trailing edge, xi is
kept CONTINUOUS by letting it go negative (xi_s - 1).
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline

from blade_cuts_new import trace_cut, BladeCut
from tip_smoothing_new import smooth_cut_samples, bspline_bump


# ---------------------------------------------------------------------------
# configuration (lengths per propeller diameter D, as in TM 2013-179)
# ---------------------------------------------------------------------------
@dataclass
class TipConfig:
    # split points. s_le / s_te = fractional arclength along the outline
    # (0 = TE-root, 1 = LE-root). If None (default) they are placed a
    # fraction `tip_extent` of the way from the root end of each edge toward
    # the tip, which reproduces the smooth-prop defaults on P4382 and adapts
    # to blades with different skew (this blade has s_tip ~ 0.37, not 0.5).
    s_le: float = None
    s_te: float = None
    # "eta"    = both split points at the same radial station eta_top, which
    #            is what Sec. 4 actually specifies: x_LE = b(1/2, eta_LE) and
    #            x_TE = b(0, eta_TE) are given by eta, and s_LE / s_TE are
    #            derived from them. Both strips then end at the same radius so
    #            both top edges are short, as drawn in Fig. 2. Default.
    # "strips" = equal LE/TE strip lengths, exact delta_c spacing on both.
    #            On a skewed blade this ends the two strips at DIFFERENT radii
    #            (eta_TE 0.641 vs eta_LE 0.540), which stretches the TE top
    #            edge into a long sliver.
    # "tip"    = equal tip legs, symmetric tip cap.
    #
    # "strips" is the default because of a hard geometric constraint, not
    # taste. r/R = sin(pi eta / 2), so eta near 1 is essentially the tip.
    # With "tip" on this skewed blade x_LE lands at eta = 0.878, i.e. 98% of
    # radius, where the section is so short that the widest possible LE strip
    # is 0.033 m; a constant-width strip is then impossible and it pinches.
    # "strips" puts x_LE at eta = 0.540, 75% of radius, where 0.144 m is
    # available and the requested strip_width fits comfortably.
    split_mode: str = "eta"
    # Radial station of x_LE and x_TE for split_mode="eta" (Sec. 4: the two
    # points are specified by eta, and s_LE / s_TE are derived from them).
    #
    # 0.64 is the value checked against the exported geometry: it puts both
    # top edges at r = 0.4950 -> 0.5916 m. Under the old "strips" mode the TE
    # happened to land there and was correct, while the LE landed at 0.540
    # (r -> 0.5251) and was too short. Both are now placed together.
    #
    # Reference for reading a view: le_strip lies at y > 0, te_strip at y < 0.
    eta_top: float = 0.64
    top_offset: float = 0.14        # used only when eta_top is None
    tip_extent: float = 0.7

    # Sec. 5: target width of the LE / TE wrap strips, per diameter D. The
    # four corner points are solved so that both ends of each side curve sit
    # this far from their edge, i.e. d_LR = d_UR in Eq. 10, which by Eq. 11
    # holds d(u) constant and gives the strip a nearly constant width.
    # strip_width_te = None means "same as the LE".
    # Set strip_width = None to fall back to the fixed xi corners below.
    strip_width: float = 0.07
    strip_width_te: float = None

    # fixed corner xi, used only when strip_width is None
    xi_ll: float = 0.1647      # root corner xi, pressure TE side
    xi_lr: float = 0.3324      # root corner xi, pressure LE side
    ur: tuple = (1.0 / 3.0, 0.5)     # (xi, eta) upper-right corner (LE side)
    ul: tuple = (1.0 / 6.0, 0.5)     # (xi, eta) upper-left corner (TE side)

    # curve spacing and smoothing (per diameter D)
    delta_c: float = 0.02      # max spacing of wrap curves along LE/TE
    d: float = 0.025           # max point separation on a sampled cut
    d_e: float = 0.02          # smoothing range from the edge along a cut
    d_t: float = 0.10          # smoothing range from the tip along the outline
    alpha: float = 0.5         # Laplace relaxation factor
    # Sec. 7 Laplace iterations. DEFAULT 0 = smoothing OFF, deliberately.
    #
    # The Sec. 7 filter exists to fix "rapidly varying geometry" near the tip
    # of traditionally-defined blades; smooth-prop's own example applies
    # corrections below 1e-4 D = 0.14 mm (TM 2013-179 Fig. 10). This
    # pipeline's blade surface is constructed smooth analytically (elliptical
    # chord taper, thickness floor, rounded dull-edge tip closure, property
    # curves to r = 1), so there is nothing left for the filter to fix.
    # Measured with n_iter = 100: it displaced the tip patch 3.3 mm off the
    # surface, 24x the doc's reference, with a cliff onset (0 -> 3.3 mm
    # between adjacent samples) at its d_e support boundary. That cliff is a
    # near-discontinuity running parallel to the edge: visible as a crease
    # in shaded CAD and hostile to meshing. Set > 0 only for blades whose
    # raw geometry is genuinely irregular near the tip.
    n_iter: int = 0

    # transition points (fraction of the way from the root end of the LE/TE
    # toward the x_LE / x_TE split points); None = halfway (report guidance)
    trans_le: float = 0.5
    trans_te: float = 0.5

    # sampling of the output grids
    n_wrap: int = 241         # samples across each wrap curve (odd)
    n_outline: int = 4000      # dense samples for the outline curve
    max_curves: int = 120      # safety cap on N_LE / N_t
    cluster: float = 0.2       # center-cluster strength of the cross samples
    # Clustering of the TIP cuts toward the two ends of the tip patch, where
    # the outline turns sharply and Eq. 33's uniform spacing under-resolves.
    # 0.0 = Eq. 33 exactly.
    tip_cluster: float = 0.6


# ---------------------------------------------------------------------------
# small parameter-space curve helpers
# ---------------------------------------------------------------------------
class ParamLine:
    """Straight line in parameter space, reparameterized by fractional
    real-space arclength (the a_i(u) distributions of TM 2013-178 Sec. 5)."""

    def __init__(self, blade, p_a, p_b, n=201):
        self.p_a = np.asarray(p_a, dtype=float)
        self.p_b = np.asarray(p_b, dtype=float)
        w = np.linspace(0.0, 1.0, n)
        params = self.p_a[None, :] + w[:, None] * (self.p_b - self.p_a)
        pts = blade.b(params[:, 0], np.clip(params[:, 1], blade.eta_min, 1.0))
        a = np.concatenate([[0.0],
                            np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                     axis=1))])
        a /= max(a[-1], 1e-300)
        a = np.maximum.accumulate(a)
        keep = np.concatenate([[True], np.diff(a) > 1e-13])
        keep[-1] = True
        self._w_of_u = CubicSpline(a[keep], w[keep])

    def __call__(self, u):
        w = np.clip(self._w_of_u(np.asarray(u, dtype=float)), 0.0, 1.0)
        return self.p_a + np.multiply.outer(w, self.p_b - self.p_a)


class LinearParamLine:
    """Plain linear interpolation in parameter space (root arcs, Eqs. 2-5)."""

    def __init__(self, p_a, p_b):
        self.p_a = np.asarray(p_a, dtype=float)
        self.p_b = np.asarray(p_b, dtype=float)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return self.p_a + np.multiply.outer(u, self.p_b - self.p_a)


class _ShiftedCurve:
    """xi-shift wrapper (converts the negative continuous rep to native)."""

    def __init__(self, base, dx):
        self.base = base
        self.dx = float(dx)

    def __call__(self, u):
        p = np.asarray(self.base(u), dtype=float).copy()
        p[..., 0] += self.dx
        return p


class SplineParamCurve:
    """Cubic spline through parameter points p_i at parameters u_i."""

    def __init__(self, u, params):
        params = np.asarray(params, dtype=float)
        self._x = CubicSpline(u, params[:, 0])
        self._e = CubicSpline(u, params[:, 1])

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return np.stack([self._x(u), self._e(u)], axis=-1)


# ---------------------------------------------------------------------------
# the outline curve c_LETE (TM 2013-178 Sec. 8, Fig. 7)
# ---------------------------------------------------------------------------
class OutlineCurve:
    """Curve along the TE (root->tip), across the closed tip line, and back
    along the LE (tip->root), parameterized by fractional arclength s.

    s = 0 where the TE meets the root ring; s = 1 where the LE does.
    """

    def __init__(self, blade, n=4000):
        self.blade = blade
        n3 = n // 3
        eta = np.linspace(blade.eta_min, 1.0, n3)
        te = np.stack([np.zeros_like(eta), eta], axis=-1)            # TE branch
        xi_t = np.linspace(0.0, 0.5, n3)[1:]
        tipl = np.stack([xi_t, np.ones_like(xi_t)], axis=-1)         # tip line
        eta_b = np.linspace(1.0, blade.eta_min, n3)[1:]
        le = np.stack([0.5 * np.ones_like(eta_b), eta_b], axis=-1)   # LE branch
        self.params = np.vstack([te, tipl, le])
        self.branch = np.concatenate([
            np.zeros(len(te), dtype=int),          # 0 = TE
            np.ones(len(tipl), dtype=int),         # 1 = tip line
            2 * np.ones(len(le), dtype=int),       # 2 = LE
        ])
        pts = blade.b(self.params[:, 0], self.params[:, 1])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        a = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(a[-1])
        self.s = a / max(a[-1], 1e-300)
        self.pts = pts
        # blade tip = point on the outline farthest from the propeller axis
        # (the x axis after para.py's rotations): TM 2013-177 Sec. 6.1.4.
        radius = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
        self.i_tip = int(np.argmax(radius))
        self.s_tip = float(self.s[self.i_tip])

    def s_at_eta(self, eta, branch):
        """Fractional arclength s of the point at radial station `eta` on the
        TE (branch 0) or LE (branch 2).

        Sec. 4 specifies the split points by eta:
            x_LE = b(1/2, eta_LE),  x_TE = b(0, eta_TE)
        and then DERIVES s: "The value of s for which c_LETE(s) = x_LE is
        denoted s_LE." This is that derivation. Doing it the other way round
        (picking s and reading off eta) is what made the two strips end at
        different radial stations, since the TE and LE have different lengths
        on a skewed blade.
        """
        m = self.branch == branch
        idx = np.flatnonzero(m)
        e = self.params[idx, 1]
        s = self.s[idx]
        order = np.argsort(e)
        return float(np.interp(float(eta), e[order], s[order]))

    def param_at_s(self, s):
        """(xi, eta) and branch id at fractional arclength s."""
        s = float(np.clip(s, 0.0, 1.0))
        i = int(np.searchsorted(self.s, s))
        i = min(max(i, 1), len(self.s) - 1)
        w = (s - self.s[i - 1]) / max(self.s[i] - self.s[i - 1], 1e-300)
        p = (1 - w) * self.params[i - 1] + w * self.params[i]
        br = self.branch[i if w > 0.5 else i - 1]
        # snap onto the exact branch definition
        if br == 0:
            p[0] = 0.0
        elif br == 2:
            p[0] = 0.5
        else:
            p[1] = 1.0
        return p, int(br)

    def xyz_at_s(self, s):
        p, _ = self.param_at_s(s)
        return self.blade.b(p[0], p[1])


# ---------------------------------------------------------------------------
# Newton projections
# ---------------------------------------------------------------------------
def _project_along_normal(blade, x_m, nhat, p_init, tol=1e-10, max_iter=40):
    """Solve b(p) = x_m + t*nhat for (xi, eta, t): TM 2013-178 Eq. 13."""
    p = np.array([p_init[0], p_init[1], 0.0])
    h = 1e-7
    for _ in range(max_iter):
        eta = float(np.clip(p[1], blade.eta_min, 1.0))
        F = blade.b(p[0], eta) - x_m - p[2] * nhat
        if np.linalg.norm(F) < tol:
            break
        J = np.empty((3, 3))
        J[:, 0] = (blade.b(p[0] + h, eta) - blade.b(p[0] - h, eta)) / (2 * h)
        e_hi = min(eta + h, 1.0)
        e_lo = max(eta - h, blade.eta_min)
        J[:, 1] = (blade.b(p[0], e_hi) - blade.b(p[0], e_lo)) / (e_hi - e_lo)
        J[:, 2] = -nhat
        try:
            dp = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        dp = np.clip(dp, -0.05, 0.05)
        p += dp
    p[1] = float(np.clip(p[1], blade.eta_min, 1.0))
    return p[:2]


def edge_reference(blade, edge_xi, eta_max=1.0, n=1200):
    """Dense sample of a leading or trailing edge, root to `eta_max`.

    eta_max MATTERS. The minimum distance to an edge curve is a
    multi-branch function: on a skewed blade a point beside the trailing edge
    can be closer to the part of that edge up near the tip than to the part
    beside it. Sampling all the way to eta = 1 lets the foot jump branches,
    and it does: for the TE upper corner the foot ran along at eta ~ 0.71 up
    to xi = 0.125 and then snapped to eta = 0.967 at xi = 0.15. Past that
    jump the distance is measured to a fixed point near the tip, so it stops
    growing with xi, the root of (distance - W) sits right on the
    discontinuity, and the corner is placed wrongly.

    Each side curve therefore measures against its OWN branch: the edge from
    the root up to the strip's top point (x_LE or x_TE), and no further.

    Sec. 5 asks for the distance "from the leading edge", i.e. from the EDGE
    CURVE. Eq. 12 evaluates it along the segment joining b(p1(u)) and
    b(p2(u)), which is the same thing only when that segment runs
    perpendicular to the edge. On a skewed blade the edge sweeps back, the
    segment is strongly slanted, and the two measures diverge badly: with
    d(u) held at 0.098 m along the segment, the true distance to the edge
    fell to 0.030 m at u = 0.9, so the strip pinched to a third of its
    intended width exactly where the skew is worst.

    Everything here therefore measures the true (minimum) distance to this
    curve, which is the quantity Sec. 5 describes and the one that governs
    cell size when the strip is meshed.
    """
    hi = float(np.clip(eta_max, blade.eta_min + 1e-6, 1.0))
    ee = np.linspace(blade.eta_min, hi, int(n))
    return blade.b(np.full_like(ee, float(edge_xi)), ee)


def dist_to_edge(E, x):
    """Minimum distance from point `x` to the sampled edge curve `E`."""
    return float(np.min(np.linalg.norm(E - np.asarray(x), axis=1)))


def _corner_at_width(blade, eta, xi_edge, xi_limit, width, iters=90, E=None):
    """xi on the section at `eta` whose distance from the edge is `width`.

    Sec. 5 wants the two end-points of a side curve to be the SAME distance
    from their edge, which by Eq. 11 makes the strip width constant. This
    solves for one such end-point, using the true distance to the edge curve.

    Returns (xi, ok). ok is False when the section is shorter than `width`,
    in which case xi_limit is returned and the caller should warn.
    """
    e = float(np.clip(eta, blade.eta_min, 1.0))
    if E is None:
        E = edge_reference(blade, xi_edge)

    def excess(xi):
        return dist_to_edge(E, blade.b(float(xi), e)) - width

    lo, hi = float(xi_edge), float(xi_limit)
    if excess(hi) < 0.0:
        return hi, False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), True


def build_side_curve(blade, p_lo, p_hi, edge_xi, p_edge_hi, n=41):
    """The c5..c8 construction (TM 2013-178 Sec. 5, Eqs. 8-14).

    p_lo, p_hi : parameter corners of the curve (on the root ring / upper)
    edge_xi    : xi of the adjacent edge (0.5 for LE side, 0.0 for TE side;
                 may be given in the continuous rep of this side)
    p_edge_hi  : parameter point on the edge adjacent to p_hi (x_LE or x_TE)
    """
    guide = ParamLine(blade, p_lo, p_hi)                       # p1(u), Eq. 8
    edge = ParamLine(blade, (edge_xi, blade.eta_min),
                     p_edge_hi)                                # p2(u), Eq. 9
    # measure against this strip's own branch of the edge only
    E = edge_reference(blade, edge_xi, float(p_edge_hi[1]))

    u = np.linspace(0.0, 1.0, n)
    b1 = blade.b(guide(u)[:, 0], np.clip(guide(u)[:, 1], blade.eta_min, 1.0))
    b2 = blade.b(edge(u)[:, 0], np.clip(edge(u)[:, 1], blade.eta_min, 1.0))
    # Eq. 10, but as the true distance to the edge curve rather than the
    # length of the slanted segment (see edge_reference).
    d_lo = dist_to_edge(E, b1[0])
    d_hi = dist_to_edge(E, b1[-1])
    d_u = d_lo + u * (d_hi - d_lo)                             # Eq. 11

    params = np.empty((n, 2))
    params[0] = p_lo
    params[-1] = p_hi
    for k in range(1, n - 1):
        a, c = b2[k], b1[k]
        # Eq. 12: the point on the segment [b(p2), b(p1)] whose distance from
        # the EDGE is d(u). The distance grows from 0 at a (which lies on the
        # edge) out to dist_to_edge(c), so bisection is safe. Falls back to
        # the segment end when the segment is too short to reach d(u).
        if dist_to_edge(E, c) <= d_u[k]:
            x_m = c
        else:
            t0, t1 = 0.0, 1.0
            for _ in range(60):
                tm = 0.5 * (t0 + t1)
                if dist_to_edge(E, a + tm * (c - a)) > d_u[k]:
                    t1 = tm
                else:
                    t0 = tm
            x_m = a + 0.5 * (t0 + t1) * (c - a)
        g = guide(u[k])
        nhat = blade.normal(g[0], float(np.clip(g[1], blade.eta_min, 1.0)))
        params[k] = _project_along_normal(blade, x_m, nhat, g)  # Eq. 13
    return SplineParamCurve(u, params)


# ---------------------------------------------------------------------------
# trans-finite interpolation below the transition cut (Sec. 8)
# ---------------------------------------------------------------------------
class StripTFI:
    """TFI patch between the root arc, the transition cut, and the two strip
    side curves, with the Eq. 38-40 reparameterization so each curve passes
    through its proper edge point at v = 1/2."""

    def __init__(self, blade, root_arc, side0, side1, cut_j, u_j):
        self.blade = blade
        self.p3 = root_arc          # v in [0,1], pressure -> suction
        self.s0 = side0             # u in [0,1], root -> top (pressure side)
        self.s1 = side1             # u in [0,1], root -> top (suction side)
        self.pc = lambda v: cut_j.param(np.asarray(v, dtype=float))
        self.u_j = float(u_j)
        # corner params
        self.c00 = side0(0.0)       # = p3(0)
        self.c01 = side1(0.0)       # = p3(1)
        self.c10 = side0(self.u_j)  # = pc(0)
        self.c11 = side1(self.u_j)  # = pc(1)

    def q(self, w, v):
        """Plain TFI on the sub-rectangle; w = u/u_j in [0,1]."""
        w = np.asarray(w, dtype=float)
        v = np.asarray(v, dtype=float)
        u = w * self.u_j
        return ((1 - w)[..., None] * self.p3(v)
                + w[..., None] * self.pc(v)
                + (1 - v)[..., None] * self.s0(u)
                + v[..., None] * self.s1(u)
                - ((1 - w) * (1 - v))[..., None] * self.c00
                - ((1 - w) * v)[..., None] * self.c01
                - (w * (1 - v))[..., None] * self.c10
                - (w * v)[..., None] * self.c11)

    def _solve_fwfv(self, p_target, w_i):
        """Newton for q(f_w, f_v) = p_target (Eq. 42)."""
        z = np.array([w_i, 0.5])
        h = 1e-6
        for _ in range(40):
            F = self.q(z[0], z[1]) - p_target
            if np.linalg.norm(F) < 1e-12:
                break
            J = np.empty((2, 2))
            J[:, 0] = (self.q(z[0] + h, z[1]) - self.q(z[0] - h, z[1])) / (2 * h)
            J[:, 1] = (self.q(z[0], z[1] + h) - self.q(z[0], z[1] - h)) / (2 * h)
            try:
                dz = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            z += np.clip(dz, -0.2, 0.2)
        return z

    def curve(self, u_i, p_edge_target):
        """Parameter curve i (u_i < u_j): callable v -> (xi, eta)."""
        w_i = u_i / self.u_j
        f_w, f_v = self._solve_fwfv(np.asarray(p_edge_target, dtype=float),
                                    w_i)

        def p_of_v(v):
            v = np.asarray(v, dtype=float)
            wp = w_i + 4.0 * v * (1 - v) * (f_w - w_i)      # Eq. 39
            vp = v + 4.0 * v * (1 - v) * (f_v - 0.5)        # Eq. 40
            return self.q(wp, vp)                            # Eq. 38

        return p_of_v


# ---------------------------------------------------------------------------
# cross-cut sample distribution for the output grids
# ---------------------------------------------------------------------------
def cross_samples(n, cluster=0.2):
    """n odd samples of [0,1] clustered toward the midpoint (the edge)."""
    if n % 2 == 0:
        n += 1
    u = np.linspace(0.0, 1.0, n)
    v = 2.0 * u - 1.0
    w = v * (cluster + (1.0 - cluster) * v * v)
    return 0.5 * (1.0 + w)


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
def build_drdc_grids(blade, cfg=None, verbose=True, row_cache=None):
    """Build the five DRDC surface grids for a BladeSurface.

    Returns dict with keys:
      'te_strip', 'le_strip', 'tip', 'central_pressure', 'central_suction'
        -> structured (rows, cols, 3) point arrays (metres)
      'meta' -> dict of diagnostics (curve counts, outline data, config)
    Shared edges between grids are identical point sets.

    row_cache : optional path to a pickle file used to checkpoint traced
    cuts and sampled rows, so an interrupted build resumes instead of
    restarting (useful for slow environments / batch runs).
    """
    import pickle, os
    cfg = cfg or TipConfig()
    D = blade.d
    log = print if verbose else (lambda *a, **k: None)

    _cache = {}
    if row_cache and os.path.exists(row_cache):
        with open(row_cache, "rb") as fh:
            _cache = pickle.load(fh)
        log(f"[tip_surfaces] row cache: {len(_cache)} entries loaded")

    def cache_get(key):
        return _cache.get(key)

    def cache_put(key, value):
        if row_cache is None:
            return
        _cache[key] = value
        tmp = row_cache + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(_cache, fh)
        os.replace(tmp, row_cache)

    # ---- outline, split points ------------------------------------------
    outline = OutlineCurve(blade, cfg.n_outline)
    L = outline.length
    s_tip = outline.s_tip
    # Eq. 49 gives the LE and TE strips a SINGLE curve count,
    #     N_LE = N_TE = floor( max(1 - s_le, s_te) L / dc ) + 1,
    # so the curve spacing is dc only on the longer strip; the shorter one
    # gets the same number of curves packed into less arclength. The two
    # strips must therefore be the same length.
    #
    # The TE runs over s in [0, s_tip] and the LE over [s_tip, 1], so on a
    # skewed blade (here s_tip ~ 0.37, not 0.5) the two edges have very
    # different arclengths. Taking the same FRACTION of each, as
    # tip_extent * s_tip and 1 - tip_extent * (1 - s_tip) did, preserves that
    # imbalance and gave strips of 0.26 L and 0.44 L: a 1.7:1 mismatch in both
    # strip width and curve spacing, which is what makes the patch layout look
    # nothing like Fig. 2.
    #
    # Instead put both split points the same arclength from their own root
    # end, limited by the shorter edge:
    #     l = tip_extent * min(s_tip, 1 - s_tip);  s_te = l,  s_le = 1 - l.
    # Both strips are then l*L long, Eq. 49 gives exactly dc spacing on both,
    # and the tip surface [l, 1-l] straddles s_tip. Because l <= tip_extent *
    # s_tip and l <= tip_extent * (1 - s_tip), s_te < s_tip < s_le holds for
    # any skew.
    #
    # This also helps Fig. 8: the split (and with it the transition point s_j,
    # placed half-way along each strip below) stays well away from the tip, so
    # the trans-finite interpolation does not propagate the high
    # parameter-space curvature of the near-tip cuts into the lower curves.
    # Two ways to do that; on a skewed blade they cannot both hold, since
    # equal tip legs means s_te + s_le = 2 s_tip and equal strips means
    # s_te + s_le = 1, which agree only when s_tip = 1/2.
    #
    #   "tip"    equal tip legs: s_te = s_tip - m, s_le = s_tip + m.
    #            The tip cap is symmetric about the tip, as drawn in Fig. 2.
    #   "strips" equal strips: s_te = l, s_le = 1 - l.
    #            Eq. 49 then gives exactly delta_c spacing on both edges.
    half = min(s_tip, 1.0 - s_tip)
    if cfg.split_mode == "eta":
        # Sec. 4: both split points are placed at the SAME radial station,
        # eta_top, just above the level top edge c9 (whose eta is cfg.ur[1]).
        # Both strips then end at the same radius, so both top edges are the
        # short spans Fig. 2 draws. s is derived from eta, per Sec. 8.
        eta_top = (cfg.eta_top if cfg.eta_top is not None
                   else float(cfg.ur[1]) + cfg.top_offset)
        eta_top = float(np.clip(eta_top, blade.eta_min + 1e-4, 0.999))
        s_te_auto = outline.s_at_eta(eta_top, 0)
        s_le_auto = outline.s_at_eta(eta_top, 2)
        eta_te_auto = eta_le_auto = eta_top
    elif cfg.split_mode == "tip":
        m = (1.0 - cfg.tip_extent) * half
        s_te_auto, s_le_auto = s_tip - m, s_tip + m
        eta_te_auto = eta_le_auto = None
    elif cfg.split_mode == "strips":
        ell = cfg.tip_extent * half
        s_te_auto, s_le_auto = ell, 1.0 - ell
        eta_te_auto = eta_le_auto = None
    else:
        raise ValueError(f"split_mode must be 'eta', 'tip' or 'strips', "
                         f"got {cfg.split_mode!r}")
    s_te_split = cfg.s_te if cfg.s_te is not None else s_te_auto
    s_le_split = cfg.s_le if cfg.s_le is not None else s_le_auto
    p_te, _ = outline.param_at_s(s_te_split)
    p_le, _ = outline.param_at_s(s_le_split)
    if not (s_te_split < s_tip < s_le_split):
        raise ValueError(
            f"s_tip={s_tip:.4f} must lie between s_te={s_te_split:.4f} and "
            f"s_le={s_le_split:.4f}; adjust the split points.")
    len_te = s_te_split * L
    len_le = (1.0 - s_le_split) * L
    imbalance = max(len_te, len_le) / max(min(len_te, len_le), 1e-12)
    log(f"[tip_surfaces] outline L={L:.4f} m, s_tip={s_tip:.4f}, "
        f"s_te={s_te_split:.4f}, s_le={s_le_split:.4f}")
    log(f"[tip_surfaces] strip lengths: TE {len_te:.4f} m, LE {len_le:.4f} m "
        f"(ratio {imbalance:.3f}); tip wrap "
        f"{(s_tip - s_te_split) * L:.4f} m aft / "
        f"{(s_le_split - s_tip) * L:.4f} m fwd")
    if imbalance > 1.05:
        log(f"[tip_surfaces] WARNING: LE and TE strips differ by "
            f"{100 * (imbalance - 1):.1f}%. Eq. 49 uses one curve count for "
            f"both, so the shorter strip will be over-refined.")

    eta0 = blade.eta_min
    # ---- corner points: Sec. 5, constant strip width ---------------------
    # "if the two end-points are the same distance from the leading edge, the
    # whole curve is equidistant from the leading edge and the surface
    # wrapping around the leading edge has a nearly constant width."
    #
    # Eq. 11 is d(u) = d_LR + u (d_UR - d_LR), linear in u, so the width is
    # constant if and ONLY if d_LR = d_UR. Eq. 10 defines those two distances
    # as consequences of the corner points, so the corners are what has to be
    # chosen. Fixed xi corners (xi_lr = 0.3324, ur = (1/3, 1/2)) do not do
    # that: they gave d_LR = 0.093 m and d_UR = 0.352 m, a 3.8:1 taper.
    #
    # Only xi is solved. eta of the upper corners stays at the value given in
    # cfg.ur / cfg.ul (the user-guide defaults, 1/2 for both), because Sec. 4
    # specifies p_UL and p_UR as free points and the guide sets them level:
    # UR = (1/3, 1/2), UL = (1/6, 1/2). A level pair keeps the top edge of the
    # central surface, c9, level as drawn in Fig. 2.
    #
    # An earlier version here tied eta_UR to eta_LE and eta_UL to eta_TE. That
    # was to stop Eq. 10 measuring a mostly-radial gap, which only happened
    # under split_mode="tip" where eta_LE reached 0.878. Under "strips"
    # eta_LE = 0.540, already beside 1/2, so the override is unnecessary and
    # it tilted c9 (eta 0.641 -> 0.540), which does not match Fig. 2.
    #
    # Set cfg.strip_width = None to go back to the fixed xi corners.
    if cfg.strip_width is None:
        p_ll = np.array([cfg.xi_ll, eta0])
        p_lr = np.array([cfg.xi_lr, eta0])
        p_ul = np.array([cfg.ul[0], cfg.ul[1]])
        p_ur = np.array([cfg.ur[0], cfg.ur[1]])
    else:
        w_le = cfg.strip_width * D
        w_te = (cfg.strip_width if cfg.strip_width_te is None
                else cfg.strip_width_te) * D
        eta_ur = float(cfg.ur[1])          # Sec. 4 / user guide: 1/2
        eta_ul = float(cfg.ul[1])          # Sec. 4 / user guide: 1/2
        # same branch restriction as build_side_curve: each corner is measured
        # against the edge only up to its own strip's top point
        E_le = edge_reference(blade, 0.5, float(p_le[1]))
        E_te = edge_reference(blade, 0.0, float(p_te[1]))
        xi_lr, ok_lr = _corner_at_width(blade, eta0, 0.5, 0.25, w_le, E=E_le)
        xi_ur, ok_ur = _corner_at_width(blade, eta_ur, 0.5, 0.25, w_le, E=E_le)
        xi_ll, ok_ll = _corner_at_width(blade, eta0, 0.0, 0.25, w_te, E=E_te)
        xi_ul, ok_ul = _corner_at_width(blade, eta_ul, 0.0, 0.25, w_te, E=E_te)
        p_lr = np.array([xi_lr, eta0])
        p_ur = np.array([xi_ur, eta_ur])
        p_ll = np.array([xi_ll, eta0])
        p_ul = np.array([xi_ul, eta_ul])
        log(f"[tip_surfaces] strip width target: LE {w_le:.4f} m, "
            f"TE {w_te:.4f} m (per D = {D:.4f} m)")
        log(f"[tip_surfaces] solved corners: xi_LR {xi_lr:.4f} "
            f"xi_UR {xi_ur:.4f} (eta {eta_ur:.4f}), xi_LL {xi_ll:.4f} "
            f"xi_UL {xi_ul:.4f} (eta {eta_ul:.4f})")
        if not all((ok_lr, ok_ur, ok_ll, ok_ul)):
            log("[tip_surfaces] WARNING: a section is too short for the "
                "requested strip width; that corner was clamped to xi = 0.25 "
                "and the strip will taper there. Reduce strip_width.")
    # suction side = mirror xi -> 1 - xi; the TE-side suction points are kept
    # in the continuous representation (xi - 1 < 0) as before
    p_ll_s = np.array([1.0 - p_lr[0], p_lr[1]])      # LE-side suction root
    p_lr_s_neg = np.array([-p_ll[0], p_ll[1]])       # TE-side suction root
    p_ul_s = np.array([1.0 - p_ur[0], p_ur[1]])      # LE-side suction upper
    p_ur_s_neg = np.array([-p_ul[0], p_ul[1]])       # TE-side suction upper

    # ---- edge curves ------------------------------------------------------
    log("[tip_surfaces] building side curves c5..c8 ...")
    c5 = build_side_curve(blade, p_ll, p_ul, 0.0, p_te)            # pressure TE
    c6 = build_side_curve(blade, p_lr, p_ur, 0.5, p_le)            # pressure LE
    c7 = build_side_curve(blade, p_ll_s, p_ul_s, 0.5, p_le)        # suction LE
    c8 = build_side_curve(blade, p_lr_s_neg, p_ur_s_neg, 0.0, p_te)  # suction TE
    c9 = LinearParamLine(p_ul, p_ur)                               # Eq. 6
    c10 = LinearParamLine(p_ul_s, p_ur_s_neg + np.array([1.0, 0.0]))  # native
    root_te = LinearParamLine(p_ll, p_lr_s_neg)                    # c1-like
    root_cp = LinearParamLine(p_ll, p_lr)                          # c2
    root_le = LinearParamLine(p_lr, p_ll_s)                        # c3
    root_cs = LinearParamLine(p_lr_s_neg + np.array([1.0, 0.0]),
                              p_ll_s)                              # c4 native

    # ---- curve counts (Eqs. 31, 49) --------------------------------------
    dc_abs = cfg.delta_c * D
    N_t = int(round(L * (s_le_split - s_te_split) / dc_abs)) + 1
    N = int(round(L * max(1.0 - s_le_split, s_te_split) / dc_abs)) + 1
    N_t = int(np.clip(N_t, 5, cfg.max_curves))
    N = int(np.clip(N, 5, cfg.max_curves))
    log(f"[tip_surfaces] N_LE = N_TE = {N}, N_tip = {N_t}")
    log(f"[tip_surfaces] curve spacing: TE {len_te / max(N - 1, 1):.4f} m, "
        f"LE {len_le / max(N - 1, 1):.4f} m (delta_c = {dc_abs:.4f} m)")

    d_abs = cfg.d * D
    d_e_abs = cfg.d_e * D
    d_t_abs = cfg.d_t * D
    t_common = cross_samples(cfg.n_wrap, cfg.cluster)

    # ---- transition point s_j (Sec. 8, Eq. 48, Fig. 8) -------------------
    # "Choosing s_j to be about half-way along the leading edge usually works
    # well" -- trans_le = trans_te = 0.5 puts it at the middle of each strip.
    # Fig. 8: if s_j sits near the tip the blade cuts there are highly curved
    # in parameter space and the trans-finite interpolation propagates that
    # curvature into every lower curve. Eq. 48, (s_j - s_tip) L > d_t, is the
    # hard floor (blade cut j must also be outside the smoothing range); the
    # clamps below enforce it, and the log reports the actual margin so a
    # highly skewed blade that is running close to the limit is visible.
    s_trans_te = cfg.trans_te * s_te_split
    s_trans_le = 1.0 - cfg.trans_le * (1.0 - s_le_split)
    smooth_lim_lo = s_tip - d_t_abs / L
    smooth_lim_hi = s_tip + d_t_abs / L
    if s_trans_te > smooth_lim_lo:
        s_trans_te = 0.9 * smooth_lim_lo
        log("[tip_surfaces] WARNING: TE transition clamped to satisfy Eq. 48")
    if s_trans_le < smooth_lim_hi:
        s_trans_le = min(1.0, 1.1 * smooth_lim_hi)
        log("[tip_surfaces] WARNING: LE transition clamped to satisfy Eq. 48")
    log(f"[tip_surfaces] transition s_j: TE {s_trans_te:.4f} "
        f"({(s_tip - s_trans_te) * L:.4f} m from tip), "
        f"LE {s_trans_le:.4f} ({(s_trans_le - s_tip) * L:.4f} m from tip); "
        f"Eq. 48 needs > {d_t_abs:.4f} m")

    u_rows = np.linspace(0.0, 1.0, N)
    # Eq. 33 sets u = (i-1)/(N_t-1), uniform, so the tip cuts are equally
    # spaced in fractional arclength along c_LETE. That is fine where the
    # outline is straight, but near the TE/tip and LE/tip corners it turns
    # sharply, and equal arclength steps there are large steps in radius: the
    # cut length ran 0.2529 -> 0.3616 (43%) between rows 0 and 1 while later
    # rows changed by 0.2%. The tip patch is then badly under-resolved at its
    # two ends and interpolates into a visible flare, the "tongue".
    #
    # tip_cluster blends the uniform Eq. 33 spacing toward a cosine
    # distribution that packs cuts at BOTH ends, where the geometry moves
    # fastest, and thins them in the middle, where consecutive cuts are nearly
    # identical. This is a deliberate departure from Eq. 33; set
    # tip_cluster = 0.0 to recover the report's uniform spacing exactly.
    u_tip = np.linspace(0.0, 1.0, N_t)
    if cfg.tip_cluster > 0.0:
        w = float(np.clip(cfg.tip_cluster, 0.0, 1.0))
        u_tip = (1.0 - w) * u_tip + w * 0.5 * (1.0 - np.cos(np.pi * u_tip))

    def sample_planar(cut, s_edge):
        """Sample one traced cut at t_common, smoothing it if within d_t of
        the tip (resampled through a per-coordinate spline)."""
        a_tip = abs(s_edge - s_tip) * L
        if a_tip < d_t_abs and cfg.n_iter > 0:
            t_s, y_s, m = smooth_cut_samples(
                cut, d_abs, cfg.alpha, a_tip, d_t_abs, d_e_abs, cfg.n_iter)
            # PCHIP: shape-preserving, cannot overshoot between the strongly
            # clustered smoothed samples near the edge
            from scipy.interpolate import PchipInterpolator
            resampled = np.stack(
                [PchipInterpolator(t_s, y_s[:, k])(t_common)
                 for k in range(3)],
                axis=-1)
            # keep the exact endpoints and edge point
            resampled[0] = y_s[0]
            resampled[-1] = y_s[-1]
            i_mid = np.argmin(np.abs(t_common - 0.5))
            resampled[i_mid] = y_s[m]
            return resampled
        return cut.xyz(t_common)

    def edge_param_for(s_edge):
        """Edge parameter representations (a-side, b-side) for a cut whose
        edge point sits at outline arclength s_edge."""
        p, br = outline.param_at_s(s_edge)
        if br == 0:                       # TE branch: continuous through xi=0
            return p, p
        if br == 2:                       # LE branch
            return p, p
        # tip fold: same 3D point, xi jumps to 1 - xi on the suction side
        return p, np.array([1.0 - p[0], 1.0])

    # ---- strip families ---------------------------------------------------
    def build_strip(kind):
        """kind in {'te', 'le'}: rows root->top, cols pressure->suction."""
        rows = np.empty((N, len(t_common), 3))
        cuts = {}
        if kind == "te":
            side_p, side_s, root_arc = c5, c8, root_te
            s_of_u = lambda u: u * s_te_split
            s_trans = s_trans_te
        else:
            side_p, side_s, root_arc = c6, c7, root_le
            s_of_u = lambda u: 1.0 - u * (1.0 - s_le_split)
            s_trans = s_trans_le

        # transition index j: first row whose edge point is on the x_TE/x_LE
        # side of the transition point
        if kind == "te":
            j = next((i for i in range(N) if s_of_u(u_rows[i]) > s_trans), N - 1)
        else:
            j = next((i for i in range(N) if s_of_u(u_rows[i]) < s_trans), N - 1)
        j = max(j, 1)

        # planar cuts for i >= j
        for i in range(j, N):
            u = u_rows[i]
            s_edge = s_of_u(u)
            key = (kind, "planar", N, len(t_common), i)
            hit = cache_get(key)
            if hit is not None:
                cut = BladeCut(blade, hit["half_a"], hit["half_b"])
                cuts[i] = (cut, s_edge)
                rows[i] = hit["row"]
                continue
            pe_a, pe_b = edge_param_for(s_edge)
            cut = trace_cut(blade, side_p(u), pe_a, side_s(u), pe_b)
            cuts[i] = (cut, s_edge)
            rows[i] = sample_planar(cut, s_edge)
            cache_put(key, dict(half_a=cut.half_a, half_b=cut.half_b,
                                row=rows[i]))

        # TFI curves for i < j, anchored to cut j (Sec. 8)
        cut_j = cuts[j][0]
        tfi = StripTFI(blade, root_arc, side_p, side_s, cut_j, u_rows[j])
        for i in range(0, j):
            u = u_rows[i]
            s_edge = s_of_u(u)
            p_tgt, _ = outline.param_at_s(s_edge)
            crv = tfi.curve(u, p_tgt)
            p = crv(t_common)
            rows[i] = blade.b(p[..., 0],
                              np.clip(p[..., 1], blade.eta_min, 1.0))
        return rows, cuts, j

    log("[tip_surfaces] tracing TE strip ...")
    te_rows, te_cuts, j_te = build_strip("te")
    log("[tip_surfaces] tracing LE strip ...")
    le_rows, le_cuts, j_le = build_strip("le")

    # ---- tip family -------------------------------------------------------
    log("[tip_surfaces] tracing tip surface ...")
    tip_rows = np.empty((N_t, len(t_common), 3))
    tip_rows[0] = te_rows[-1]                       # shared cut c11
    tip_rows[-1] = le_rows[-1]                      # shared cut c12
    for k in range(1, N_t - 1):
        u = u_tip[k]
        s_edge = s_te_split + u * (s_le_split - s_te_split)     # Eq. 32
        key = ("tip", "planar", N_t, len(t_common), k)
        hit = cache_get(key)
        if hit is not None:
            tip_rows[k] = hit["row"]
            continue
        pe_a, pe_b = edge_param_for(s_edge)
        p0 = c9(u)
        p2_native = c10(1.0 - u)                          # Eq. 32 (x2)
        _, br = outline.param_at_s(s_edge)
        p2 = p2_native - np.array([1.0, 0.0]) if br == 0 else p2_native
        cut = trace_cut(blade, p0, pe_a, p2, pe_b)
        tip_rows[k] = sample_planar(cut, s_edge)
        cache_put(key, dict(row=tip_rows[k]))

    # ---- central surfaces (param TFI, boundaries shared exactly) ---------
    log("[tip_surfaces] filling central surfaces ...")

    def central(sc_a, sc_b, bot, top):
        """Standard TFI in parameter space. Rows: root -> top (u_rows);
        cols: TE side -> LE side (u_tip). Boundaries: sc_a (col 0),
        sc_b (last col), bot (row 0), top (last row)."""
        rows = np.empty((N, N_t, 3))
        c00, c01 = sc_a(0.0), sc_b(0.0)
        c10_, c11_ = sc_a(1.0), sc_b(1.0)
        for i in range(N):
            u = u_rows[i]
            for k in range(N_t):
                v = u_tip[k]
                q = ((1 - v) * sc_a(u) + v * sc_b(u)
                     + (1 - u) * bot(v) + u * top(v)
                     - (1 - u) * (1 - v) * c00 - (1 - u) * v * c01
                     - u * (1 - v) * c10_ - u * v * c11_)
                rows[i, k] = blade.b(q[0], float(np.clip(q[1],
                                                         blade.eta_min, 1.0)))
        return rows

    cp = central(c5, c6, root_cp, LinearParamLine(p_ul, p_ur))
    cs = central(_ShiftedCurve(c8, +1.0), c7, root_cs,
                 LinearParamLine(p_ur_s_neg + np.array([1.0, 0.0]), p_ul_s))

    # enforce exact shared boundaries with the strip grids
    cp[:, 0] = te_rows[:, 0]
    cp[:, -1] = le_rows[:, 0]
    cs[:, 0] = te_rows[:, -1]
    cs[:, -1] = le_rows[:, -1]
    cp[-1, :] = tip_rows[:, 0]
    cs[-1, :] = tip_rows[:, -1]

    # ---- diagnostics: warn if consecutive rows fold back / cross ---------
    def row_reversals(g, name):
        steps = np.diff(g, axis=0)                      # (rows-1, cols, 3)
        dots = np.sum(steps[1:] * steps[:-1], axis=-1)  # (rows-2, cols)
        n_bad = int(np.sum(dots < 0.0))
        if n_bad:
            log(f"[tip_surfaces] WARNING: {name}: {n_bad} row-direction "
                f"reversals (possible fold/crossing); inspect this grid.")
        return n_bad

    reversals = {name: row_reversals(g, name) for name, g in
                 (("te_strip", te_rows), ("le_strip", le_rows),
                  ("tip", tip_rows), ("central_pressure", cp),
                  ("central_suction", cs))}

    meta = dict(cfg=cfg, N=N, N_t=N_t, L_outline=L, s_tip=s_tip,
                j_te=j_te, j_le=j_le, t_common=t_common,
                eta_min=blade.eta_min, r_tip=blade.r_tip,
                reversals=reversals,
                # split points and the Sec. 8 transition (Eq. 48, Fig. 8)
                s_te=s_te_split, s_le=s_le_split,
                s_trans_te=s_trans_te, s_trans_le=s_trans_le,
                len_te=len_te, len_le=len_le,
                d_t=d_t_abs,
                margin_te=(s_tip - s_trans_te) * L,
                margin_le=(s_trans_le - s_tip) * L)
    return dict(te_strip=te_rows, le_strip=le_rows, tip=tip_rows,
                central_pressure=cp, central_suction=cs, meta=meta)
