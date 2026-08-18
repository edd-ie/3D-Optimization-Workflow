"""
blade_cuts_new.py

Blade cuts per DRDC TM 2013-178 Sec. 6: the intersection of the blade surface
with a plane through three points x0 (one side), x1 (on the leading edge,
trailing edge, or the closed-tip fold line), x2 (other side). The cut is
traced by marching in the blade parameter space (xi, eta):

  - tangent in parameter space  t = (-b_eta . n, b_xi . n)          (Eq. 18)
  - adaptive step  alpha = A_max (1 - 0.999 |b - x_ref|/|x1 - x_ref|) (Eq. 23)
  - correction back onto the plane: Newton-Raphson along the line
    perpendicular to the step (Eqs. 16, 20), with fallbacks to a
    Newton-Raphson in xi alone (Eq. 21) and to bisection (Eq. 22); both are
    needed near the tip where the coordinate lines become nearly parallel.

Representation: each cut is stored as TWO half curves in parameter space
(x0 -> x1 and x1 -> x2), splined separately and joined at t = 1/2, which is
pinned exactly on the edge point (the role of Eq. 24's knot rule). Two
halves are necessary because a cut crossing the closed tip has a jump in xi
at the fold (the same 3D point is (xi, 1) on one side and (1 - xi, 1) on the
other), and a trailing-edge cut wraps periodically. On each half the traced
points lie exactly on the blade, so sampled cut points are exact surface
points regardless of spline fit error.

xi is CONTINUOUS along a half (it may leave [0, 1]; BladeSurface.b wraps).
"""

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

ALPHA_MAX = 6.0e-3      # Eq. 23 maximum parameter-space step (report: 3e-3;
                        # doubled here since the cut is resampled downstream)
ALPHA_FRAC = 0.99       # Eq. 23 shrink factor (min step = ALPHA_MAX/100)
ETA_NEAR_TIP = 0.90     # edge points at/above this eta use the near-tip
                        # marching strategy (stage march + eta continuation)


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
class BladeCut:
    """A blade cut: two parameter-space half-splines joined at the edge.

    t in [0, 1]; t = 0 at x0, t = 1/2 exactly at the edge point x1,
    t = 1 at x2.
    """

    def __init__(self, blade, half_a, half_b):
        # half_a/half_b: (t_knots in [0,1] local, params (n,2)).
        # PCHIP (shape-preserving) interpolation: a cubic spline of the
        # parameter path overshoots badly at the fold-spike corner of
        # tip-crossing cuts; PCHIP cannot overshoot and the resulting cut
        # points still lie exactly on the blade.
        self.blade = blade
        self.half_a = half_a          # kept so cuts can be cached/rebuilt
        self.half_b = half_b
        ta, pa = half_a
        tb, pb = half_b
        self._xa = PchipInterpolator(ta, pa[:, 0])
        self._ea = PchipInterpolator(ta, pa[:, 1])
        self._xb = PchipInterpolator(tb, pb[:, 0])
        self._eb = PchipInterpolator(tb, pb[:, 1])

    def param(self, t):
        """Parameter-space point(s); vectorized over t in [0, 1]."""
        t = np.asarray(t, dtype=float)
        ua = np.clip(2.0 * t, 0.0, 1.0)
        ub = np.clip(2.0 * t - 1.0, 0.0, 1.0)
        pa = np.stack([self._xa(ua), self._ea(ua)], axis=-1)
        pb = np.stack([self._xb(ub), self._eb(ub)], axis=-1)
        return np.where((t <= 0.5)[..., None], pa, pb)

    def xyz(self, t):
        p = self.param(t)
        eta = np.clip(p[..., 1], self.blade.eta_min, 1.0)
        return self.blade.b(p[..., 0], eta)

    def edge_xyz(self):
        return self.xyz(np.array(0.5))


# ---------------------------------------------------------------------------
# plane utilities and correctors
# ---------------------------------------------------------------------------
def _plane(x0, x1, x2):
    n = np.cross(x1 - x0, x1 - x2)
    m = np.linalg.norm(n)
    if m < 1e-300:
        raise ValueError("Cut points are collinear; no cutting plane.")
    return n / m


def _fdist(blade, p, x0, nhat):
    eta = float(np.clip(p[1], blade.eta_min, 1.0))
    return float(np.dot(blade.b(p[0], eta) - x0, nhat))


def _newton_perp(blade, q, perp, x0, nhat, tol, max_iter=25):
    s = 0.0
    h = 1.0e-7
    for _ in range(max_iter):
        p = q + s * perp
        f = _fdist(blade, p, x0, nhat)
        if abs(f) < tol:
            return q + s * perp, True
        fp = (_fdist(blade, p + h * perp, x0, nhat) - f) / h
        if abs(fp) < 1e-14:
            break
        s += float(np.clip(-f / fp, -10 * ALPHA_MAX, 10 * ALPHA_MAX))
    return q + s * perp, False


def _newton_xi(blade, q, x0, nhat, tol, max_iter=25):
    xi, eta = float(q[0]), float(q[1])
    h = 1.0e-7
    for _ in range(max_iter):
        f = _fdist(blade, (xi, eta), x0, nhat)
        if abs(f) < tol:
            return np.array([xi, eta]), True
        fp = (_fdist(blade, (xi + h, eta), x0, nhat) - f) / h
        if abs(fp) < 1e-14:
            break
        xi -= float(np.clip(f / fp, -10 * ALPHA_MAX, 10 * ALPHA_MAX))
    return np.array([xi, eta]), False


def _bisect_perp(blade, q, perp, x0, nhat, tol, span=5.0 * ALPHA_MAX):
    f0 = _fdist(blade, q, x0, nhat)
    if abs(f0) < tol:
        return q, True
    s_lo = s_hi = None
    for mag in np.linspace(span / 40.0, span, 40):
        for sgn in (+1.0, -1.0):
            s = sgn * mag
            f = _fdist(blade, q + s * perp, x0, nhat)
            if np.sign(f) != np.sign(f0):
                s_lo, s_hi = min(0.0, s), max(0.0, s)
                f_lo = f0 if s_lo == 0.0 else f
                break
        if s_lo is not None:
            break
    if s_lo is None:
        return q, False
    for _ in range(60):
        s_mid = 0.5 * (s_lo + s_hi)
        f = _fdist(blade, q + s_mid * perp, x0, nhat)
        if abs(f) < tol:
            return q + s_mid * perp, True
        if np.sign(f) == np.sign(f_lo):
            s_lo, f_lo = s_mid, f
        else:
            s_hi = s_mid
    return q + 0.5 * (s_lo + s_hi) * perp, True


def _correct(blade, q, perp, x0, nhat, tol):
    p, ok = _newton_perp(blade, q, perp, x0, nhat, tol)
    if ok:
        return p
    p, ok = _newton_xi(blade, q, x0, nhat, tol)
    if ok:
        return p
    p, ok = _bisect_perp(blade, q, perp, x0, nhat, tol)
    return p if ok else q


def _tangent(blade, p, nhat):
    """Eq. 18 tangent via one batched surface evaluation (3 points)."""
    eta = float(np.clip(p[1], blade.eta_min, 1.0))
    h = 1e-6
    e_hi = min(eta + h, 1.0)
    pts = blade.b(np.array([p[0], p[0] + h, p[0]]),
                  np.array([eta, eta, e_hi]))
    b_xi = (pts[1] - pts[0]) / h
    b_eta = (pts[2] - pts[0]) / max(e_hi - eta, 1e-12)
    t = np.array([-np.dot(b_eta, nhat), np.dot(b_xi, nhat)])   # Eq. 18
    m = np.linalg.norm(t)
    return t / max(m, 1e-300)


# ---------------------------------------------------------------------------
# half-cut marching
# ---------------------------------------------------------------------------
def _march_half(blade, p_from, p_to, x_ref, x_edge, nhat, tol,
                max_steps=None, flip_initial=False):
    """March along the cut from p_from to p_to. The Eq. 23 step size shrinks
    as |b - x_ref| approaches |x_edge - x_ref| (dense sampling at the edge).

    The marching direction keeps CONTINUITY with the previous step (the sign
    of the Eq. 18 tangent is chosen to match the previous direction); only
    the very first step aims at the target. Re-aiming every step can make
    the march oscillate when the intersection curve bends away from the
    straight line to the target.
    """
    if max_steps is None:
        max_steps = int(60.0 / ALPHA_MAX)
    L = max(np.linalg.norm(x_edge - x_ref), 1e-300)
    p = np.array(p_from, dtype=float)
    tgt = np.array(p_to, dtype=float)
    pts = [p.copy()]
    prev_dir = None

    # If the half-cut ends (or starts) on the closed-tip fold (eta = 1), the
    # parameter-space curve terminates ON the eta boundary, where clamped
    # corrections stall and drift along the fold line. March only up to
    # eta = 1 - FOLD_DELTA and let the spline bridge the last sliver to the
    # exact fold point (sub-mm in real space; inside the smoothing zone).
    FOLD_DELTA = 5.0e-3
    fold_end = abs(tgt[1] - 1.0) < 1e-12
    fold_start = abs(p[1] - 1.0) < 1e-12
    eta_cap = 1.0 - FOLD_DELTA if (fold_end or fold_start) else 1.0

    for _ in range(max_steps):
        d_tgt = np.linalg.norm(tgt - p)
        t_hat = _tangent(blade, p, nhat)
        if prev_dir is None:
            # Initial sign: head toward the target IN PARAMETER SPACE. A 3D
            # "which way is the target" test is ambiguous at the leading
            # edge (both branches initially close the 3D gap) and sends the
            # march the long way around the intersection loop.
            if np.dot(t_hat, tgt - p) < 0.0:
                t_hat = -t_hat
            if flip_initial:
                t_hat = -t_hat
        elif np.dot(t_hat, prev_dir) < 0.0:
            t_hat = -t_hat

        eta = float(np.clip(p[1], blade.eta_min, 1.0))
        frac = min(np.linalg.norm(blade.b(p[0], eta) - x_ref) / L, 1.0)
        alpha = ALPHA_MAX * max(1.0 - ALPHA_FRAC * frac, 1.0e-3)
        if d_tgt <= max(alpha, 1.0e-6):
            break
        if fold_end and p[1] >= eta_cap:
            break
        q = p + min(alpha, d_tgt) * t_hat
        perp = np.array([t_hat[1], -t_hat[0]])
        p_new = _correct(blade, q, perp, x_ref, nhat, tol)
        p_new[1] = float(np.clip(p_new[1], blade.eta_min, eta_cap))
        step = p_new - p
        if np.linalg.norm(step) < 1e-14:
            break
        prev_dir = step
        pts.append(p_new.copy())
        p = p_new

    pts.append(tgt.copy())
    return np.array(pts)


def _path_len(params):
    return float(np.sum(np.linalg.norm(np.diff(params, axis=0), axis=1)))


def _best_half(blade, p_from, p_to, x_ref, x_edge, nhat, tol,
               detour_factor=2.5):
    """March a half-cut, self-correcting the initial direction.

    An intersection curve is a closed loop, so from any point there are two
    ways around it; the short way is the cut we want. If the first attempt
    wanders (traced parameter length much longer than the straight-line
    distance to the target, i.e. it went the long way round the loop), the
    march is repeated with the opposite initial direction and the shorter
    path is kept. This is what previously produced a single grossly
    oversized cut row near the leading edge, and with it a large spurious
    lobe on the exported surface.
    """
    straight = np.linalg.norm(np.asarray(p_to, dtype=float)
                              - np.asarray(p_from, dtype=float))
    seg = _march_half(blade, p_from, p_to, x_ref, x_edge, nhat, tol)
    if _path_len(seg) <= detour_factor * max(straight, 1e-12):
        return seg
    seg2 = _march_half(blade, p_from, p_to, x_ref, x_edge, nhat, tol,
                       flip_initial=True)
    return seg2 if _path_len(seg2) < _path_len(seg) else seg


def _half_spline_knots(blade, params):
    """Knots proportional to REAL-space arclength of the traced points, so
    the sampling t maps evenly onto the physical cut (important near the
    tip where the parameter metric is degenerate)."""
    pts = blade.b(params[:, 0], np.clip(params[:, 1], blade.eta_min, 1.0))
    a = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                 axis=1))])
    if a[-1] <= 0:
        raise RuntimeError("Degenerate half cut (zero length).")
    t = a / a[-1]
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    keep[-1] = True
    if not keep[-2] and len(t) > 2:
        keep[-2] = False
    return t[keep], params[keep]


def _newton_xi_at_eta(blade, xi_seed, eta, x0, nhat, tol, window=0.06):
    """Solve the plane equation for xi at fixed eta (Eq. 21), seeded from the
    previous xi; bisection fallback in a window around the seed."""
    xi = float(xi_seed)
    h = 1e-7
    for _ in range(30):
        f = _fdist(blade, (xi, eta), x0, nhat)
        if abs(f) < tol:
            return xi, True
        fp = (_fdist(blade, (xi + h, eta), x0, nhat) - f) / h
        if abs(fp) < 1e-14:
            break
        xi += float(np.clip(-f / fp, -0.02, 0.02))
    lo, hi = xi_seed - window, xi_seed + window
    flo = _fdist(blade, (lo, eta), x0, nhat)
    fhi = _fdist(blade, (hi, eta), x0, nhat)
    if flo * fhi <= 0.0:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = _fdist(blade, (mid, eta), x0, nhat)
            if abs(fm) < tol:
                return mid, True
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi), True
    return xi, False


def _march_to_fold(blade, p_side, p_fold, x_side, x_fold, nhat, tol,
                   switch_frac=0.08, n_climb=60):
    """Trace a half-cut whose edge point lies AT or NEAR the blade tip.

    Near the tip the parameter-space cut becomes a needle-like spike (the
    sine radius mapping compresses real distances there, and the cutting
    plane is nearly tangent to the blade), which plain stepwise marching
    either skips or crawls through until it exhausts its step budget.

    Strategy: standard marching (stage 1) until the 3D distance to the edge
    point drops below switch_frac * |x_edge - x_side|, then eta-continuation
    up to the edge's own eta, solving for xi with a Newton/bisection at each
    eta (stage 2; TM 2013-178 Eq. 21 applied systematically). Works for an
    edge exactly on the fold (eta = 1) and for one just below it.
    Returns points ordered side -> edge.
    """
    L = max(np.linalg.norm(x_fold - x_side), 1e-300)
    eta_target = float(np.clip(p_fold[1], blade.eta_min, 1.0))

    # stage 1: standard march with a 3D stopping rule
    p = np.array(p_side, dtype=float)
    pts = [p.copy()]
    prev_dir = None
    max_steps = int(60.0 / ALPHA_MAX)
    for _ in range(max_steps):
        eta = float(np.clip(p[1], blade.eta_min, 1.0))
        b_here = blade.b(p[0], eta)
        if np.linalg.norm(b_here - x_fold) < switch_frac * L:
            break
        t_hat = _tangent(blade, p, nhat)
        if prev_dir is None:
            # parameter-space direction toward the edge (a 3D test is
            # ambiguous at the leading/trailing edge)
            if np.dot(t_hat, np.asarray(p_fold, dtype=float) - p) < 0.0:
                t_hat = -t_hat
        elif np.dot(t_hat, prev_dir) < 0.0:
            t_hat = -t_hat
        frac = min(np.linalg.norm(b_here - x_side) / L, 1.0)
        alpha = ALPHA_MAX * max(1.0 - ALPHA_FRAC * frac, 1.0e-3)
        q = p + alpha * t_hat
        perp = np.array([t_hat[1], -t_hat[0]])
        p_new = _correct(blade, q, perp, x_side, nhat, tol)
        p_new[1] = float(np.clip(p_new[1], blade.eta_min,
                                 max(eta_target - 1e-4, blade.eta_min)))
        step = p_new - p
        if np.linalg.norm(step) < 1e-14:
            break
        prev_dir = step
        pts.append(p_new.copy())
        p = p_new

    # stage 2: eta-continuation up the near-tip spike to the edge's eta
    eta0 = float(pts[-1][1])
    xi = float(pts[-1][0])
    if eta_target > eta0:
        for k in range(1, n_climb + 1):
            eta = eta_target - (eta_target - eta0) * np.cos(
                0.5 * np.pi * k / n_climb)
            eta = min(eta, eta_target - 1e-6)
            xi_new, ok = _newton_xi_at_eta(blade, xi, eta, x_side, nhat, tol)
            if ok:
                xi = xi_new
            pts.append(np.array([xi, eta]))

    pts.append(np.array(p_fold, dtype=float))
    return np.array(pts)


def trace_cut(blade, p0, p_edge_a, p2, p_edge_b=None, tol_factor=1.0e-8):
    """Trace the blade cut through three points given in parameter space.

    p0       : (xi, eta) of x0 on the starting side
    p_edge_a : (xi, eta) of the edge point x1 as seen from the p0 side
    p2       : (xi, eta) of x2 on the ending side (xi continuous with the
               p_edge_b representation; e.g. a TE cut may use negative xi)
    p_edge_b : representation of the SAME edge point on the p2 side. Needed
               when the cut crosses the closed-tip fold (xi jumps to 1-xi)
               or when a different periodic branch is convenient. Defaults
               to p_edge_a.
    """
    if p_edge_b is None:
        p_edge_b = p_edge_a
    p0 = np.asarray(p0, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p_edge_a = np.asarray(p_edge_a, dtype=float)
    p_edge_b = np.asarray(p_edge_b, dtype=float)

    x0 = blade.b(p0[0], p0[1])
    x1 = blade.b(p_edge_a[0], np.clip(p_edge_a[1], blade.eta_min, 1.0))
    x1b = blade.b(p_edge_b[0], np.clip(p_edge_b[1], blade.eta_min, 1.0))
    x2 = blade.b(p2[0], p2[1])
    if np.linalg.norm(x1 - x1b) > 1e-6 * max(np.linalg.norm(x1 - x0), 1.0):
        raise ValueError("p_edge_a and p_edge_b are not the same 3D point.")
    nhat = _plane(x0, x1, x2)
    scale = max(np.linalg.norm(x1 - x0), np.linalg.norm(x1 - x2))
    tol = tol_factor * scale

    # Any edge point in the top of the blade gets the near-tip treatment:
    # there the cutting plane is nearly tangent to the surface and the
    # parameter-space cut is a needle spike that plain marching cannot
    # follow (it exhausts its step budget in EITHER direction, so the
    # short-path retry cannot rescue it either).
    near_tip = float(p_edge_a[1]) >= ETA_NEAR_TIP
    if near_tip:
        seg_a = _march_to_fold(blade, p0, p_edge_a, x0, x1, nhat, tol)
        seg_b = _march_to_fold(blade, p2, p_edge_b, x2, x1, nhat, tol)[::-1]
    else:
        seg_a = _best_half(blade, p0, p_edge_a, x0, x1, nhat, tol)
        seg_b = _best_half(blade, p_edge_b, p2, x2, x1, nhat, tol)

    return BladeCut(blade,
                    _half_spline_knots(blade, seg_a),
                    _half_spline_knots(blade, seg_b))
