"""
tip_smoothing_new.py

Sampling and controlled smoothing of blade cuts per DRDC TM 2013-178 Sec. 7.

  - sample_params(): the two-sided tanh distribution of Eqs. 25-28. Max
    spacing d at the cut ends, min spacing h = d/100 at the edge, and the
    sample midpoint lands exactly on the edge (t = 1/2, which the cut spline
    pins to the LE/TE point).
  - laplace_smooth(): the weighted Laplace (4th-difference) filter of Eq. 29
    with per-point weights beta_i of Eq. 30, modulated by the cubic B-spline
    bump (Fig. 6) so smoothing acts only within d_e of the edge along the cut
    and fades to zero at arclength distance d_t from the tip. The edge
    midpoint is never moved, so the tip location is preserved exactly.

Defaults follow the smooth-prop user guide (TM 2013-179): all lengths are
normalized by the propeller diameter D.
"""

import numpy as np

H_OVER_D = 0.01          # h = d/100 (TM 2013-178 Sec. 7)


def bspline_bump(x):
    """Cubic B-spline bump with knots {-1, -1/2, 0, 1/2, 1}, normalized so
    f(0) = 1 and f(|x|>=1) = 0 (TM 2013-178 Fig. 6)."""
    t = 2.0 * np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(t)
    m1 = t <= 1.0
    out[m1] = (2.0 / 3.0) - t[m1] ** 2 + 0.5 * t[m1] ** 3
    m2 = (t > 1.0) & (t < 2.0)
    out[m2] = ((2.0 - t[m2]) ** 3) / 6.0
    return out / (2.0 / 3.0)


def _solve_delta(rhs, tol=1e-12, max_iter=200):
    """Solve sinh(D)/D = rhs for D > 0 (rhs > 1)."""
    if rhs <= 1.0:
        return 0.0
    lo, hi = 1e-8, 1.0
    while np.sinh(hi) / hi < rhs:
        hi *= 2.0
        if hi > 700:
            return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if np.sinh(mid) / mid < rhs:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def sample_params(L, d, h=None):
    """Sample parameters s_i in [0,1] per Eqs. 25-28.

    L : real-space length scale of the half cut, |x1 - x0|
    d : maximum point separation (same units as L)
    h : separation at the edge (default d/100)
    Returns (s, m): sample parameters and the index of the midpoint (edge).
    """
    if h is None:
        h = H_OVER_D * d
    d = min(d, 0.45 * L) if L > 0 else d
    h = min(h, d)
    r = (L - h) / max(L - d, 1e-300)
    if r <= 1.0 or L <= d:
        N = 21
    else:
        N = 1 + 2 * int(round(np.log(d / h) / np.log(r)))
        N = max(N, 11)
    if N % 2 == 0:
        N += 1
    rhs = 2.0 * L / ((N - 1) * np.sqrt(d * h))
    delta = _solve_delta(rhs)
    alpha = np.sqrt(h / d)

    x = np.linspace(0.0, 1.0, N)

    def h_fn(u):
        if delta <= 0:
            return u
        return 0.5 * (1.0 + np.tanh((u - 0.5) * delta) / np.tanh(0.5 * delta))

    def g_fn(u):
        hu = h_fn(u)
        return hu / (alpha + (1.0 - alpha) * hu)

    s = np.where(x <= 0.5, 0.5 * g_fn(2.0 * x),
                 0.5 + 0.5 * g_fn(2.0 * x - 1.0))
    s[0], s[-1] = 0.0, 1.0
    m = (N - 1) // 2
    s[m] = 0.5
    return s, m


def laplace_smooth(y, m, alpha, a_tip, d_t, d_e, n_iter):
    """Weighted Laplace filter, Eqs. 29-30.

    y      : (N, 3) sampled cut points (modified copy returned)
    m      : index of the edge point (held fixed)
    alpha  : relaxation factor
    a_tip  : arclength along the LE/TE outline from this cut's edge point x1
             to the blade tip
    d_t    : smoothing range from the tip along the outline
    d_e    : smoothing range from the edge along the cut
    n_iter : number of filter iterations
    """
    y = np.array(y, dtype=float)
    N = y.shape[0]
    if N < 5 or n_iter <= 0 or alpha <= 0:
        return y

    x1 = y[m].copy()
    f_tip = float(bspline_bump(a_tip / max(d_t, 1e-300)))
    if f_tip <= 0.0:
        return y

    for _ in range(int(n_iter)):
        dist_e = np.linalg.norm(y - x1, axis=1)
        beta = alpha * f_tip * bspline_bump(dist_e / max(d_e, 1e-300))
        beta[m] = 0.0
        upd = np.zeros_like(y)
        upd[2:-2] = (-y[:-4] + 4 * y[1:-3] - 6 * y[2:-2]
                     + 4 * y[3:-1] - y[4:])
        y = y + (beta[:, None] / 6.0) * upd
        y[m] = x1
    return y


def smooth_cut_samples(cut, d, alpha, a_tip, d_t, d_e, n_iter):
    """Sample a ParamCurve on both half-cuts with the tanh rule, smooth the
    3D points, and return (t_samples, smoothed_points, edge_index).

    The two halves [0, 1/2] and [1/2, 1] are sampled separately (each has its
    own L = |end - edge|), then concatenated with the shared edge point.
    """
    x_edge = cut.edge_xyz()
    x_start = cut.xyz(0.0)
    x_end = cut.xyz(1.0)

    s_a, _ = sample_params(np.linalg.norm(x_edge - x_start), d)
    s_b, _ = sample_params(np.linalg.norm(x_end - x_edge), d)
    # first half: map so density increases toward the edge (s=1 end)
    t_a = 0.5 * s_a
    t_b = 0.5 + 0.5 * s_b
    t = np.concatenate([t_a, t_b[1:]])
    m = len(t_a) - 1

    y = cut.xyz(t)
    y_s = laplace_smooth(y, m, alpha, a_tip, d_t, d_e, n_iter)
    return t, y_s, m
