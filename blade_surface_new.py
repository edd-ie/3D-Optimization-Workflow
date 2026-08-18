"""
blade_surface_new.py

Smooth global blade-surface parameterization b(xi, eta) following the DRDC
method (Hally, TM 2013-177 / TM 2013-178). This is the foundation of the new
tip-generation pipeline and replaces nothing in the original code base: the
original x_blade.py / para.py aerodynamic path is untouched.

Structure (TM 2013-177 Sec. 6.2, and the warning in Sec. 6.2.3):
  - We do NOT spline the 3D blade surface. We build the chord-normalized
    "surface of reference sections" (canonical airfoil family) which is smooth
    everywhere including the tip, and compose it ANALYTICALLY with the
    property curves (chord, pitch, skew, rake) using exactly the same
    modulation formulas as para.py, so the geometry matches the existing
    blade away from the tip.
  - Parameter conventions (TM 2013-178 Sec. 2):
      xi in [0, 1] around a section: 0 at the trailing edge, along one side to
      0.5 at the leading edge, back along the other side to 1 at the trailing
      edge. Periodic: b(xi - 1, eta) = b(xi, eta).
      eta in [eta_min, 1] from root to tip with r(eta) = r_tip*sin(pi*eta/2)
      (TM 2013-177 Eq. 25) so chord slope stays finite at the tip.
  - Blunt trailing edge (Sec. 6.2.2): the sharp wedge closure of the airfoil
    table is replaced by a rounded (dull) cap built as a C1 Hermite blend of
    thickness^2, which gives a vertical tangent (equal normals) at the TE.
  - Rounded leading edge: thickness is splined against sqrt(station) so the
    section closes at the LE with a vertical tangent (finite LE radius)
    instead of a corner.
  - Tip closure (Sec. 6.2.1.4): on the last radial interval [r_close, r_tip]
    the canonical section is replaced by a Hermite-in-r blend whose tip value
    is the xi-mirror average 0.5*(P(xi) + P(1-xi)) (the mean line) and whose
    tip slope follows Eq. 28. The modification is confined to that interval
    and produces no overshoot.

Only numpy/scipy are used here; no pythonOCC, so this module is unit-testable
anywhere.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

from rot_axis import rot_axis

ROOT = Path(__file__).resolve().parent
AIRFOIL_DATA_PATH = ROOT / "airfoil_data_fixed.csv"

# Geometry constants shared with para.py
PROP_DIAMETER = 1.4          # d in para.py
R_ROOT_DEFAULT = 0.17        # first radial station of the design
TE_STATION_END = 1.005       # para.py appends the closing TE point at X_c = 1.005
TE_BLEND_START = 0.995       # station where the blunt TE cap begins
# Chord (per diameter) at which the Sec. 6.2.1.4 fold begins. The fold turns
# whatever chord remains into a zero-thickness sheet, so it is anchored to the
# CHORD, not to a radius: every fixed radius broke as soon as the taper shape
# changed (0.98 folded 190 mm of chord, 0.996 folded 30 mm, 0.999 folded 7 mm
# under the cubic taper but 37 mm under the elliptical one). With the ROUNDED
# closure (thickness ~ sqrt, vertical tangent, per Sec. 6.2.2's dull-edge
# principle) a wider band is safe and desirable: thickness stays substantial
# through most of the band (sqrt(1-u^3) = 0.94 at mid-band) and rolls off
# only at the very edge, so 0.010 D = 14 mm gives a visibly rounded tip cap
# with no knife sheet anywhere.
CLOSE_CHORD = 0.010
CHORD_FLOOR = 1.0e-4         # minimum chord (per diameter) treated as "closed"

# ---------------------------------------------------------------------------
# Tip chord taper
# ---------------------------------------------------------------------------
# The design defines the chord on R_VALUES, which stops at r/R = 0.999, and
# para_control_bez_updated interpolates with bounds_error=False, so beyond
# that the chord is held CONSTANT at its last value (0.070668 per diameter,
# i.e. 0.0989 m). The blade surface maps eta = 1 to r/R = 1.0, outside that
# data, so it inherits the clamped value and the tip section still has a
# 0.099 m chord.
#
# The tip closure of TM 2013-177 Sec. 6.2.1.4 folds the tip section onto its
# camber line. Applied to a section whose chord has NOT tapered, that produces
# a 0.099 m long tab of zero thickness at r = r_tip: geometrically a knife
# edge, and the "tongue" seen in the CAD. The closure assumes the chord
# tapers toward the tip; this design's does not.
#
# So taper it. Over r/R in [TIP_TAPER_START, r_tip] the chord is scaled by
#
#     f(t) = sqrt(1 - t^3),      t = (r - a) / (r_tip - a)
#
# The SHAPE of this function is the point, and it comes from the report.
# TM 2013-177 Eq. 25 maps r = r_tip sin(pi eta / 2) precisely so that a chord
# behaving like sqrt(r_tip - r) near the tip (an ELLIPTICAL tip, the natural
# propeller tip shape) has finite d(chord)/d(eta): dc/dr -> -inf is cancelled
# by dr/deta -> 0 at the same rate. The framework is designed around designs
# whose chord "reduces rapidly to zero" at the tip (TM 2013-179 Sec. 1).
# Earlier tapers here (1 - t^2, then 1 - t^3) have FINITE dc/dr at the tip:
# they hold the chord nearly full until a few mm from the tip and then drop
# it, which is a square tab planform, and it rendered as exactly the "tongue"
# this taper was meant to remove.
#
# sqrt(1 - t^3) has f(0) = 1 with f'(0) = f''(0) = 0 (C2 join, no shading
# crease at the band start) and f ~ sqrt(3) sqrt(1 - t) near t = 1: the
# elliptical collapse Eq. 25 expects, giving a rounded planform tip with
# finite chord slope in eta. d(chord)/dr stays finite, so
# the Eq. 25 mapping r(eta) = r_tip sin(pi eta / 2) still removes the
# singularity.
#
# Set TIP_TAPER_START = None (or taper_start=None) to disable and recover the
# previous behaviour.
TIP_TAPER_START = 0.90       # r/R at which the tip chord taper begins. Starts
                             # BEFORE the closure band on purpose: chord (and with
                             # it thickness, which is defined per chord) must be
                             # small by the time the fold starts at TIP_CLOSE_START.


# ---------------------------------------------------------------------------
# Canonical (chord-normalized) section family
# ---------------------------------------------------------------------------
class CanonicalSectionFamily:
    """The surface of reference sections (x_s, y_s)(xi, r) in chord units.

    Built from the same airfoil table used by para.py (camber distribution,
    camber slope, thickness distribution), modulated by MaxCamber(r) and
    MaxThickness(r) (both per-chord quantities, as in para.py).
    """

    def __init__(self, max_camber, max_thickness,
                 te_blend_start=TE_BLEND_START, te_station_end=TE_STATION_END):
        self.max_camber = max_camber
        self.max_thickness = max_thickness
        self.s_end = float(te_station_end)
        self.s0 = float(te_blend_start)

        table = pd.read_csv(AIRFOIL_DATA_PATH, skiprows=1, header=None).values
        X_c = table[:, 0]           # stations, 0 (LE) .. 1.0
        y_c = table[:, 1]           # camber distribution (unnormalized)
        der_y = table[:, 2]         # camber-line slope
        th_d = table[:, 3]          # thickness distribution (per chord)

        cam_d = y_c / np.max(y_c)   # same normalization as para.py

        # Extend camber/slope smoothly to the TE closing station (both -> 0
        # there, exactly as para.py appends (1.005, 0, 0, 0)).
        s_ext = np.append(X_c, self.s_end)
        cam_ext = np.append(cam_d, 0.0)
        der_ext = np.append(der_y, 0.0)

        # Camber and slope: splined against q = sqrt(s) so the leading-edge
        # behaviour is regular (NACA-type distributions vary like sqrt(s)
        # near the LE).
        q_ext = np.sqrt(s_ext / self.s_end)
        self._cam = CubicSpline(q_ext, cam_ext)
        self._der = CubicSpline(q_ext, der_ext)

        # Thickness: splined against sqrt(s) over the raw table (0..1.0) so
        # t ~ a*sqrt(s) near the LE => rounded (vertical-tangent) LE.
        q_raw = np.sqrt(X_c / self.s_end)
        self._th = CubicSpline(q_raw, th_d)
        self._q_table_end = q_raw[-1]           # sqrt(1.0/s_end)

        # Blunt TE cap on [s0, s_end]: C1 Hermite blend of t^2 that reaches
        # zero with finite negative d(t^2)/ds, i.e. t ~ sqrt(s_end - s):
        # vertical tangent => dull TE (equal normals across the TE), the
        # closure TM 2013-178 requires. End slope chosen elliptical:
        # d(t^2)/ds(s_end) = -2 t0^2 / (s_end - s0)  (TM 2013-177 Sec. 6.2.2
        # rc = d/2 * sin(beta); sin(beta) ~ 1 at section level).
        q0 = np.sqrt(self.s0 / self.s_end)
        t0 = float(self._th(min(q0, self._q_table_end)))
        dt_dq = float(self._th(min(q0, self._q_table_end), 1))
        dt_ds0 = dt_dq / (2.0 * np.sqrt(self.s0 * self.s_end))
        ell = self.s_end - self.s0
        self._cap = dict(
            t0sq=t0 * t0,
            m0=2.0 * t0 * dt_ds0,           # d(t^2)/ds at s0
            m1=-2.0 * t0 * t0 / ell,        # d(t^2)/ds at s_end (elliptical)
            ell=ell,
        )

    # -- raw distributions ---------------------------------------------------
    def _thickness_dist(self, s):
        """Half-thickness distribution t_d(s) with rounded LE and blunt TE."""
        s = np.asarray(s, dtype=float)
        out = np.empty_like(s)
        in_cap = s >= self.s0
        # main body
        q = np.sqrt(np.clip(s[~in_cap], 0.0, None) / self.s_end)
        out[~in_cap] = self._th(np.minimum(q, self._q_table_end))
        # blunt cap: cubic Hermite of t^2 on [s0, s_end]
        c = self._cap
        u = np.clip((s[in_cap] - self.s0) / c["ell"], 0.0, 1.0)
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        tsq = (h00 * c["t0sq"] + h10 * c["ell"] * c["m0"]
               + h01 * 0.0 + h11 * c["ell"] * c["m1"])
        out[in_cap] = np.sqrt(np.clip(tsq, 0.0, None))
        return out

    def _camber_dist(self, s):
        q = np.sqrt(np.clip(np.asarray(s, dtype=float), 0.0, None) / self.s_end)
        return self._cam(np.clip(q, 0.0, 1.0))

    def _camber_slope(self, s):
        q = np.sqrt(np.clip(np.asarray(s, dtype=float), 0.0, None) / self.s_end)
        return self._der(np.clip(q, 0.0, 1.0))

    # -- xi -> (station, side) mapping ---------------------------------------
    def station_of_xi(self, xi):
        """Map xi in [0,1] to (station s, side).

        side = -1 on the xi<0.5 half (para.py "lower"/y_b side),
        side = +1 on the xi>0.5 half (para.py "upper"/y_u side).
        The mapping is exactly mirror-symmetric: xi and 1-xi give the same
        station, which makes the tip mirror-average closure exact.
        A cosine stretch clusters stations toward both LE and TE.
        """
        xi = np.asarray(xi, dtype=float)
        v = np.abs(1.0 - 2.0 * xi)          # 1 at TE, 0 at LE, symmetric
        s = self.s_end * 0.5 * (1.0 - np.cos(np.pi * v))
        side = np.where(xi < 0.5, -1.0, 1.0)
        return s, side

    # -- canonical section evaluation ----------------------------------------
    def basis(self, xi):
        """Return (s, F_cam, F_th) with the canonical section given by
            xs(xi, r) = s
            ys(xi, r) = F_cam(xi) * MaxCamber(r) + F_th(xi) * MaxThickness(r)

        NOTE: para.py computes the chordwise thickness offset
        (x_u = x_c - th*sin(theta1)) but then maps the 3D point using the
        RAW station x_c, so the offset never enters the legacy geometry.
        To reproduce the existing blade exactly, xs here is the raw station
        as well; only ys carries the thickness term. This makes ys LINEAR in
        (MaxCamber(r), MaxThickness(r)), which the tip closure exploits.
        F_cam is xi-mirror symmetric and F_th antisymmetric.
        """
        xi = np.asarray(xi, dtype=float)
        s, side = self.station_of_xi(xi)
        F_cam = self._camber_dist(s)
        th1 = np.arctan2(self._camber_slope(s), 1.0)
        F_th = side * self._thickness_dist(s) * np.cos(th1)
        return s, F_cam, F_th

    def eval_raw(self, xi, r):
        """Canonical (xs, ys) in chord units, WITHOUT tip closure."""
        xi = np.asarray(xi, dtype=float)
        r = np.asarray(r, dtype=float)
        s, F_cam, F_th = self.basis(xi)
        ys = F_cam * self.max_camber(r) + F_th * self.max_thickness(r)
        return np.stack([np.broadcast_to(s, ys.shape).copy(), ys], axis=-1)


# ---------------------------------------------------------------------------
# The blade surface b(xi, eta)
# ---------------------------------------------------------------------------
class BladeSurface:
    """Smooth, periodic-in-xi blade surface with a closed, smoothed-ready tip.

    Constructed from the SAME property functions the optimizer produces
    (MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake, with Pitch
    and ChordLength already Bezier-modified), so the CAD geometry follows the
    design variables exactly as para.py does away from the tip.
    """

    @staticmethod
    def _c1_extend(fn, r_end=0.999, h=1.0e-4):
        """Extend a design property curve C1-linearly past the data end.

        For r <= r_end the curve is untouched; beyond it, value and slope are
        continued from r_end, removing the clamp kink. The extension spans
        only [0.999, 1.0], i.e. 0.1% of radius.
        """
        f_end = float(np.asarray(fn(r_end), dtype=float))
        slope = (f_end - float(np.asarray(fn(r_end - h), dtype=float))) / h

        def extended(r):
            r = np.asarray(r, dtype=float)
            inside = np.asarray(fn(np.minimum(r, r_end)), dtype=float)
            return np.where(r <= r_end, inside,
                            f_end + slope * (r - r_end))

        return extended

    def _thickness_floor(self, max_thickness):
        """Keep the tip from becoming razor thin (per-chord thickness floor).

        The design's thickness-per-chord ratio falls toward the tip (measured
        1.8% at r/R 0.95 down to 0.4% near the tip), so even with the chord
        tapered the outer band is a near-zero-thickness sheet: bad for
        meshing, and visually the residual "tongue". Per the workflow rule:
        for any blade through this pipeline, the near-tip thickness ratio is
        pulled up toward its value at the 95% station (the taper_start
        station), so the tip closes as a scaled-down airfoil -- a rounded
        tip -- instead of a knife.

        Blend, for r in the band [a, r_tip] with t = (r-a)/(r_tip-a):

            tau_new(r) = max( tau(r),  tau(a) + (tau(r) - tau(a)) (1 - t^3) )

        (1 - t^3) has zero first and second derivatives at t = 0, so the join
        at r = a is C2 (no shading crease), and at the tip tau_new = tau(a):
        the full 95%-station ratio. Absolute thickness still goes to zero
        with the chord, so nothing overhangs. Set thick_floor=False (or
        taper_start=None) to disable.
        """
        if self.taper_start is None or not self.thick_floor:
            return max_thickness
        a = self.taper_start
        tau_a = float(np.asarray(max_thickness(a), dtype=float))

        def floored(r):
            tau = np.asarray(max_thickness(r), dtype=float)
            rr = np.asarray(r, dtype=float)
            t = np.clip((rr - a) / max(1.0 - a, 1e-12), 0.0, 1.0)
            target = tau_a + (tau - tau_a) * (1.0 - t ** 3)
            return np.maximum(tau, target)

        return floored

    def _tapered_chord(self, chord_length):
        """Wrap the design chord so it reaches zero at r/R = 1.

        See TIP_TAPER_START. Returns `chord_length` unchanged when the taper
        is disabled. The taper runs to r/R = 1 rather than to self.r_tip,
        because r_tip is itself found FROM the chord (_find_tip_radius looks
        for where the chord crosses the floor) and must not depend on it.
        """
        if self.taper_start is None:
            return chord_length
        a = self.taper_start

        def tapered(r):
            c = np.asarray(chord_length(r), dtype=float)
            rr = np.asarray(r, dtype=float)
            t = np.clip((rr - a) / max(1.0 - a, 1e-12), 0.0, 1.0)
            return c * np.sqrt(1.0 - t ** 3)

        return tapered

    def __init__(self, max_camber, pitch, chord_length, max_thickness,
                 skew_angle, rake,
                 d=PROP_DIAMETER, r_root=R_ROOT_DEFAULT,
                 tip_close_start=None, close_chord=CLOSE_CHORD,
                 chord_floor=CHORD_FLOOR, thick_floor=True,
                 taper_start=TIP_TAPER_START):
        self.d = float(d)
        # NOTE on the data end. The property curves used to be interpolators
        # over R_VALUES ending at 0.999, which CLAMPED beyond and kinked the
        # surface there (measured 15 deg/step normal turn at r/R 0.9987).
        # x_blade_new.R_VALUES now runs to exactly 1.0 with a cosine-clustered
        # tip band, so the curves cover the whole blade and no extension is
        # needed. _c1_extend is kept for property curves from OTHER sources
        # whose data may still stop short; it is a no-op for r <= its r_end.
        self.MaxCamber = max_camber
        self.Pitch = pitch
        self.ChordLength_raw = chord_length
        self.taper_start = None if taper_start is None else float(taper_start)
        self.thick_floor = bool(thick_floor)
        self.ChordLength = self._tapered_chord(chord_length)
        self.MaxThickness_raw = max_thickness
        self.MaxThickness = self._thickness_floor(max_thickness)
        self.SkewAngle = skew_angle
        self.Rake = rake
        self.r_root = float(r_root)
        self.chord_floor = float(chord_floor)

        # the canonical family must see the SAME floored thickness, since the
        # b() evaluation and the tip closure both build on it
        self.canon = CanonicalSectionFamily(max_camber, self.MaxThickness)


        # Effective tip radius: if the (Bezier-modified) chord crosses the
        # floor before r = 1, the blade closes there naturally and we treat
        # that radius as the tip.
        self.r_tip = self._find_tip_radius()

        # Where the Sec. 6.2.1.4 fold begins. Defined by CHORD, not by a
        # fixed radius: the fold turns whatever chord remains into a
        # zero-thickness sheet, so the only safe place to start it is where
        # the chord has ALREADY collapsed. Every fixed-radius choice broke
        # when the taper changed shape (0.999 was tuned for the cubic taper's
        # 7 mm there; the elliptical taper has 37 mm at the same radius,
        # which regrew the tip sheet fourfold).
        #
        # The band [r_close, r_tip] can be radially tiny without numerical
        # trouble: Eq. 25 maps r = r_tip sin(pi eta/2), so dr/deta -> 0 at
        # the tip and the last sliver of radius occupies a WIDE eta band
        # (chord = 0.002 D lands near eta = 0.995, i.e. 0.5% of parameter
        # space for ~0.02 mm of radius). The folded sheet is then close_chord
        # wide at most: 2.8 mm, the same scale as the blunt TE cap, and is
        # meshed the same way.
        self.close_chord = float(close_chord)
        r_lo = self.taper_start if self.taper_start is not None else 0.9
        self.r_close = self._solve_chord_radius(
            self.close_chord, r_lo * self.r_tip)
        if tip_close_start is not None:
            self.r_close = min(self.r_close, float(tip_close_start))
        self.r_close = min(self.r_close, (1.0 - 1e-6) * self.r_tip)

        # eta mapping (TM 2013-177 Eq. 25 generalized to r_tip):
        # r(eta) = r_tip * sin(pi*eta/2), eta in [eta_min, 1].
        self.eta_min = (2.0 / np.pi) * np.arcsin(self.r_root / self.r_tip)

        # Cache tip-closure reference data (canonical, per Sec. 6.2.1.4).
        self._rc = self.r_close
        self._dr_fd = 1.0e-5

        # z-rotation applied by para.py at the end of section construction.
        self._rotz = rot_axis(np.array([0.0, 0.0, 1.0]), np.pi)

    def _solve_chord_radius(self, target, r_lo, iters=80):
        """Largest radius at which the (tapered) chord still equals `target`
        (per diameter). Chord decreases monotonically over the taper band, so
        bisection between r_lo and r_tip is safe."""
        lo, hi = float(r_lo), float(self.r_tip)

        def c(r):
            return float(np.asarray(self.ChordLength(r), dtype=float))

        if c(lo) <= target:
            return lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if c(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -- radius / eta mapping -------------------------------------------------
    def _find_tip_radius(self):
        rr = np.linspace(0.9, 1.0, 201)
        ch = np.asarray(self.ChordLength(rr), dtype=float)
        below = np.where(ch <= self.chord_floor)[0]
        if below.size == 0:
            return 1.0
        i = below[0]
        if i == 0:
            return float(rr[0])
        # linear interpolation to the floor crossing
        r0, r1 = rr[i - 1], rr[i]
        c0, c1 = ch[i - 1], ch[i]
        return float(r0 + (self.chord_floor - c0) * (r1 - r0) / (c1 - c0))

    def r_of_eta(self, eta):
        return self.r_tip * np.sin(0.5 * np.pi * np.asarray(eta, dtype=float))

    def eta_of_r(self, r):
        return (2.0 / np.pi) * np.arcsin(
            np.clip(np.asarray(r, dtype=float) / self.r_tip, -1.0, 1.0))

    def eta_range(self):
        return self.eta_min, 1.0

    # -- canonical section with tip closure ----------------------------------
    def _closure_constants(self):
        """Scalar ingredients of the tip closure (cached)."""
        if not hasattr(self, "_cc"):
            rc, rt, h = self._rc, self.r_tip, self._dr_fd
            mc = self.MaxCamber
            mt = self.MaxThickness
            self._cc = dict(
                mc0=float(mc(rc)), mt0=float(mt(rc)),
                mc1=float(mc(rt)), mt1=float(mt(rt)),
                dmc0=float((mc(rc + h) - mc(rc - h)) / (2 * h)),
                dmt0=float((mt(rc + h) - mt(rc - h)) / (2 * h)),
            )
        return self._cc

    def _canonical(self, xi, r):
        """Canonical section (xs, ys) with the DRDC tip closure applied on
        [r_close, r_tip] (TM 2013-177 Sec. 6.2.1.4, Hermite variant, Eq. 28).

        Because ys is linear in (MaxCamber(r), MaxThickness(r)) and F_cam /
        F_th are mirror-symmetric / antisymmetric, the mirror-average tip
        value and the Eq. 28 slope have the closed forms used below; the
        closure therefore costs no extra surface evaluations.
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        xi, r = np.broadcast_arrays(xi, r)
        s, F_cam, F_th = self.canon.basis(xi)
        s = np.broadcast_to(s, np.broadcast_shapes(s.shape, r.shape)).copy()
        F_cam = np.broadcast_to(F_cam, s.shape)
        F_th = np.broadcast_to(F_th, s.shape)

        ys = np.asarray(F_cam * self.MaxCamber(r)
                        + F_th * self.MaxThickness(r), dtype=float)

        mask = r > self._rc
        if np.any(mask):
            # ROUNDED tip closure. The original combined Hermite (Eq. 28
            # applied to camber + thickness together) sends the thickness to
            # zero with FINITE slope: a wedge, i.e. a knife edge, which under
            # shading is a visible crease-and-lobe at the tip. TM 2013-177
            # Sec. 6.2.2 closes edges "dull" instead: a semi-elliptical
            # cross-section with vertical tangent (Eq. 29, rc = d/2 sin(beta)).
            # The same principle is applied here in the RADIAL direction:
            #
            #   camber   : cubic Hermite to the mean line with the Eq. 28
            #              slope (unchanged, the doc's parabolic path)
            #   thickness: MaxThickness(r) * sqrt(1 - u^3)  -> vanishes with
            #              vertical tangent in r = rounded edge. Eq. 25 makes
            #              the eta-derivative finite (same cancellation as
            #              the elliptical chord taper).
            #
            # C1 at r_close: sqrt(1 - u^3) has value 1 and zero slope at
            # u = 0, so both value and radial slope match the open surface.
            # At u = 1 the thickness term is exactly zero, so
            # b(xi, 1) = b(1 - xi, 1) still holds exactly.
            c = self._closure_constants()
            rc, rt = self._rc, self.r_tip
            dr = rt - rc
            y0 = F_cam[mask] * c["mc0"]                                # camber
            s0 = F_cam[mask] * c["dmc0"]
            y1 = F_cam[mask] * c["mc1"]                                # mean line
            s1 = 2.0 * (y1 - y0) / dr - s0                             # Eq. 28
            u = np.clip((r[mask] - rc) / dr, 0.0, 1.0)
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            ys_cam = h00 * y0 + h10 * dr * s0 + h01 * y1 + h11 * dr * s1
            g = np.sqrt(np.clip(1.0 - u**3, 0.0, None))
            ys_m = ys_cam + F_th[mask] * np.asarray(
                self.MaxThickness(r[mask]), dtype=float) * g
            ys = ys.copy()
            ys[mask] = ys_m
        return np.stack([s, ys], axis=-1)

    # -- the surface ----------------------------------------------------------
    def b(self, xi, eta):
        """Point(s) on the blade. xi periodic; eta in [eta_min, 1].

        Returns (..., 3) array in the same coordinates/units as para.py
        output (metres, z-rotated by pi like the legacy points).
        """
        xi = np.mod(np.asarray(xi, dtype=float), 1.0)
        eta = np.asarray(eta, dtype=float)
        xi, eta = np.broadcast_arrays(xi, eta)
        scalar_in = xi.ndim == 0
        xi = np.atleast_1d(xi)
        eta = np.atleast_1d(eta)
        r = self.r_of_eta(eta)

        canon = self._canonical(xi, r)                  # (..., 2) chord units
        d = self.d
        chord = np.maximum(np.asarray(self.ChordLength(r), dtype=float),
                           self.chord_floor) * d
        x_c = canon[..., 0] * chord
        y_c = canon[..., 1] * chord

        pitch_dia = np.asarray(self.Pitch(r), dtype=float) * d
        phi = np.arctan2(pitch_dia, 2.0 * np.pi * r * (d / 2.0))
        skew_deg = np.asarray(self.SkewAngle(r), dtype=float)
        rake = np.asarray(self.Rake(r), dtype=float) * d
        Rr = r * (d / 2.0)

        # Exactly para.py's mapping (TM 2013-177 Eqs. 23-24 in expanded form).
        xp = (-rake + Rr * np.deg2rad(skew_deg) * np.tan(phi)
              + (0.5 * chord - x_c) * np.sin(phi) + y_c * np.cos(phi))
        ang = np.deg2rad(
            skew_deg - (180.0 * ((0.5 * chord - x_c) * np.cos(phi)
                                 - y_c * np.sin(phi))) / (np.pi * Rr))
        yp = Rr * np.sin(ang)
        zp = Rr * np.cos(ang)

        pts = np.stack([xp, yp, zp], axis=-1) @ self._rotz
        return pts[0] if scalar_in else pts

    def b_xi(self, xi, eta, h=1.0e-6):
        return (self.b(np.asarray(xi) + h, eta) -
                self.b(np.asarray(xi) - h, eta)) / (2.0 * h)

    def b_eta(self, xi, eta, h=1.0e-6):
        eta = np.asarray(eta, dtype=float)
        lo, hi = self.eta_min, 1.0
        e_hi = np.minimum(eta + h, hi)
        e_lo = np.maximum(eta - h, lo)
        return (self.b(xi, e_hi) - self.b(xi, e_lo)) / (e_hi - e_lo)[..., None]

    def normal(self, xi, eta):
        n = np.cross(self.b_xi(xi, eta), self.b_eta(xi, eta))
        mag = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.maximum(mag, 1.0e-300)

    # -- convenience ----------------------------------------------------------
    def edge_point(self, edge, eta):
        """Point on the leading ('le') or trailing ('te') edge at eta."""
        xi = 0.5 if edge == "le" else 0.0
        return self.b(xi, eta)

    def tip_line(self, n=101):
        """The closed-tip mean line b(xi, 1), xi in [0, 0.5]."""
        xi = np.linspace(0.0, 0.5, n)
        return self.b(xi, np.ones_like(xi))


def blade_surface_from_functions(Pitch, ChordLength, MaxCamber=None,
                                 MaxThickness=None, SkewAngle=None, Rake=None):
    """Build a BladeSurface using the fixed design polynomials from
    x_blade_new.py for any property not supplied."""
    from x_blade_new import (BASE_MAX_CAMBER, BASE_MAX_THICKNESS,
                             BASE_SKEW_ANGLE, BASE_RAKE)
    return BladeSurface(
        MaxCamber or BASE_MAX_CAMBER,
        Pitch,
        ChordLength,
        MaxThickness or BASE_MAX_THICKNESS,
        SkewAngle or BASE_SKEW_ANGLE,
        Rake or BASE_RAKE,
    )
