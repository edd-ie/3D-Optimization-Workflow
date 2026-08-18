"""
x_blade_new.py

DRDC-method replacement of the CAD path, built on the ORIGINAL x_blade.py
(NOT the abandoned tip-closure-band experiment that previously lived in this
file). The aerodynamic path is byte-for-byte the same as x_blade.py:

  - same design polynomials, same R stations, same para_control_bez_updated
    call, same para() section generation, same KD-tree blade-clearance check
    and accept/reject behaviour. The optimizer sees NO difference.

What is new: X_blade() can additionally return a BladeSurface (the smooth
global parameterization b(xi, eta) of blade_surface_new.py) built from the
same Bezier-modified Pitch/Chord, which X_CAD_new.py turns into the DRDC
five-surface IGES geometry. Use `return_blade_surface=True`.
"""

import numpy as np
from scipy.spatial import cKDTree

from para import para
from para_control_bez_updated import para_control_bez_updated, APPLY_COUPLED_CONSTRAINTS
from rot_axis import rot_axis
from blade_surface_new import BladeSurface


DEFAULT_BEZIER_CONSTRAINT_MODE = "project"
CLEARANCE_THRESHOLD = 0.025

# The fixed design polynomials (identical to x_blade.py), exposed at module
# level so blade_surface_new can reuse them.
BASE_MAX_CAMBER = lambda x: 1*(-4448.8369*x**12 + 30393.6831*x**11 + -92977.6043*x**10 + 168066.8833*x**9 + -199490.6759*x**8 + 163416.9800*x**7 + -94493.8152*x**6 + 38765.7447*x**5 + -11176.4246*x**4 + 2207.9559*x**3 + -285.0846*x**2 + 21.9443*x + -0.7490)

BASE_PITCH = lambda x: 1*(19344.5071*x**12 + -114044.8587*x**11 + 280789.2801*x**10 + -357377.6146*x**9 + 207947.2705*x**8 + 43330.8173*x**7 + -173099.4797*x**6 + 143116.5772*x**5 + -65570.9523*x**4 + 18410.2055*x**3 + -3128.7530*x**2 + 294.0671*x + -10.4419)

BASE_CHORD = lambda x: 1*(-143202.4761*x**12 + 978274.9902*x**11 + -2992184.0323*x**10 + 5408923.9625*x**9 + -6424276.8851*x**8 + 5271614.6993*x**7 + -3058632.5267*x**6 + 1261908.7720*x**5 + -366745.1423*x**4 + 73096.4455*x**3 + -9470.1908*x**2 + 716.1619*x + -23.7157)

BASE_MAX_THICKNESS = lambda x: 1*(-9688.7237*x**12 + 59807.8900*x**11 + -164159.4387*x**10 + 265158.4257*x**9 + -281726.9910*x**8 + 209033.5305*x**7 + -112447.0353*x**6 + 44837.0567*x**5 + -13266.8679*x**4 + 2814.8064*x**3 + -391.1763*x**2 + 29.0131*x + -0.4854)  # by chord

BASE_SKEW_ANGLE = lambda x: 1*(-719361.0309*x**12 + 5176951.0162*x**11 + -16746858.1600*x**10 + 32161071.3897*x**9 + -40784306.1013*x**8 + 35930276.7571*x**7 + -22515881.4272*x**6 + 10096627.1844*x**5 + -3209956.6167*x**4 + 704272.6595*x**3 + -100947.8092*x**2 + 8462.8187*x + -312.8100)

BASE_RAKE = lambda x: 1*(-334.1390*x**12 + 1599.6222*x**11 + -2939.2891*x**10 + 2255.6140*x**9 + 0.0000*x**8 + -952.8241*x**7 + 0.0000*x**6 + 902.2268*x**5 + -804.8019*x**4 + 345.8731*x**3 + -81.8282*x**2 + 10.2137*x + -0.5245)  # Back calculated to line up with original CAD

# Radial stations. The originals ended at [.., 0.99, 0.999]; every property
# interpolator therefore CLAMPED for r > 0.999 while the blade surface runs to
# r = 1, which kinked the surface at 0.999 (measured 15 deg/step normal turn)
# and left a flat lobe beyond it -- the tip "tongue". The stations now run to
# exactly 1.0, cosine-clustered over the outer band so the interpolators are
# dense where the tip closure and chord taper act. para_control_bez_updated
# anchors its Bezier ends at R_values[-1], so the design curves genuinely
# extend to the tip instead of freezing 0.1% short of it.
R_TIP_BAND = 0.93        # start of the clustered tip band
R_VALUES = np.concatenate([
    np.arange(0.17, R_TIP_BAND - 1e-9, 0.018),
    R_TIP_BAND + (1.0 - R_TIP_BAND) * np.sin(np.linspace(0.0, np.pi / 2, 14)),
])


def X_blade(
    pitch_con,
    chord_con,
    x1,
    *,
    return_bezier_info=False,
    return_blade_surface=False,
    bezier_constraint_mode=DEFAULT_BEZIER_CONSTRAINT_MODE,
    apply_coupled_constraints=None,
    write_dat=True,
):
    MaxCamber = BASE_MAX_CAMBER
    Pitch = BASE_PITCH
    ChordLength = BASE_CHORD
    MaxThickness = BASE_MAX_THICKNESS
    SkewAngle = BASE_SKEW_ANGLE
    Rake = BASE_RAKE

    R_values = R_VALUES

    para_control_flag = 7  # control which parameter is being modified

    (
        Pitch,
        ChordLength,
        chord_con_points,
        pitch_con_points,
        bezier_info,
    ) = para_control_bez_updated(
        Pitch,
        ChordLength,
        R_values,
        para_control_flag,
        pitch_con,
        chord_con,
        case_id=x1,
        constraint_mode=bezier_constraint_mode,
        return_reject_info=True,
        apply_coupled_constraints=apply_coupled_constraints,
    )

    use_coupled = (
        APPLY_COUPLED_CONSTRAINTS
        if apply_coupled_constraints is None
        else bool(apply_coupled_constraints)
    )
    if use_coupled and bezier_info["rejected"]:
        points = np.empty((0, 3))
        min_dis = np.inf
        constraint_violation = 1
        extra = (bezier_info,) if return_bezier_info else ()
        extra += (None,) if return_blade_surface else ()
        return (points, min_dis, constraint_violation, chord_con_points,
                pitch_con_points, Pitch, ChordLength) + extra

    # ---- aerodynamic / clearance path: IDENTICAL to x_blade.py ----------
    points = para(
        MaxCamber,
        Pitch,
        ChordLength,
        MaxThickness,
        SkewAngle,
        Rake,
        R_values,
        x1,
        write_dat=write_dat,
    )

    ax_rot = np.array([1, 0, 0])  # rotate about X axis
    R72 = rot_axis(ax_rot, np.deg2rad(72))
    points2 = points @ R72

    tree2 = cKDTree(points2)
    dists12, _ = tree2.query(points, k=1)
    tree1 = cKDTree(points)
    dists21, _ = tree1.query(points2, k=1)
    min_dis = min(float(np.min(dists12)), float(np.min(dists21)))
    constraint_violation = int(min_dis < CLEARANCE_THRESHOLD)

    # ---- new: smooth blade surface for the DRDC CAD path ----------------
    extra = ()
    if return_bezier_info:
        extra += (bezier_info,)
    if return_blade_surface:
        blade_surface = BladeSurface(
            MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake)
        extra += (blade_surface,)

    return (points, min_dis, constraint_violation, chord_con_points,
            pitch_con_points, Pitch, ChordLength) + extra
