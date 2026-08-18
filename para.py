import numpy as np
import pandas as pd
from pathlib import Path
from rot_axis import rot_axis


ROOT = Path(__file__).resolve().parent
AIRFOIL_DATA_PATH = ROOT / 'airfoil_data_fixed.csv'
GEOMETRY_OUTPUT_DIR = ROOT


def para(MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake, R_values, x1, write_dat=True, section_scale=None):
    """
    Python equivalent of para.m function

    section_scale : array_like or None
        Optional per-section uniform scale factor (one value per radial station).
        A value of 1.0 leaves the profile unchanged; values < 1.0 shrink the whole
        cross-section toward its reference point, which is used to taper the
        appended tip-closure sections into a faired, closed tip. Defaults to all
        ones (no scaling) for backward compatibility.
    """
    # Scaling factor for the geometry (diameter of the propeller hub or blade)
    d = 1.4
    
    # Evaluate the anonymous functions at R_values
    max_camber_at_R = MaxCamber(R_values)
    pitch_dia_at_R = Pitch(R_values) * d
    chord_length_at_R = ChordLength(R_values)
    max_thickness_at_R = MaxThickness(R_values)
    skew_angle_at_R = SkewAngle(R_values)
    pitch = np.arctan2(pitch_dia_at_R, (2 * np.pi * R_values * (d / 2)))
    Rake_at_R = Rake(R_values) * d
    skew_angle_at_R[0] = 0

    

    # ---- Airfoil Generation Using the Fitted Polynomials ----
    
    # Load airfoil data from external file
    Airfoil = pd.read_csv(AIRFOIL_DATA_PATH, skiprows=1, header=None).values
    X_c = np.append(Airfoil[:, 0], 1.005)  # Chord coordinates (normalized)
    y_c = np.append(Airfoil[:, 1], 0)      # Camber distribution
    der_y = np.append(Airfoil[:, 2], 0)    # Slope of camber
    th_d = np.append(Airfoil[:, 3], 0)     # Thickness distribution

    # Normalized camber distribution for the reference airfoil

    cam_d = y_c / np.max(y_c)

    # Number of sections (airfoils)
    n = len(R_values)

    if section_scale is None:
        section_scale = np.ones(n)
    else:
        section_scale = np.asarray(section_scale, dtype=float).ravel()
        if section_scale.shape[0] != n:
            raise ValueError(
                f"section_scale must have one entry per section ({n}), got {section_scale.shape[0]}"
            )

    # Use the fitted polynomial values at the specific R values to modify the airfoil shape
    chord_len = chord_length_at_R * d  # Scale chord length by diameter
    # Uniform per-section taper. Because camber, thickness and the chordwise
    # stations are all derived from chord_len, scaling it shrinks the entire
    # section toward its reference point (used to close the tip).
    chord_len = chord_len * section_scale
    max_c = max_camber_at_R * chord_len  # Max camber based on chord length
    max_th = max_thickness_at_R * chord_len  # Max thickness based on chord length

    # Loop over the number of sections (airfoil instances) and generate geometry
    points = np.empty((0, 3))

    for i in range(n):
        # Creating the airfoils
        camber = cam_d * max_c[i]  # Camber distribution scaling
        th = th_d * max_th[i]      # Thickness distribution scaling

        x_c = X_c * chord_len[i]

        # Calculate upper and lower surface points of the airfoil
        theta1 = np.arctan2(der_y, 1)  # Angle of the camber line slope
        x_u = x_c - th * np.sin(theta1)  # Upper surface (x-coordinates)
        y_u_L = camber + th * np.cos(theta1)  # Upper surface (y-coordinates)

        x_l = x_c + th * np.sin(theta1)  # Lower surface (x-coordinates)
        y_b_L = camber - th * np.cos(theta1)  # Lower surface (y-coordinates)

        x_u_p = np.zeros(len(x_c))
        y_u_p = np.zeros(len(x_c))
        z_u_p = np.zeros(len(x_c))

        x_b_p = np.zeros(len(x_c))
        y_b_p = np.zeros(len(x_c))
        z_b_p = np.zeros(len(x_c))

        for w in range(len(x_c)):
            # Upper surface
            # Calculate xp
            x_u_p[w] = (-Rake_at_R[i] + 
                       (R_values[i] * (d / 2)) * np.deg2rad(skew_angle_at_R[i]) * np.tan(pitch[i]) + 
                       (0.5 * chord_len[i] - x_c[w]) * np.sin(pitch[i]) + 
                       y_u_L[w] * np.cos(pitch[i]))
            
            # Calculate yp
            y_u_p[w] = (R_values[i] * (d / 2) * 
                       np.sin(np.deg2rad(skew_angle_at_R[i] - 
                             (180 * ((0.5 * chord_len[i] - x_c[w]) * np.cos(pitch[i]) - 
                                    y_u_L[w] * np.sin(pitch[i]))) / 
                             (np.pi * R_values[i] * (d / 2)))))
            
            # Calculate zp
            z_u_p[w] = (R_values[i] * (d / 2) * 
                       np.cos(np.deg2rad(skew_angle_at_R[i] - 
                             (180 * ((0.5 * chord_len[i] - x_c[w]) * np.cos(pitch[i]) - 
                                    y_u_L[w] * np.sin(pitch[i]))) / 
                             (np.pi * R_values[i] * (d / 2)))))

            # Lower surface
            # Calculate xp
            x_b_p[w] = (-Rake_at_R[i] + 
                       (R_values[i] * (d / 2)) * np.deg2rad(skew_angle_at_R[i]) * np.tan(pitch[i]) + 
                       (0.5 * chord_len[i] - x_c[w]) * np.sin(pitch[i]) + 
                       y_b_L[w] * np.cos(pitch[i]))
            
            # Calculate yp
            y_b_p[w] = (R_values[i] * (d / 2) * 
                       np.sin(np.deg2rad(skew_angle_at_R[i] - 
                             (180 * ((0.5 * chord_len[i] - x_c[w]) * np.cos(pitch[i]) - 
                                    y_b_L[w] * np.sin(pitch[i]))) / 
                             (np.pi * R_values[i] * (d / 2)))))
            
            # Calculate zp
            z_b_p[w] = (R_values[i] * (d / 2) * 
                       np.cos(np.deg2rad(skew_angle_at_R[i] - 
                             (180 * ((0.5 * chord_len[i] - x_c[w]) * np.cos(pitch[i]) - 
                                    y_b_L[w] * np.sin(pitch[i]))) / 
                             (np.pi * R_values[i] * (d / 2)))))

       
        ax3 = np.array([0, 0, 1])

        rotation_matrix = rot_axis(ax3, np.pi)

        # Store points for the airfoil geometry
        Airfoil_X_Chord1 = np.concatenate([x_u_p[:26], np.flip(x_b_p[:27])])
        Airfoil_Y_Chord1 = np.concatenate([y_u_p[:26], np.flip(y_b_p[:27])])
        Airfoil_Z_Chord1 = np.concatenate([z_u_p[:26], np.flip(z_b_p[:27])])

        pts1 = np.column_stack([Airfoil_X_Chord1, Airfoil_Y_Chord1, Airfoil_Z_Chord1]) @ rotation_matrix

        points = np.vstack([points, pts1])

    if write_dat:
        GEOMETRY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = GEOMETRY_OUTPUT_DIR / f'ORCA{x1}.dat'
        with filename.open('w') as fileID:
            for point in points:
                fileID.write(f'{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n')
    return points

     