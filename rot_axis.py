import numpy as np
from skew_symmetric_matrix import skew_symmetric_matrix

def rot_axis(ax, angle):
    """
    Create rotation matrix around arbitrary axis using Rodrigues' rotation formula
    """
    ax = ax / np.linalg.norm(ax)
    result = (np.eye(3) * np.cos(angle) + 
              (1 - np.cos(angle)) * np.outer(ax, ax) + 
              np.sin(angle) * skew_symmetric_matrix(ax))
    return result 