import numpy as np

def skew_symmetric_matrix(v):
    """
    Create a skew-symmetric matrix from a 3D vector
    """
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]]) 