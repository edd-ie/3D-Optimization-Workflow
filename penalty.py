"""
This code returns the values by how much the particle violates the constraints.
                                                Irtiza Khan
                                                Date started: September 26, 2025
"""

import numpy as np
from scipy.spatial import ConvexHull


def penalty_check(pop, search_bounds):
    """
    Return the values by how much the particle violates the constraints.
    """
    # Precompute convex hull once
 
    num_var = search_bounds.shape[0]
    pop_length = pop.shape[0]
    penalty_val = np.zeros(pop_length)

    for i in range(pop_length):
        var = pop[i]
        for j in range(num_var):
            if var[j] < search_bounds[j, 0] or var[j] > search_bounds[j, 1]:
                penalty_val[i] += max(0, (var[j] - search_bounds[j, 0])) + max(0, (search_bounds[j, 1] - var[j]))

    return penalty_val

   