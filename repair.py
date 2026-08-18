"""
This code repairs the particle if it violates the constraints.
                                                Irtiza Khan
                                                Date started: September 26, 2025
"""

import numpy as np
from scipy.spatial import ConvexHull

def repair(pop, search_bounds):
    
    num_var = search_bounds.shape[0]
    pop_length = pop.shape[0]
    
    new_pop = search_bounds[:, 0] + (search_bounds[:, 1] - search_bounds[:, 0]) * np.random.rand(pop_length, num_var)
            
    return new_pop
