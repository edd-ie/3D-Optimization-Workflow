import numpy as np
from scipy.stats import qmc



def bound_check(seed, search_bounds, dim, batch_size):
    """
    Return at least N_samples Latin-hypercube candidates inside a convex polyhedron.

    Draw in batches and accumulate accepted points until the target count is reached.
    """
    rng = np.random.RandomState(seed)
    sampler = qmc.LatinHypercube(d=dim, seed=rng.randint(0, 1_000_000), optimization="random-cd")
    U = sampler.random(batch_size)
    X_batch = qmc.scale(U, search_bounds[:, 0], search_bounds[:, 1])

    return X_batch