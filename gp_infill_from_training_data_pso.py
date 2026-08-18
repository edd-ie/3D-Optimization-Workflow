"""
PSO-backed variant of gp_infill_from_training_data.

This keeps the original workflow intact and swaps only the optimizer path
from ESPSOLS to pso_function for GP acquisition optimization.
"""

import gp_infill_from_training_data as base
from PSO_function import pso_function


def pso_opt(gp, search_bounds, y_best, opt, swarm_size, num_iter, r):
    """Drop-in replacement for the ESPSOLS optimization hook."""
    X_phys, scores, _ = pso_function(
        gp=gp,
        num_var=search_bounds.shape[0],
        bounds=search_bounds,
        y_best=float(y_best),
        swarm_size=int(swarm_size),
        num_iter=int(num_iter),
        r=r,
        opt=str(opt),
        type_of_Problem="max",
    )
    return X_phys, scores


# Replace only the optimizer hook; keep all remaining pipeline logic unchanged.
base.espsols_opt = pso_opt


if __name__ == "__main__":
    base.main()
