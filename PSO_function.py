"""
Basic PSO optimizer for GP acquisition optimization.

This module intentionally contains only the functionality required by
`gp_infill_from_training_data_pso.py` to avoid unrelated dependencies.
"""

import numpy as np
from scipy.stats import norm


def _acquisition_scores(gp, pop_u, y_best, opt, type_of_Problem):
    """Evaluate acquisition on unit-space points and return maximization scores."""
    pop_u = np.asarray(pop_u, dtype=float)
    opt_name = str(opt)
    problem_type = str(type_of_Problem).lower()

    if opt_name == "MaxMSE":
        _, sd = gp.predict(pop_u, return_std=True)
        return np.asarray(sd, dtype=float) ** 2

    if opt_name == "EI":
        mu, sd = gp.predict(pop_u, return_std=True)
        mu = np.asarray(mu, dtype=float)
        sigma = np.maximum(np.asarray(sd, dtype=float), 1e-12)
        if problem_type == "min":
            z = (float(y_best) - mu) / sigma
            return (float(y_best) - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        z = (mu - float(y_best)) / sigma
        return (mu - float(y_best)) * norm.cdf(z) + sigma * norm.pdf(z)

    mu = np.asarray(gp.predict(pop_u), dtype=float).ravel()
    if problem_type == "min":
        return -mu
    return mu


def pso_function(gp, num_var, bounds, y_best, swarm_size, num_iter, r, opt, type_of_Problem):
    """
    Basic PSO for GP acquisition with ESPSOLS-compatible signature.

    Returns:
        X_phys: candidate points in physical space (sorted best->worst)
        scores: acquisition scores for X_phys (higher is better)
        pop_phys: final swarm positions in physical space
    """
    _ = r  # kept for compatibility with ESPSOLS call sites

    d = int(num_var)
    n = int(swarm_size)
    iters = int(num_iter)
    bounds = np.asarray(bounds, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = np.maximum(hi - lo, 1e-12)

    def from_unit(u):
        u = np.asarray(u, dtype=float)
        return lo + u * span

    # Simple PSO in unit space [0, 1]^d
    pop = np.random.rand(n, d)
    vel = np.random.uniform(-0.2, 0.2, size=(n, d))

    c1 = 2.05
    c2 = 2.05
    phi = c1 + c2
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))  # constriction factor

    pbest_pos = pop.copy()
    pbest_scores = _acquisition_scores(
        gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem
    )

    g_idx = int(np.argmax(pbest_scores))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_score = float(pbest_scores[g_idx])

    for t in range(iters):
        if np.isclose(chi, 1.0):
            w = 0.9 - (0.5 * (float(t) / max(float(iters), 1.0)))  # 0.9 -> 0.4
        else:
            w = 1.0
        r1 = np.random.rand(n, d)
        r2 = np.random.rand(n, d)

        vel = chi * (w * vel + c1 * r1 * (pbest_pos - pop) + c2 * r2 * (gbest_pos - pop))
        vel = np.clip(vel, -0.3, 0.3)
        pop = np.clip(pop + vel, 0.0, 1.0)

        scores = _acquisition_scores(
            gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem
        )
        improve = scores > pbest_scores
        if np.any(improve):
            pbest_scores[improve] = scores[improve]
            pbest_pos[improve] = pop[improve].copy()

            best_idx = int(np.argmax(pbest_scores))
            if float(pbest_scores[best_idx]) > gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest_pos = pbest_pos[best_idx].copy()

    # Return ranked candidates (best first), ESPSOLS-like for top-k collector.
    order = np.argsort(pbest_scores)[::-1]
    ranked_u = pbest_pos[order]
    ranked_scores = np.asarray(pbest_scores[order], dtype=float)

    return from_unit(ranked_u), ranked_scores, from_unit(pop)
"""
Basic PSO optimizer for GP acquisition optimization.

This module intentionally contains only the functionality required by
`gp_infill_from_training_data_pso.py` to avoid unrelated dependencies.
"""

import numpy as np
from scipy.stats import norm


def _acquisition_scores(gp, pop_u, y_best, opt, type_of_Problem):
    """Evaluate acquisition on unit-space points and return maximization scores."""
    pop_u = np.asarray(pop_u, dtype=float)
    opt_name = str(opt)
    problem_type = str(type_of_Problem).lower()

    if opt_name == "MaxMSE":
        _, sd = gp.predict(pop_u, return_std=True)
        return np.asarray(sd, dtype=float) ** 2

    if opt_name == "EI":
        mu, sd = gp.predict(pop_u, return_std=True)
        mu = np.asarray(mu, dtype=float)
        sigma = np.maximum(np.asarray(sd, dtype=float), 1e-12)
        if problem_type == "min":
            z = (float(y_best) - mu) / sigma
            return (float(y_best) - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        z = (mu - float(y_best)) / sigma
        return (mu - float(y_best)) * norm.cdf(z) + sigma * norm.pdf(z)

    mu = np.asarray(gp.predict(pop_u), dtype=float).ravel()
    if problem_type == "min":
        return -mu
    return mu


def pso_function(gp, num_var, bounds, y_best, swarm_size, num_iter, r, opt, type_of_Problem):
    """
    Basic PSO for GP acquisition with ESPSOLS-compatible signature.

    Returns:
        X_phys: candidate points in physical space (sorted best->worst)
        scores: acquisition scores for X_phys (higher is better)
        pop_phys: final swarm positions in physical space
    """
    _ = r  # kept for compatibility with ESPSOLS call sites

    d = int(num_var)
    n = int(swarm_size)
    iters = int(num_iter)
    bounds = np.asarray(bounds, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = np.maximum(hi - lo, 1e-12)

    def from_unit(u):
        u = np.asarray(u, dtype=float)
        return lo + u * span

    # Simple PSO in unit space [0, 1]^d
    pop = np.random.rand(n, d)
    vel = np.random.uniform(-0.2, 0.2, size=(n, d))

    c1 = 2.05
    c2 = 2.05
    phi = c1 + c2
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))  # constriction factor

    pbest_pos = pop.copy()
    pbest_scores = _acquisition_scores(
        gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem
    )

    g_idx = int(np.argmax(pbest_scores))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_score = float(pbest_scores[g_idx])

    for t in range(iters):
        if np.isclose(chi, 1.0):
            w = 0.9 - (0.5 * (float(t) / max(float(iters), 1.0)))  # 0.9 -> 0.4
        else:
            w = 1.0
        r1 = np.random.rand(n, d)
        r2 = np.random.rand(n, d)

        vel = chi * (w * vel + c1 * r1 * (pbest_pos - pop) + c2 * r2 * (gbest_pos - pop))
        vel = np.clip(vel, -0.3, 0.3)
        pop = np.clip(pop + vel, 0.0, 1.0)

        scores = _acquisition_scores(
            gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem
        )
        improve = scores > pbest_scores
        if np.any(improve):
            pbest_scores[improve] = scores[improve]
            pbest_pos[improve] = pop[improve]

            best_idx = int(np.argmax(pbest_scores))
            if float(pbest_scores[best_idx]) > gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest_pos = pbest_pos[best_idx].copy()

    # Return ranked candidates (best first), ESPSOLS-like for top-k collector.
    order = np.argsort(pbest_scores)[::-1]
    ranked_u = pbest_pos[order]
    ranked_scores = np.asarray(pbest_scores[order], dtype=float)

    return from_unit(ranked_u), ranked_scores, from_unit(pop)
"""
This is the implementation of the PSO optimizer in the function form. 
This has the same structure as the basic_pso.m  
                                                Irtiza Khan
                                                Date started: May 05, 2025
"""

import numpy as np
from scipy.stats import norm

try:
    from get_test_problem import get_test_problem
except ModuleNotFoundError:
    get_test_problem = None

def PSO_function(swarm_size, num_iter, C1, C2, PSO_variant, w_max, w_min,
                 Particle_Gen_method, spacing_per_dim, Contraint_handling, craziness, 
                 penalty_factor, problem_id):
    """
    PSO optimizer function
    
    Returns:
        gbest: global best value
        gbest_pos: global best position
        iterations: number of iterations completed
        gbest_track: tracking of global best over iterations
        avg_fitness_per_iter: average fitness per iteration
    """
    
    if get_test_problem is None:
        raise ModuleNotFoundError(
            "Legacy PSO_function requires optional test-problem dependencies. "
            "Use pso_function(...) for GP infill optimization."
        )
    obj_func, num_var, bounds, constrain_func, constrain_bound, best_known_sol, best_known_val = get_test_problem(problem_id)

    swarm_size_eq_space = spacing_per_dim**num_var
    C = C1 + C2

    # Initialize velocity, constriction factor, & intertia 
    Vmax = 0.2 * (bounds[:, 1] - bounds[:, 0])          # shape: (num_var,)
    Vmax_col = Vmax[:, None]                             # shape: (num_var,1)
    Vmax_1d = Vmax                                       # for clamping per-dimension
    if PSO_variant == 1:                                                        # with the constriction factor effect                                                                
        X = 2/(C-2+np.sqrt(C**2-4*C))                                          # contriction factor
        w = 1                                                                   # inertia

        # Initialize random velocities within variable limits
        vel = -Vmax_col + 2*Vmax_col * np.random.rand(num_var, swarm_size)      # velocity initialized with range a of 20% of the variable dynamic range

    elif PSO_variant == 2:                                                      # without the constriction factor effect 
        X = 1
        vel = -Vmax_col + 2*Vmax_col * np.random.rand(num_var, swarm_size) 
      
    # Initialize random position within variable limits
    if Particle_Gen_method == 1:
        lower_col = bounds[:, 0][:, None]
        range_col = (bounds[:, 1] - bounds[:, 0])[:, None]
        pos = lower_col + range_col * np.random.rand(num_var, swarm_size)

    # distribute particles equally on the each dimension 
    # (only set up for 2 dimnesions for testing purposes) 
    # elif Particle_Gen_method==2:
    #     pos=zeros(num_var,swarm_size_eq_space);
    #     for i=1:spacing_per_dim
    #         increment=(bounds(1,2)-bounds(1,1))/(spacing_per_dim-1);
    #         pos(:,spacing_per_dim*(i-1)+1:spacing_per_dim*i)=...
    #             [ones(1,spacing_per_dim)*(bounds(1,1)+increment*(i-1));...
    #             bounds(2,1)+(bounds(2,2)-bounds(2,1)).* ...
    #             (ones(1,spacing_per_dim).*(0:1/(spacing_per_dim-1):1))];
    #     end
    elif Particle_Gen_method == 2:
        # number of points per dimension
        m = spacing_per_dim
        # precompute 1×m vectors for each var
        gridAxes = []
        for d in range(num_var):
            gridAxes.append(np.linspace(bounds[d, 0], bounds[d, 1], m))
        
        # build the full n-D grid
        grid = np.meshgrid(*gridAxes, indexing='ij')
        
        # flatten into pos: each column is one particle
        swarm_size_eq_space = m**num_var
        pos = np.zeros((num_var, swarm_size_eq_space))
        for d in range(num_var):
            pos[d, :] = grid[d].flatten()

    new_vel = vel.copy()                                                        # set the randomly generated velocity to new velocity
    t = 1                                                                       # iteration

    # Generate data containers
    pbest = np.zeros(swarm_size)                                                # container for pbest values
    pbest_pos = np.zeros((num_var, swarm_size))                                # container for pbest position
    gbest_track = np.zeros(num_iter)                                            # track the global best
    avg_fitness_per_iter = np.zeros(num_iter)                                   # track the average func. val. per iter.

    # Main loop
    while t <= num_iter:

        # REJECTION 
        if Contraint_handling == 1:
            for i in range(swarm_size):
                var = pos[:, i]

                # check if particle violates any boundary
                while not all(constrain_func[j](var) <= constrain_bound[j] for j in range(len(constrain_func))):

                    # generate new particle within bounds
                    var = (bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * \
                           np.random.rand(num_var))

                # accept the newly found valid point as part of the swarm
                pos[:, i] = var

        # PENALTY
        penalty_val = np.zeros(swarm_size)                                      # container for penalty values

        if Contraint_handling == 2:
            for i in range(swarm_size):
                var = pos[:, i]

                # Calculate penalty value
                def penalty_func(var):
                    return sum(max(0, constrain_func[j](var) - constrain_bound[j]) for j in range(len(constrain_func)))
                
                penalty_val[i] = penalty_func(var)                              # assign the value as penalty value for that particle

        # loop to find function/ fitness value
        fitness = np.zeros(swarm_size)                                          # container for fitness values

        for i in range(swarm_size):
            fitness[i] = obj_func(pos[:, i])                                    # evaluate fitness values

        # Calculate avg fitness for the iteration & store
        avg_fitness = np.sum(fitness)/swarm_size                                # average function values for this iteration

        avg_fitness_per_iter[t-1] = np.sum(fitness)/swarm_size                  # store avearge func. value per iteration

        # Adjust function value based on penalty
        for i in range(swarm_size):
            fitness[i] = fitness[i] + abs(penalty_factor*avg_fitness * penalty_val[i])     # evaluate modified fitness values based on penalty

        # Calculate best fitness for the iteration                      
        iter_best = np.min(fitness)                                             # calculate best value and index of this iteration
        iter_indx = np.argmin(fitness)

        # loop to update personal best value and position
        for i in range(swarm_size):
            # intially set pbest to the 1st iteration values
            if t == 1:
                pbest[i] = fitness[i]                                           
                pbest_pos[:, i] = pos[:, i].copy()

            # for next iterations check if value gets better
            else:
                if fitness[i] < pbest[i]:                                       # if gets better (lower)
                    pbest[i] = fitness[i]                                       # update pbest for each particle
                    pbest_pos[:, i] = pos[:, i].copy()                                 # update pbest position for each particle

        # finding and updating global best value and position
        if t == 1:
            gbest = iter_best                                           
            gbest_pos = pos[:, iter_indx].copy()                                      # set gbest position to the best particle's position
            counter = 0

        # for next iterations check if value gets better
        else:
            if iter_best < gbest:                                         
                gbest = iter_best                                       
                gbest_pos = pos[:, iter_indx].copy()                                  # set gbest position to the best particle's position
                counter = 0
            else: 

                # stopping criteria based on improvement  
                counter = counter + 1
                if counter > 50:
                    # print(f'No improvement for 50 iters; stopping at t = {t}')
                    gbest_track[t-1:] = np.ones(num_iter-t+1) * gbest_track[t-2]                                # Fill the rest of the last found val 
                    avg_fitness_per_iter[t-1:] = np.ones(num_iter-t+1) * avg_fitness_per_iter[t-2]                       # Fill the rest of the last found val
                    break

        gbest_track[t-1] = gbest

        # update time
        t = t + 1                                                                 

        # calculate new velocity and new position
        # if inertia effect is being used, then reduce intertia with iteration
        if PSO_variant == 2:
            w = w_max - w_min*(t/num_iter)                                      # update inertia with time

        r1 = np.random.rand(num_var, swarm_size)                                # random factor for C1
        r2 = np.random.rand(num_var, swarm_size)                                # random factor for C2

        if t < num_iter:                                                        # only update for the number of iteration
            for i in range(swarm_size): 
                new_vel[:, i] = X*(w*vel[:, i] + (C1*r1[:, i]) * \
                    (pbest_pos[:, i] - pos[:, i]) + (C2*r2[:, i]) * \
                    (gbest_pos - pos[:, i]))                                     # update velocity

                # Add craziness effect based on settings
                if craziness == 1:
                    new_vel[:, i] = X*(w*vel[:, i] + (C1*r1[:, i]) * \
                    (pbest_pos[:, i] - pos[:, i]) + (C2*r2[:, i]) * \
                    (gbest_pos - pos[:, i]) + vel[:, i] * np.random.rand(num_var))

                # Clamp velocities using Vmax (1-D per-dimension limits)
                new_vel[:, i] = np.maximum(np.minimum(new_vel[:, i], Vmax_1d), -Vmax_1d)
        
                pos[:, i] = pos[:, i] + new_vel[:, i]                           # update position

                # Clamp particle positions using bounds
                pos[:, i] = np.maximum(np.minimum(pos[:, i], bounds[:, 1]), bounds[:, 0])

            vel = new_vel.copy()                                                 # setting new velocity as the prev. velocity for the next iter.

    iterations = t - 1
    
    return gbest, gbest_pos, iterations, gbest_track, avg_fitness_per_iter


def _acquisition_scores(gp, pop_u, y_best, opt, type_of_Problem):
    """Evaluate acquisition on unit-space points and return maximization scores."""
    pop_u = np.asarray(pop_u, dtype=float)
    opt_name = str(opt)
    problem_type = str(type_of_Problem).lower()

    if opt_name == "MaxMSE":
        _, sd = gp.predict(pop_u, return_std=True)
        return np.asarray(sd, dtype=float) ** 2

    if opt_name == "EI":
        mu, sd = gp.predict(pop_u, return_std=True)
        mu = np.asarray(mu, dtype=float)
        sigma = np.maximum(np.asarray(sd, dtype=float), 1e-12)
        if problem_type == "min":
            z = (float(y_best) - mu) / sigma
            return (float(y_best) - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        z = (mu - float(y_best)) / sigma
        return (mu - float(y_best)) * norm.cdf(z) + sigma * norm.pdf(z)

    mu = np.asarray(gp.predict(pop_u), dtype=float).ravel()
    if problem_type == "min":
        return -mu
    return mu


def pso_function(gp, num_var, bounds, y_best, swarm_size, num_iter, r, opt, type_of_Problem):
    """
    Basic PSO for GP acquisition with ESPSOLS-compatible signature.

    Returns:
        X_phys: candidate points in physical space (sorted best->worst)
        scores: acquisition scores for X_phys (higher is better)
        pop_phys: final swarm positions in physical space
    """
    _ = r  # kept for compatibility with ESPSOLS call sites

    d = int(num_var)
    n = int(swarm_size)
    iters = int(num_iter)
    bounds = np.asarray(bounds, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = np.maximum(hi - lo, 1e-12)

    def from_unit(U):
        U = np.asarray(U, dtype=float)
        return lo + U * span

    # Simple PSO in unit space [0, 1]^d
    pop = np.random.rand(n, d)
    vel = np.random.uniform(-0.2, 0.2, size=(n, d))

    c1 = 2.05
    c2 = 2.05
    phi = c1 + c2
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))  # constriction factor

    pbest_pos = pop.copy()
    pbest_scores = _acquisition_scores(gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem)

    g_idx = int(np.argmax(pbest_scores))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_score = float(pbest_scores[g_idx])

    for t in range(iters):
        if np.isclose(chi, 1.0):
            w = 0.9 - (0.5 * (float(t) / max(float(iters), 1.0)))  # 0.9 -> 0.4
        else:
            w = 1.0
        r1 = np.random.rand(n, d)
        r2 = np.random.rand(n, d)

        vel = chi * (w * vel + c1 * r1 * (pbest_pos - pop) + c2 * r2 * (gbest_pos - pop))
        vel = np.clip(vel, -0.3, 0.3)
        pop = np.clip(pop + vel, 0.0, 1.0)

        scores = _acquisition_scores(gp, pop, y_best=y_best, opt=opt, type_of_Problem=type_of_Problem)
        improve = scores > pbest_scores
        if np.any(improve):
            pbest_scores[improve] = scores[improve]
            pbest_pos[improve] = pop[improve].copy()

            best_idx = int(np.argmax(pbest_scores))
            if float(pbest_scores[best_idx]) > gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest_pos = pbest_pos[best_idx].copy()

    # Return ranked candidates (like ESPSOLS returns multiple niche seeds)
    order = np.argsort(pbest_scores)[::-1]
    ranked_u = pbest_pos[order]
    ranked_scores = np.asarray(pbest_scores[order], dtype=float)

    return from_unit(ranked_u), ranked_scores, from_unit(pop)
