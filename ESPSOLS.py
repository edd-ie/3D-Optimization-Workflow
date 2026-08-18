"""
This is the implementation of the EF assisted Speciation PSO with local search. 
                                                Irtiza Khan
                                                Date started: October 23, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from bound_check import bound_check
from scipy.stats import norm
# Clear workspace equivalent
# rng default equivalent
# np.random.seed(42)
# Get test problem info

def ESPSOLS(gp, num_var, bounds, y_best, swarm_size, num_iter, r, opt, type_of_Problem):
    obj_func = gp.predict   
    C1 = 2.05
    C2 = 2.05
    PSO_variant = 1
    w_max = 0.9 
    w_min = 0.5
    Particle_Gen_method = 1
    Contraint_handling = 0
    craziness = 0
    # Determine PSO direction
    type_of_PSO = (
        "min" if (opt == "mu" and type_of_Problem == "min") else "max"
    )

    search_bounds = np.asarray(bounds)
    lo = search_bounds[:, 0]
    hi = search_bounds[:, 1]
    def to_unit(X):
        X = np.asarray(X, dtype=float)
        return (X - lo) / (hi - lo)
    def from_unit(U):
        U = np.asarray(U, dtype=float)
        return lo + U * (hi - lo)
    def _scaled_dist(a, b, r_param):
            diff = a - b
            if np.ndim(r_param) == 0:
                return np.linalg.norm(diff)
            r_vec = np.asarray(r_param, dtype=float)
            if r_vec.shape[0] == diff.shape[0]:
                safe = np.where(r_vec == 0.0, 1.0, r_vec)
                return np.linalg.norm(diff / safe)
            return np.linalg.norm(diff)

    def dynamic_radius(r,swarm_size,pop,L_sorted_fitness_idx):
            S = []                                              # list of seed vectors
            assignments = np.ones(swarm_size, dtype=int)       # seed index for each particle (–1 = unassigned)
            order = L_sorted_fitness_idx                        # visit in sorted fitness order
            for k in order:
                x = pop[k, :]
                found = False
                for s_idx, s_vec in enumerate(S):
                    if np.ndim(r) == 0:
                        in_species = (np.linalg.norm(x - s_vec) <= float(r))
                    else:
                        r_vec = np.asarray(r, dtype=float)
                        in_species = np.all(np.abs(x - s_vec) <= r_vec)
                    if in_species:
                        assignments[k] = s_idx
                        found = True
                        break
                if not found:
                    S.append(x.copy())
                    assignments[k] = len(S)-1
            dist_total = 0.0
            for s in (S):
                dist_total += _scaled_dist(s, S[0], r)
            return S, assignments

        


    # Initialize velocity, constriction factor, & intertia (operate in unit space)
    Vmax_phys = (bounds[:, 1] - bounds[:, 0])[None, :]
    Vmax = np.ones((1, num_var), dtype=float)               # unit-space velocity cap
    Vmax_1d = Vmax                                          # ensure 1-D vector for per-particle clamping
    C = C1 + C2
    if PSO_variant == 1:                                                        # with the constriction factor effect                                                                
        X = 2/(C-2+np.sqrt(C**2-4*C))                                          # contriction factor
        w = 1                                                                   # inertia

        # Initialize random velocities within variable limits
        vel = -0.5*Vmax + 0.5*Vmax * np.random.rand(swarm_size,num_var)      # velocity initialized with range a of 20% of the variable dynamic range

    elif PSO_variant == 2:                                                      # without the constriction factor effect 
        X = 1
        vel = -0.5*Vmax + 0.5*Vmax * np.random.rand(swarm_size, num_var) 
    
    # Initialize random position within variable limits (convert to unit space)
    if Particle_Gen_method == 1:
        pop = bound_check(33, search_bounds, num_var, swarm_size)
        pop = to_unit(pop)
        # pop = search_bounds[:, 0] + (search_bounds[:, 1] - search_bounds[:, 0]) * np.random.rand(swarm_size, num_var)



    new_vel = vel.copy()                                                        # set the randomly generated velocity to new velocity
    t = 0                                                                       # iteration

    # Generate data containers
    pbest = np.zeros(swarm_size)                                                # container for pbest values
    pbest_pos = np.zeros((swarm_size, num_var))                                # container for pbest position
    gbest_track = np.zeros(num_iter)                                            # track the global best
    avg_fitness_per_iter = np.zeros(num_iter)                                   # track the average func. val. per iter.
    pop_store = []                                                               # container for population
    fitness_store = []                                                        # container for fitness values

    # Main loop
    while t < num_iter:

        # No explicit constraint handling in this variant; penalty stays zero
        penalty_val = np.zeros(swarm_size)

        # loop to find function/ fitness value
        if opt == "MaxMSE":
           _,sd = gp.predict(pop, return_std=True)
           fitness = sd**2
        elif opt == "EI": 
            mu,sd = gp.predict(pop, return_std=True)
            sigma = np.maximum(sd, 1e-12)
            if type_of_Problem == "min":
                z = (y_best - mu) / sigma
                fitness = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            elif type_of_Problem == "max":
                z = (mu - y_best) / sigma
                fitness = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        elif opt == "mu":
            fitness = gp.predict(pop)
    


        # pop_store.append(pop.copy())
        # fitness_store.append(fitness.copy())

        # # Calculate avg fitness for the iteration & store
        avg_fitness = np.sum(fitness)/swarm_size                                # average function values for this iteration

        # avg_fitness_per_iter[t] = avg_fitness                  # store avearge func. value per iteration

        penalty_factor = 5
        if type_of_PSO == 'min':
            fitness = fitness + abs(penalty_factor*avg_fitness * penalty_val)      # evaluate modified fitness values based on penalty
        elif type_of_PSO == 'max':
            fitness = fitness - abs(penalty_factor*avg_fitness * penalty_val)      # evaluate modified fitness values based on penalty

        # Update personal bests (pbest) using current fitness
        if t == 0:
            pbest = fitness.copy()
            pbest_pos = pop.copy()
        else:
            if type_of_PSO == 'min':
                better_mask = fitness < pbest
            elif type_of_PSO == 'max':
                better_mask = fitness > pbest
            else:
                better_mask = np.zeros_like(pbest, dtype=bool)
            pbest[better_mask] = fitness[better_mask]
            pbest_pos[better_mask, :] = pop[better_mask, :]

        if type_of_PSO == 'min':
            L_sorted_fitness = np.sort(fitness) 
            L_sorted_fitness_idx = np.argsort(fitness)                                  # calculate best value and index of this iteration
            L_sorted_pop = pop[L_sorted_fitness_idx]
        elif type_of_PSO == 'max':
            L_sorted_fitness = np.sort(fitness)[::-1]                                       # descending for maximization
            L_sorted_fitness_idx = np.argsort(-fitness)
            L_sorted_pop = pop[L_sorted_fitness_idx]
        
        # golden section search for the optimal radius      
        # r = golden_section_search(r,swarm_size,pop,L_sorted_fitness_idx,type_of_PSO)
        S, assignments = dynamic_radius(r,swarm_size,pop,L_sorted_fitness_idx)

        # nBest position per particle (your seed_p) in original population order:
        seeds_arr = np.vstack(S)                            # (n_seeds, num_var)
        seed_p = seeds_arr[assignments]                     # (swarm_size, num_var)


        # find the largest species
        counts = np.bincount(assignments, minlength=len(S))     # size per niche
        LN_idx = np.argmax(counts);  
        SN_idx = np.argmin(counts)
        LN_mask = (assignments == LN_idx)
        SM_mask = (assignments == SN_idx)
        LN_idx_list = np.where(LN_mask)[0]                      # row indices of LN particles
        SM_idx_list = np.where(SM_mask)[0]

        # sort LN by fitness
        if type_of_PSO == 'min':
            LN_sorted_local = LN_idx_list[np.argsort(fitness[LN_idx_list])]
        elif type_of_PSO == 'max':
            LN_sorted_local = LN_idx_list[np.argsort(-fitness[LN_idx_list])]

        # number to steer (integer, nonnegative)
        DS = int(max(0, (counts[LN_idx] - counts[SN_idx]) // 2))
        DV = seeds_arr[LN_idx] - seeds_arr[SN_idx]              # direction vector

        # last DS indices in LN (the worst DS)
        LN_last_DS_idx = LN_sorted_local[-DS:] if DS > 0 else np.array([], dtype=int)

        
    
                                    
        if PSO_variant == 2:
            w = w_max - w_min*(t/num_iter)                                      # update inertia with time

        r1 = np.random.rand(swarm_size, num_var)                                # random factor for C1
        r2 = np.random.rand(swarm_size, num_var)                                # random factor for C2

                                                                # only update for the number of iteration
        
        new_vel = X*(w*vel + (C1*r1) * (pbest_pos - pop) + (C2*r2) *(seed_p - pop))                                     # update velocity

        # Add craziness effect based on settings
        if craziness == 1:
            new_vel = X*(w*vel + (C1*r1) * (pbest_pos - pop) + (C2*r2) * (seed_p - pop) + vel * np.random.rand(num_var))


        if DS > 0:
            new_vel[LN_last_DS_idx, :] += DV

        new_vel = np.clip(new_vel, -1*Vmax, 1*Vmax)
        pop = pop + new_vel                          # update position
        pop = np.clip(pop, 0.0, 1.0)
        vel = new_vel.copy()                                                 # setting new velocity as the prev. velocity for the next iter.

        
        # local search
        for i in range(swarm_size):
            d = np.linalg.norm(pbest_pos - pbest_pos[i, :], axis=1)
            d[i] = np.inf
            idx = np.argmin(d)
            if type_of_PSO == 'min':
                if pbest[idx] <= pbest[i]:
                    temp= pbest_pos[i,:] + C1*np.random.rand(num_var) * 2*(1+(t/num_iter))*(pbest_pos[idx,:] - pbest_pos[i,:])
                else:
                    temp = pbest_pos[i,:] + C1*np.random.rand(num_var) *2*(1+(t/num_iter))*( pbest_pos[i,:] - pbest_pos[idx,:])

                temp = np.clip(temp, 0.0, 1.0)
                # Evaluate temp fitness consistent with 'opt'
                if opt == "MaxMSE":
                    _, sd_tmp = gp.predict(temp.reshape(1, -1), return_std=True)
                    temp_fitness = (sd_tmp[0] ** 2)
                elif opt == "EI":
                    mu_tmp, sd_tmp = gp.predict(temp.reshape(1, -1), return_std=True)
                    sigma = max(sd_tmp[0], 1e-12)
                    if type_of_Problem == "min":
                        z = (y_best - mu_tmp[0]) / sigma
                        temp_fitness = (y_best - mu_tmp[0]) * norm.cdf(z) + sigma * norm.pdf(z)
                    else:
                        z = (mu_tmp[0] - y_best) / sigma
                        temp_fitness = (mu_tmp[0] - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    temp_fitness = float(gp.predict(temp.reshape(1, -1))[0])
                if temp_fitness < pbest[i]:
                    pbest_pos[i,:] = temp.copy()
                    pbest[i] = float(temp_fitness)
            elif type_of_PSO == 'max':
                if pbest[idx] >= pbest[i]:
                    temp= pbest_pos[i,:] + C1*np.random.rand(num_var) * 2*(1+(t/num_iter))*(pbest_pos[idx,:] - pbest_pos[i,:])
                else:
                    temp = pbest_pos[i,:] + C1*np.random.rand(num_var) * 2*(1+(t/num_iter))*(pbest_pos[i,:] - pbest_pos[idx,:])
                temp = np.clip(temp, 0.0, 1.0)
                # Evaluate temp fitness consistent with 'opt'
                if opt == "MaxMSE":
                    _, sd_tmp = gp.predict(temp.reshape(1, -1), return_std=True)
                    temp_fitness = (sd_tmp[0] ** 2)
                elif opt == "EI":
                    mu_tmp, sd_tmp = gp.predict(temp.reshape(1, -1), return_std=True)
                    sigma = max(sd_tmp[0], 1e-12)
                    if type_of_Problem == "min":
                        z = (y_best - mu_tmp[0]) / sigma
                        temp_fitness = (y_best - mu_tmp[0]) * norm.cdf(z) + sigma * norm.pdf(z)
                    else:
                        z = (mu_tmp[0] - y_best) / sigma
                        temp_fitness = (mu_tmp[0] - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    temp_fitness = float(gp.predict(temp.reshape(1, -1))[0])
                if temp_fitness > pbest[i]:
                    pbest_pos[i,:] = temp.copy()
                    pbest[i] = float(temp_fitness)


        # update time
        t += 1  

        if t == num_iter:
            fitness = gp.predict(pop)
            pop_store.append(pop.copy())
            fitness_store.append(fitness.copy())


    # detect all the unique niches 
    # --- seed fitness and sorting ---
    S = np.vstack(S) if isinstance(S, list) else np.asarray(S)  # ensure (n,D)
    S_fitness = gp.predict(S)

    if type_of_PSO == 'min':
        S_order = np.argsort(S_fitness)
    else:
        S_order = np.argsort(-S_fitness)

    S_sorted          = S[S_order]
    S_fitness_sorted  = S_fitness[S_order]

    # --- unique niches by distance r (supports anisotropic r vectors) ---
    kept_idx = []
    rejected_idx = []

    for local_idx, s in enumerate(S_sorted):
        if not kept_idx:
            kept_idx.append(local_idx)
            continue
        is_close = False
        if np.ndim(r) == 0:
            thr = float(r)
            for i in kept_idx:
                if np.linalg.norm(S_sorted[i] - s) <= thr:
                    is_close = True
                    break
        else:
            r_vec = np.asarray(r, dtype=float)
            for i in kept_idx:
                if np.all(np.abs(S_sorted[i] - s) <= r_vec):
                    is_close = True
                    break
        if not is_close:
            kept_idx.append(local_idx)
        else:
            rejected_idx.append(local_idx)

    # final groups
    niches_final   = S_sorted[kept_idx]       # kept seeds (unique niches)
    rejected = S_sorted[rejected_idx]   # too-close-to-better seeds



    # Evaluate final seed scores consistent with 'opt'
    if niches_final.shape[0] == 0:
        seed_final = np.empty((0,), dtype=float)
    elif opt == "MaxMSE":
        _, sd_final = gp.predict(niches_final, return_std=True)
        seed_final = (sd_final ** 2)
    elif opt == "EI":
        mu_f, sd_f = gp.predict(niches_final, return_std=True)
        sigma = np.maximum(sd_f, 1e-12)
        if type_of_Problem == "min":
            z = (y_best - mu_f) / sigma
            seed_final = (y_best - mu_f) * norm.cdf(z) + sigma * norm.pdf(z)
        else:
            z = (mu_f - y_best) / sigma
            seed_final = (mu_f - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
    else:
        seed_final = gp.predict(niches_final)


    return from_unit(niches_final), seed_final, from_unit(pop)

