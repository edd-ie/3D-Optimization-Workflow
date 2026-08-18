"""
This is the interpretation of my basic understanding of PSO. 
There are confusion with the basic version including contriction factor 
and the inertia. This also involves the concept of Vmax, how to handle 
and bound it.
                                                Irtiza Khan
                                                Date started: May 05, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from bound_check import bound_check
from penalty import penalty_check
from repair import repair


# Clear workspace equivalent
# rng default equivalent
# np.random.seed(42)
def PSO_main(gp, num_var, bounds, constrain_func, constrain_bound, type_of_PSO: str):
  
                        # location of the global maximum
    swarm_size = 50 
                                                            # number of particles in swarm
    num_iter = 100                                                              # number of iteration                                                                 
    C1 = 2.05                                                                   # Pbest coefficient 
    C2 = 2.05                                                                   # Gbest coefficient
    C = C1 + C2
    PSO_variant = 1                                                             # PSO variant => 
                                                                                # 1: constriction factor 
                                                                                # 2: intertia & Vmax

    w_max = 0.9                                                                 # inertia range for dynamic inertia                                                   
    w_min = 0.5
  


    Contraint_handling = 2                                                      # Contraint handling technique 
                                                                                # 0 => no constraints 
                                                                                # 1 => repair 
                                                                                # 2 => penalty
    craziness = 0
    search_bounds = np.asarray([bounds])


    # Initialize velocity, constriction factor, & intertia 
    Vmax = 0.2 * (search_bounds[:, 1] - search_bounds[:, 0])            # shape: (num_var,)
    Vmax_1d = Vmax                                         # ensure 1-D vector for per-particle clamping




    if PSO_variant == 1:                                                        # with the constriction factor effect                                                                
        X = 2/(C-2+np.sqrt(C**2-4*C))                                          # contriction factor
        w = 1                                                                   # inertia

        # Initialize random velocities within variable limits
            # velocity initialized with range a of 20% of the variable dynamic range

    elif PSO_variant == 2:                                                      # without the constriction factor effect 
        X = 1
    

    # fresh initialization per run
    vel = -Vmax + 2 * Vmax * np.random.rand(swarm_size, num_var)
    pop, _ = bound_check(42, search_bounds, num_var, swarm_size)
    new_vel = vel.copy()
    t = 0

    # Containers
    pbest = np.zeros(swarm_size)
    pbest_pos = np.zeros((swarm_size, num_var))
    gbest_track = np.zeros(num_iter)
    avg_fitness_per_iter = np.zeros(num_iter)
    counter = 0                                # track the average func. val. per iter.
    infeasible_fraction = np.zeros(num_iter)
    avg_penalty_per_iter = np.zeros(num_iter)

    # Main loop
    while t < num_iter:

        # Constraint handling & metrics (compute once per iteration)
        penalty_val = penalty_check(pop, search_bounds)
        mask = penalty_val > 0
        bad_pop = pop[mask]
        infeasible_fraction[t] = bad_pop.shape[0] / float(swarm_size)
        avg_penalty_per_iter[t] = float(np.mean(penalty_val)) if penalty_val.size else 0.0

        if Contraint_handling == 1:
            if bad_pop.shape[0] > 0:
                pop[mask] = repair(bad_pop, search_bounds)
        elif Contraint_handling == 2:
            if bad_pop.shape[0] > int(0.8*swarm_size):
                pop[mask] = repair(bad_pop, search_bounds)

       
        results = gp.predict(pop, return_std=False)
      
        fitness = results

        # Calculate avg fitness for the iteration & store
        avg_fitness = np.sum(fitness)/swarm_size                                # average function values for this iteration

        avg_fitness_per_iter[t] = avg_fitness                                   # store avearge func. value per iteration

        
        fitness = fitness - abs(5*avg_fitness * penalty_val)      # evaluate modified fitness values based on penalty

        # top 10% of the population
        if type_of_PSO == 'minimization':
            top_10_idx = np.argmin(fitness)[:int(0.1*swarm_size)].tolist()
            top_10_pop = pop[top_10_idx]
            top_10_fitness = fitness[top_10_idx]
        elif type_of_PSO == 'maximization':
            top_10_idx = np.argmax(fitness)[:int(0.1*swarm_size)].tolist()
            top_10_pop = pop[top_10_idx].tolist()
            top_10_fitness = fitness[top_10_idx].tolist()
        

        # Calculate best fitness for the iteration 
        if type_of_PSO == 'minimization':
            iter_best = np.min(fitness)                                             # calculate best value and index of this iteration
            iter_indx = np.argmin(fitness)
        elif type_of_PSO == 'maximization':
            iter_best = np.max(fitness)                                             # calculate best value and index of this iteration
            iter_indx = np.argmax(fitness)

        # loop to update personal best value and position
        for i in range(swarm_size):
            # intially set pbest to the 1st iteration values
            if t == 0:
                pbest[i] = fitness[i]                                           
                pbest_pos[i] = pop[i].copy()

            # for next iterations check if value gets better
            else:
                if type_of_PSO == 'minimization':
                    if fitness[i] < pbest[i]:                                       # if gets better (lower)
                        pbest[i] = fitness[i]                                       # update pbest for each particle
                        pbest_pos[i] = pop[i].copy()                                 # update pbest position for each particle
                elif type_of_PSO == 'maximization':
                    if fitness[i] > pbest[i]:                                       # if gets better (lower)
                        pbest[i] = fitness[i]                                       # update pbest for each particle
                        pbest_pos[i] = pop[i] .copy()                                # update pbest position for each particle

        # finding and updating global best value and position
        if t == 0:
            gbest = iter_best                                           
            gbest_pos = pop[iter_indx]                                       # set gbest position to the best particle's position
            counter = 0

        # for next iterations check if value gets better
        else:
            if type_of_PSO == 'minimization':
                if iter_best < gbest:                                         
                    gbest = iter_best                                       
                    gbest_pos = pop[iter_indx].copy()                                   # set gbest position to the best particle's position
                    counter = 0
            elif type_of_PSO == 'maximization':
                if iter_best > gbest:                                         
                    gbest = iter_best                                       
                    gbest_pos = pop[iter_indx].copy()                                   # set gbest position to the best particle's position
                    counter = 0
            else: 
                # stopping criteria based on improvement  
                counter = counter + 1
                if counter > 15:
                    print(f'No improvement for 15 iters; stopping at t = {t}')
                    break

        gbest_track[t] = gbest

                                                                    

        # calculate new velocity and new position
        # if inertia effect is being used, then reduce intertia with iteration
        if PSO_variant == 2:
            w = w_max - w_min*(t/num_iter)                                      # update inertia with time

        r1 = np.random.rand(swarm_size, num_var)                                # random factor for C1
        r2 = np.random.rand(swarm_size, num_var)                                # random factor for C2

        # update velocity
        new_vel = X*(w*vel + (C1*r1) *(pbest_pos - pop) + (C2*r2) *(gbest_pos - pop)) # update velocity
                                

        # Add craziness effect based on settings
        if craziness == 1:
            new_vel = X*(w*vel + (C1*r1) *(pbest_pos - pop) + (C2*r2) *(gbest_pos - pop) + vel * np.random.rand(num_var))

        # Clamp velocities using Vmax (1-D per-dimension limits)
        new_vel = np.maximum(np.minimum(new_vel, Vmax_1d), -Vmax_1d)

        pop = pop + new_vel                           # update position


        vel = new_vel.copy()                  # setting new velocity as the prev. velocity for the next iter.

     
        t += 1 

        return top_10_pop, top_10_fitness

    

   


    """
        Modifications

        May 07, 2025 - Completed basic code: contriction factor or Vmax, 
                    & inertia, Particle generation - random or equally 
                    distributed in all dimensions. 
                    - Clamping or not? - need to test, graphs for visulization 
                    - tracking Global best (good for now)

        May 08, 2025 - Added in constraint handling: rejection & penalty. 
                    Rejection struggles for problem 3.1 (too many constraints)  
                    but other problems work. 
                    - Penalty works for all the functions pretty 
                    well. 3.1 is sometimes struggling with just avg_fitness *
                    penalty. 5 * avg_fitness * penalty seems to work
                    consistantly.

        May 09, 2025 - Added in velocity and position clamping, otherwise particle
                    leave bounds for concave functions.
                    - Added average fitness value per iteration tracking so see
                    if optimizer is actually converging the particles. 
                    - What other tracking should be included for the parametric
                    analysis 
                    - Need to add more comments to make code clearer.

        May 12, 2025 - Added in comments, should be enough. 
                    - Added craziness
        """
