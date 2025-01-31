import os 
import numpy as np 
import jax.numpy as jnp 
import jax
from jax import jit, vmap, jacfwd

from vertax.start import create_geograph
from vertax.opt import inner, min_fun
from vertax.geo import get_area, get_length, update_pbc
from vertax.plot import plot_geograph
from vertax.topo import update_T1

def fire_opt(grad_potential, 
             energy,
             vertTable, 
             heTable, 
             faceTable, 
             areas_params, 
             edges_params,
             dt=0.001, 
             max_steps=1000, 
             alpha_start=0.001, 
             alpha_decay=0.99, 
             max_dt=1.0, 
             f_tol=1e-6):
    """
    FIRE Optimizer.

    Args:
        grad_potential: Callable that returns the gradient of the potential energy.
        initial_positions: Array of shape (N, 2) representing initial positions.
        dt: Initial time step.
        max_steps: Maximum number of steps for the minimization.
        alpha_start: Initial value of the damping parameter.
        alpha_decay: Decay factor for alpha (e.g., 0.99).
        max_dt: Maximum allowable time step.
        f_tol: Force tolerance to determine convergence.

    Returns:
        positions: Final positions of shape (N, 2).
        forces: Final forces of shape (N, 2).
        converged: Boolean indicating whether the minimization converged.
    """
    @jax.jit
    def step(state):
        vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = state

        #jax.debug.print("{x}\n", x=dt)
        #jax.debug.print("{x}\n", x=forces[0])

        # Update velocities using forces
        velocities = velocities + dt * forces
        #jax.debug.print("{x}\n", x=velocities[0])

        # Calculate the power P = sum(v · f) for all particles
        power = jnp.sum(velocities * forces)

        #jax.debug.print("{x}", x=velocities * forces)

        #jax.debug.print("{x}", x=power)

        # Update positions
        vertTable = vertTable + dt * velocities
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, MIN_DISTANCE=0.1)
        #jax.debug.print("{x}\n", x=vertTable)

        forces_norm_old = jnp.linalg.norm(forces)

        # Recompute forces
        forces = -grad_potential(vertTable, heTable, faceTable, areas_params, edges_params)
        #jax.debug.print("{x}\n", x=forces)

        # Compute norms for velocities and forces
        velocities_norm = jnp.linalg.norm(velocities)
        forces_norm = jnp.linalg.norm(forces)

        jax.debug.print("{x}", x=forces_norm_old)
        jax.debug.print("{x}\n", x=forces_norm)


        # If power is positive, increase dt and reduce alpha
        def positive_power():
            # jax.debug.print("positive power\n")
            new_dt = jnp.minimum(dt * 1.1, max_dt)
            new_alpha = alpha * alpha_decay
            velocities_adjusted = (1 - new_alpha) * velocities + new_alpha * forces * velocities_norm / (forces_norm + 1e-12)
            return new_dt, new_alpha, velocities_adjusted

        # If power is negative, reset velocities and reduce dt
        def negative_power():
            # jax.debug.print("negative power\n")
            new_dt = dt * 0.5
            new_alpha = alpha_start
            return new_dt, new_alpha, jnp.zeros_like(velocities)

        dt, alpha, velocities = jax.lax.cond(
            power > 0,
            positive_power,
            negative_power,
        )

        # Convergence check
        #converged = forces_norm < f_tol
        #converged = jnp.where(step_count == 100, True, False)
        converged = jnp.where(jnp.abs(forces_norm_old-forces_norm) < 1e-8, True, False)

        # jax.debug.print("{x}\n", x=energy(vertTable, heTable, faceTable, areas_params, edges_params))

        return (vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count + 1), converged

    # Initialize state
    forces = -grad_potential(vertTable, heTable, faceTable, areas_params, edges_params)
    state = (
        vertTable,
        heTable,
        faceTable,
        jnp.zeros_like(vertTable),  # Initial velocities (N, 2)
        forces,
        alpha_start,
        dt,
        0,  # Step count
    )

    @jax.jit
    def cond_fn(state_converged):
        _, converged = state_converged
        return ~converged

    @jax.jit
    def body_fn(state_converged):
        state, _ = state_converged
        state, converged = step(state)
        return state, converged

    # Iterate the FIRE algorithm
    final_state, converged = jax.lax.while_loop(
        cond_fn, body_fn, (state, False)
    )

    vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = final_state
    return vertTable, heTable, faceTable, forces, converged


################
### SEETINGS ###
################

n_cells = 20

K_areas = 0.01
area_param = 1 #0.6  # initial condition areas parameters 
edge_param = 0.1 #0.7  # initial condition edges parameters 

inner_epochs = 10
inner_lr = 0.01


############## 
### ENERGY ### 
############## 

@jit
def area_part(face: float, area_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    
    a = get_area(face, vertTable, heTable, faceTable)
    
    return (a - area_param) ** 2

@jit
def edge_part(edge: float, edge_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    
    l = get_length(edge, vertTable, heTable, L_box)[0]
    
    return edge_param * l

@jit
def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, areas_params: jnp.array, edges_params: jnp.array):
    
    mapped_areas_part = lambda face, a_param: area_part(face, a_param, vertTable, heTable, faceTable)
    mapped_edges_part = lambda edge, e_param: edge_part(edge, e_param, vertTable, heTable, faceTable)
    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), areas_params)
    edges_part = vmap(mapped_edges_part)(jnp.arange(len(heTable)), edges_params)
    
    # jax.debug.print("{x}\n", x=areas_part)
    # jax.debug.print("{x}\n", x=edges_part)

    return  jnp.sum(edges_part) + (0.5 * K_areas) * jnp.sum(areas_part)


##################
### SIMULATION ###
##################

L_box = jnp.sqrt(n_cells)

seeds = L_box * np.random.random_sample((n_cells, 2))

vertTable, faceTable, heTable = create_geograph(seeds, show=True)

areas_params = jnp.asarray([area_param] * len(faceTable))
edges_params = jnp.asarray([edge_param] * len(heTable))

def grad_energy(vertTable, heTable, faceTable, areas_params, edges_params):
    return jacfwd(energy, argnums=0)(vertTable, heTable, faceTable, areas_params, edges_params)

vertTable_gd, heTable_gd, faceTable_gd = vertTable, heTable, faceTable

# for i in range(5000):
#     print(energy(vertTable_gd, heTable_gd, faceTable_gd, areas_params, edges_params))
#     vertTable_gd = vertTable_gd - 0.01 * jacfwd(energy, argnums=0)(vertTable, heTable, faceTable, areas_params, edges_params)
#     vertTable_gd, heTable_gd, faceTable_gd = update_pbc(vertTable_gd, heTable_gd, faceTable_gd)
#     vertTable_gd, heTable_gd, faceTable_gd = update_T1(vertTable_gd, heTable_gd, faceTable_gd, MIN_DISTANCE=0.1)

# plot_geograph(vertTable_gd.astype(float), 
#             faceTable_gd.astype(int), 
#             heTable_gd.astype(int), 
#             L_box=L_box, 
#             multicolor=True, 
#             lines=True, 
#             vertices=False, 
#             path='./', 
#             name='i', 
#             save=False, 
#             show=True)

# Run FIRE minimizer
vertTable, heTable, faceTable, final_forces, converged = fire_minimizer_2d(grad_energy, energy, vertTable, heTable, faceTable, areas_params, edges_params)

plot_geograph(vertTable.astype(float), 
              faceTable.astype(int), 
              heTable.astype(int), 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)


# print(energy(vertTable,heTable,faceTable,areas_params,edges_params))



# def potential_energy_2d(positions):
#     """
#     Example potential energy: A simple 2D harmonic oscillator for each particle.
#     positions: Array of shape (N, 2).
#     """
#     return 0.5 * jnp.sum(positions**2)

# def grad_potential_2d(positions):
#     """Gradient of the 2D harmonic oscillator potential."""
#     return jax.grad(potential_energy_2d)(positions)

# # Initial positions for 3 particles in 2D
# initial_positions = jnp.array([
#     [1.0, -1.5],  # Particle 1
#     [2.0, 0.5],   # Particle 2
#     [-0.5, 1.5]   # Particle 3
# ])

# # Run FIRE minimizer
# final_positions, final_forces, converged = fire_minimizer_2d(grad_potential_2d, initial_positions)

# print("Final Positions:\n", final_positions)
# print("Final Forces:\n", final_forces)
# print("Converged:", converged)


# areas_params = jnp.asarray([area_param] * len(faceTable))
# edges_params = jnp.asarray([edge_param] * len(heTable))

# sgd_inner = optax.sgd(learning_rate=inner_lr)

# vertTable = min_fun(vertTable, 
#                     heTable, 
#                     faceTable, 
#                     areas_params, 
#                     edges_params, 
#                     energy,
#                     sgd_inner, 
#                     inner_epochs)














# import jax
# import jax.numpy as jnp

# def fire_minimizer_2d(grad_potential, initial_positions, dt=0.1, max_steps=1000, 
#                       alpha_start=0.1, alpha_decay=0.99, max_dt=1.0, f_tol=1e-6):
#     """
#     Generalized FIRE Minimizer for 2D positions implemented in JAX.

#     Args:
#         grad_potential: Callable that returns the gradient of the potential energy.
#         initial_positions: Array of shape (N, 2) representing initial positions.
#         dt: Initial time step.
#         max_steps: Maximum number of steps for the minimization.
#         alpha_start: Initial value of the damping parameter.
#         alpha_decay: Decay factor for alpha (e.g., 0.99).
#         max_dt: Maximum allowable time step.
#         f_tol: Force tolerance to determine convergence.

#     Returns:
#         positions: Final positions of shape (N, 2).
#         forces: Final forces of shape (N, 2).
#         converged: Boolean indicating whether the minimization converged.
#     """
#     def step(state):
#         positions, velocities, forces, alpha, dt, step_count = state

#         # Update velocities using forces
#         velocities = velocities + dt * forces

#         # Calculate the power P = sum(v · f) for all particles
#         power = jnp.sum(velocities * forces)

#         # Update positions
#         positions = positions + dt * velocities

#         # Recompute forces
#         forces = -grad_potential(positions)

#         # Compute norms for velocities and forces
#         velocities_norm = jnp.linalg.norm(velocities)
#         forces_norm = jnp.linalg.norm(forces)

#         # If power is positive, increase dt and reduce alpha
#         def positive_power():
#             new_dt = jnp.minimum(dt * 1.1, max_dt)
#             new_alpha = alpha * alpha_decay
#             velocities_adjusted = (1 - new_alpha) * velocities + new_alpha * forces / (forces_norm + 1e-12)
#             return new_dt, new_alpha, velocities_adjusted

#         # If power is negative, reset velocities and reduce dt
#         def negative_power():
#             new_dt = dt * 0.5
#             new_alpha = alpha_start
#             return new_dt, new_alpha, jnp.zeros_like(velocities)

#         dt, alpha, velocities = jax.lax.cond(
#             power > 0,
#             positive_power,
#             negative_power,
#         )

#         # Convergence check
#         converged = forces_norm < f_tol

#         return (positions, velocities, forces, alpha, dt, step_count + 1), converged

#     # Initialize state
#     forces = -grad_potential(initial_positions)
#     state = (
#         initial_positions,
#         jnp.zeros_like(initial_positions),  # Initial velocities (N, 2)
#         forces,
#         alpha_start,
#         dt,
#         0,  # Step count
#     )

#     # Iterate the FIRE algorithm
#     def cond_fn(state_converged):
#         _, converged = state_converged
#         return ~converged

#     def body_fn(state_converged):
#         state, _ = state_converged
#         state, converged = step(state)
#         return state, converged

#     final_state, converged = jax.lax.while_loop(
#         cond_fn, body_fn, (state, False)
#     )

#     positions, velocities, forces, alpha, dt, step_count = final_state
#     return positions, forces, converged


# def potential_energy_2d(positions):
#     """
#     Example potential energy: A simple 2D harmonic oscillator for each particle.
#     positions: Array of shape (N, 2).
#     """
#     return 0.5 * jnp.sum(positions**2)

# def grad_potential_2d(positions):
#     """Gradient of the 2D harmonic oscillator potential."""
#     return jax.grad(potential_energy_2d)(positions)

# # Initial positions for 3 particles in 2D
# initial_positions = jnp.array([
#     [1.0, -1.5],  # Particle 1
#     [2.0, 0.5],   # Particle 2
#     [-0.5, 1.5]   # Particle 3
# ])

# # Run FIRE minimizer
# final_positions, final_forces, converged = fire_minimizer_2d(grad_potential_2d, initial_positions)

# print("Final Positions:\n", final_positions)
# print("Final Forces:\n", final_forces)
# print("Converged:", converged)




#########################
### 1D IMPLEMENTATION ###
#########################

# import jax
# import jax.numpy as jnp
# 
# # Define the FIRE minimizer
# def fire_minimizer(grad_potential, initial_positions, dt=0.1, max_steps=1000, 
#                    alpha_start=0.1, alpha_decay=0.99, max_dt=1.0, f_tol=1e-6):
#     """
#     FIRE Minimizer implemented in JAX.

#     Args:
#         grad_potential: Callable that returns the gradient of the potential energy.
#         initial_positions: Initial positions of the particles.
#         dt: Initial time step.
#         max_steps: Maximum number of steps for the minimization.
#         alpha_start: Initial value of the damping parameter.
#         alpha_decay: Decay factor for alpha (e.g., 0.99).
#         max_dt: Maximum allowable time step.
#         f_tol: Force tolerance to determine convergence.

#     Returns:
#         positions: Final positions.
#         forces: Final forces.
#         converged: Boolean indicating whether the minimization converged.
#     """
#     def step(state):
#         positions, velocities, forces, alpha, dt, step_count = state

#         # Update velocities using forces
#         velocities = velocities + dt * forces

#         # Calculate the power P = v · f
#         power = jnp.sum(velocities * forces)

#         # Update positions
#         positions = positions + dt * velocities

#         # Recompute forces
#         forces = -grad_potential(positions)

#         # Normalize forces and velocities if needed
#         velocities_norm = jnp.linalg.norm(velocities)
#         forces_norm = jnp.linalg.norm(forces)

#         # If power is positive, increase dt and reduce alpha
#         def positive_power():
#             new_dt = jnp.minimum(dt * 1.1, max_dt)
#             new_alpha = alpha * alpha_decay
#             velocities_adjusted = (1 - new_alpha) * velocities + new_alpha * forces / forces_norm
#             return new_dt, new_alpha, velocities_adjusted

#         # If power is negative, reset velocities and reduce dt
#         def negative_power():
#             new_dt = dt * 0.5
#             new_alpha = alpha_start
#             return new_dt, new_alpha, jnp.zeros_like(velocities)

#         dt, alpha, velocities = jax.lax.cond(
#             power > 0,
#             positive_power,
#             negative_power,
#         )

#         # Convergence check
#         converged = forces_norm < f_tol

#         return (positions, velocities, forces, alpha, dt, step_count + 1), converged

#     # Initialize state
#     forces = -grad_potential(initial_positions)
#     state = (
#         initial_positions,
#         jnp.zeros_like(initial_positions),  # Initial velocities
#         forces,
#         alpha_start,
#         dt,
#         0,  # Step count
#     )

#     # Iterate the FIRE algorithm
#     def cond_fn(state_converged):
#         _, converged = state_converged
#         return ~converged

#     def body_fn(state_converged):
#         state, _ = state_converged
#         state, converged = step(state)
#         return state, converged

#     final_state, converged = jax.lax.while_loop(
#         cond_fn, body_fn, (state, False)
#     )

#     positions, velocities, forces, alpha, dt, step_count = final_state
#     return positions, forces, converged




# def potential_energy(x):
#     """Example: A simple harmonic oscillator potential."""
#     return 0.5 * jnp.sum(x**2)

# def grad_potential(x):
#     """Gradient of the harmonic oscillator potential."""
#     return jax.grad(potential_energy)(x)

# # Initial positions
# initial_positions = jnp.array([1.0, -1.5, 0.0])

# # Run FIRE
# final_positions, final_forces, converged = fire_minimizer(grad_potential, initial_positions)

# print("Final Positions:", final_positions)
# print("Final Forces:", final_forces)
# print("Converged:", converged)