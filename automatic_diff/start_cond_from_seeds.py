import os 

import numpy as np 

import jax.numpy as jnp 
from jax import jit, vmap

import optax 

from vertax.start import create_geograph_from_seeds
from vertax.opt import min_fun
from vertax.geo import get_area, get_length
from vertax.plot import plot_geograph


################
### SEETINGS ###
################

n_cells = 20

for edge_param in [0.4,0.5,0.6]:

    K_areas = 20
    area_param = 0.4 #0.6  # initial condition areas parameters 
    edge_param = 0.5 #0.7  # initial condition edges parameters 

    inner_epochs = 100
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
        
        l = get_length(edge, vertTable, heTable, L_box)
        
        return edge_param * l

    @jit
    def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, areas_params: jnp.array, edges_params: jnp.array):
        
        mapped_areas_part = lambda face, a_param: area_part(face, a_param, vertTable, heTable, faceTable)
        mapped_edges_part = lambda edge, e_param: edge_part(edge, e_param, vertTable, heTable, faceTable)
        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), areas_params)
        edges_part = vmap(mapped_edges_part)(jnp.arange(len(heTable)), edges_params)
        
        return  jnp.sum(edges_part) + (0.5 * K_areas) * jnp.sum(areas_part)


    ##################
    ### SIMULATION ###
    ##################

    L_box = jnp.sqrt(n_cells)

    seeds = L_box * np.random.random_sample((n_cells, 2))

    vertTable, heTable, faceTable = create_geograph_from_seeds(seeds, show=True)

    areas_params = jnp.asarray([area_param] * len(faceTable))
    edges_params = jnp.asarray([edge_param] * len(heTable))

    sgd_inner = optax.sgd(learning_rate=inner_lr)

    vertTable = min_fun(vertTable, 
                        heTable, 
                        faceTable, 
                        areas_params, 
                        edges_params, 
                        energy,
                        sgd_inner, 
                        inner_epochs)

    ##################
    ##### SAVING #####
    ##################

    os.makedirs('./initial_condition/eq_simulation/', exist_ok=True)

    plot_geograph(vertTable.astype(float), 
                faceTable.astype(int), 
                heTable.astype(int), 
                L_box=L_box, 
                multicolor=True, 
                lines=True, 
                vertices=False, 
                path='./initial_condition/eq_simulation/', 
                name='eq_simulation', 
                save=True, 
                show=True)

    with open('./initial_condition/eq_simulation/eq_settings.txt', 'w') as file:
        file.write('n_cells = ' + str(n_cells) + '\n')
        file.write('K_areas = ' + str(K_areas) + '\n')
        file.write('area_param = ' + str(area_param) + '\n')
        file.write('edge_param = ' + str(edge_param) + '\n')
        file.write('inner_epochs = ' + str(inner_epochs) + '\n')
        file.write('inner_lr = ' + str(inner_lr) + '\n')

    np.savetxt('./initial_condition/eq_simulation/vertTable.csv', vertTable, delimiter='\t', fmt='%1.18f')
    np.savetxt('./initial_condition/eq_simulation/faceTable.csv', faceTable, delimiter='\t', fmt='%1.0d')
    np.savetxt('./initial_condition/eq_simulation/heTable.csv', heTable, delimiter='\t', fmt='%1.0d')

    vertTable = np.loadtxt('./initial_condition/eq_simulation/vertTable.csv', delimiter='\t', dtype=np.float64)
    faceTable = np.loadtxt('./initial_condition/eq_simulation/faceTable.csv', delimiter='\t', dtype=np.int32)
    heTable = np.loadtxt('./initial_condition/eq_simulation/heTable.csv', delimiter='\t', dtype=np.int32)

    np.save('./initial_condition/eq_simulation/vertTable', vertTable)
    np.save('./initial_condition/eq_simulation/faceTable', faceTable)
    np.save('./initial_condition/eq_simulation/heTable', heTable)

    # os.remove('./initial_condition/eq_simulation/vertTable.csv')
    # os.remove('./initial_condition/eq_simulation/faceTable.csv')
    # os.remove('./initial_condition/eq_simulation/heTable.csv')

