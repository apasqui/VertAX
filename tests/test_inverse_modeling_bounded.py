"""Inverse modelling test for the bounded case."""

# Package imports
import jax.numpy as jnp
import numpy as np
import optax
from numpy.testing import assert_allclose

from vertax.bounded import BoundedMesh
from vertax.cost import cost_ratio
from vertax.energy import energy_bounded
from vertax.method_enum import BilevelOptimizationMethod


def test_regression() -> None:
    """Test for regression (December 2025)."""
    # Settings
    epochs = 2
    n_cells = 20
    n_edges = (n_cells - 1) * 3

    # Initial condition
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    rng_seed = 1
    bounded_mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=rng_seed)
    rng = np.random.default_rng(seed=rng_seed)
    bounded_mesh.min_dist_T1 = 0.025

    # Solver
    bounded_mesh.inner_solver = optax.sgd(learning_rate=0.01)
    bounded_mesh.outer_solver = optax.adam(learning_rate=0.01, nesterov=True)
    bounded_mesh.max_nb_iterations = 1000
    bounded_mesh.tolerance = 1e-6
    bounded_mesh.patience = 5
    bounded_mesh.bilevel_optimization_method = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
    bounded_mesh.vertices_params = jnp.asarray([0.0])
    bounded_mesh.edges_params = jnp.asarray(rng.random(n_edges) * 20 - 10)
    bounded_mesh.faces_params = jnp.asarray([0.0])

    # Energy minimization
    bounded_mesh.inner_opt(loss_function_inner=energy_bounded)

    for j in range(epochs + 1):
        print("epoch: " + str(j) + "/" + str(epochs) + "\t cost: " + str(cost_ratio(bounded_mesh.vertices)))

        bounded_mesh.bilevel_opt(loss_function_inner=energy_bounded, loss_function_outer=cost_ratio)

    saved_path = "tests/reference_result_test_inverse_modeling_bounded.npz"
    # bounded_mesh.save_mesh(saved_path)
    ref_mesh = BoundedMesh.load_mesh(saved_path)

    assert_allclose(bounded_mesh.vertices, ref_mesh.vertices, rtol=0.001)
    assert_allclose(bounded_mesh.edges, ref_mesh.edges, rtol=0.001)
    assert_allclose(bounded_mesh.faces, ref_mesh.faces, rtol=0.001)
    assert_allclose(bounded_mesh.angles, ref_mesh.angles, rtol=0.001)


if __name__ == "__main__":
    test_regression()
