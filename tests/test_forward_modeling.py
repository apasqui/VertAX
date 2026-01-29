"""Simple forward test for the periodic case."""

# Package imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
from time import perf_counter

import jax.numpy as jnp
import optax
from numpy.testing import assert_allclose

from vertax.energy import energy_shape_factor_homo
from vertax.pbc import PbcMesh


def test_forward_modeling_for_regressions() -> None:
    """Check identical result of a standard test with previous results (october 2025)."""
    t_start = perf_counter()

    # Settings
    n_cells = 100
    # Initial condition
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    pbc_mesh = PbcMesh.periodic_voronoi_from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
    pbc_mesh.min_dist_T1 = 0.2

    # Solver
    pbc_mesh.inner_solver = optax.sgd(learning_rate=0.01)
    pbc_mesh.max_nb_iterations = 1000
    pbc_mesh.tolerance = 1e-4
    pbc_mesh.patience = 5

    pbc_mesh.vertices_params = jnp.asarray([0.0])
    pbc_mesh.edges_params = jnp.asarray([0.0])
    pbc_mesh.faces_params = jnp.asarray([3.7])

    def energy(
        vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, _he_params: Array, face_params: Array
    ) -> Array:
        """We use an energy given in vertAX for this example.

        But only indirectly as the loss function for an inner optimization needs a specific function signature.
        """
        return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)

    # Energy minimization
    pbc_mesh.inner_opt(loss_function_inner=energy)

    print(f"Test forward modelling took {(perf_counter() - t_start):.2f} s.")

    # To create a new reference for the regression test only
    ref_path = "tests/reference_result_test_forward_modeling.npz"
    # pbc_mesh.save_mesh(ref_path)
    ref_mesh = PbcMesh.load_mesh(ref_path)

    assert_allclose(pbc_mesh.vertices, ref_mesh.vertices, rtol=0.001)
    assert_allclose(pbc_mesh.edges, ref_mesh.edges, rtol=0.001)
    assert_allclose(pbc_mesh.faces, ref_mesh.faces, rtol=0.001)


if __name__ == "__main__":
    test_forward_modeling_for_regressions()
