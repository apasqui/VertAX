"""Manual check that a rectangular plot is working."""

import jax.numpy as jnp
import jax.random

from vertax.pbc import PBCMesh
from vertax.plot import plot_mesh


def show_rectangular_mesh() -> None:
    """Manual plot of a PBC mesh with a non square domain."""
    # Settings
    n_cells = 100
    width = 15
    height = 10

    # Initial condition
    key = jax.random.PRNGKey(1)
    seeds = jnp.array((width, height)) * jax.random.uniform(key, (n_cells, 2))
    mesh = PBCMesh._periodic_voronoi_from_seeds(seeds, width, height)

    plot_mesh(
        mesh.vertices,
        mesh.edges,
        mesh.faces,
        width,
        height,
        multicolor=True,
        lines=True,
        vertices=False,
        path="./tests/",
        name="rectangular_plot",
        show=False,
        save=True,
    )


if __name__ == "__main__":
    show_rectangular_mesh()
