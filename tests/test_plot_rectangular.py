"""Manual check that a rectangular plot is working."""

from vertax.pbc import PBCMesh
from vertax.plot import plot_mesh


def show_rectangular_mesh() -> None:
    """Manual plot of a PBC mesh with a non square domain."""
    # Settings
    n_cells = 100
    width = 15
    height = 10

    # Initial condition
    mesh = PBCMesh.periodic_voronoi_from_random_seeds(n_cells, width, height, random_key=1)

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
