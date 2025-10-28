"""Abstract mesh module."""

from pathlib import Path
from typing import Any, NoReturn, Self, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array

T = TypeVar("T")


class NoPublicConstructor(type):
    """Metaclass that ensures a private constructor.

    If a class uses this metaclass like this:

        class SomeClass(metaclass=NoPublicConstructor):
            pass

    If you try to instantiate your class (`SomeClass()`),
    a `TypeError` will be thrown.
    """

    def __call__(cls, *args, **kwargs) -> NoReturn:  # noqa
        """Make it impossible to call with ClassName()."""
        msg = f"{cls.__module__}.{cls.__qualname__} has no public constructor"
        raise TypeError(msg)

    def _create(cls: type[T], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        return super().__call__(*args, **kwargs)  # type: ignore


class Mesh(metaclass=NoPublicConstructor):
    """Generic mesh structure."""

    def __init__(self) -> None:
        """Do nothing but create attributes. Do not call this."""
        self.vertices: Array = jnp.array([])
        self.edges: Array = jnp.array([])
        self.faces: Array = jnp.array([])
        self.width: float = 0
        self.height: float = 0

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        Args:
            path (str): Path to the saved file. The extension is .npz.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, allow_pickle=False, vertices=self.vertices, edges=self.edges, faces=self.faces)

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        Args:
            path (str): Path to the mesh file (.npz).

        Returns:
            Mesh: the mesh loaded from the .npz file.
        """
        mesh_file = np.load(path)
        return cls(mesh_file["vertices"], mesh_file["edges"], mesh_file["faces"])
