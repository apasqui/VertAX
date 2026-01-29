"""Enumerations declaration, separated here to avoid dependency cycles."""

from enum import Enum


class BilevelOptimizationMethod(Enum):
    """Which optimization method to use in the bi-level optimization."""

    AUTOMATIC_DIFFERENTIATION = "ad"
    EQUILIBRIUM_PROPAGATION = "ep"
    IMPLICIT_DIFFERENTIATION = "id"
    ADJOINT_STATE = "as"
