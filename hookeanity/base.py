"""Basic types.

Attributes:
    DIM (int): Number of spatial dimensions.
    Real: Native Python or NumPy floating-point number.
    Point: Point coordinates in 3D.
    Grid: Grid of 3D point coordinates.
    ScalarField: Scalar field defined over a grid.
    VectorField: Vector field defined over a grid.
    ScalarValues: Scalar values on a mesh.
    VectorValues: Vector values on a mesh.
"""

from collections.abc import Callable
from typing import Any, TypeIs

import numpy as np
from jaxtyping import Float
from typeguard import TypeCheckError, check_type
from typing_extensions import TypeForm

DIM = 3
type Real = float | np.floating
type Point = Float[np.ndarray, "coordinates"]  # noqa: F821
type Grid = Float[np.ndarray, "points coordinates"]
type ScalarValues = Float[np.ndarray, "points"]  # noqa: F821
type VectorValues = Float[np.ndarray, "points components"]
type ScalarField = Callable[[Point | Grid], Real | ScalarValues]
type VectorField = Callable[[Point | Grid], Point | VectorValues]


def instance[TypeHint](arg: Any, hint: TypeForm[TypeHint] | Any) -> TypeIs[TypeHint]:
    """Check if a variable is an instance of the given type hint.

    Current implementation is restricted to acceptable `Type` hints.

    Args:
        arg: variable to check
        hint: type hint to check against

    Returns:
        bool: True if arg is an instance of hint, False otherwise
    """
    try:
        check_type(arg, hint)
        return True
    except TypeCheckError:
        return False
