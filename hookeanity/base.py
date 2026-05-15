"""Basic types.

Attributes:
    DIM (int): Number of spatial dimensions
    TypeHint (typing.TypeVar): Type variable for hints
    Real (typing.TypeAlias): Native python or numpy floating-point number
    Point (typing.TypeAlias): Point coordinates in 3D
    Grid (typing.TypeAlias): Grid of 3D point coordinates
    ScalarField (typing.TypeAlias): Scalar field defined over a grid
    VectorField (typing.TypeAlias): Vector field defined over a grid
    ScalarValues (typing.TypeAlias): Scalar values on a mesh
    VectorValues (typing.TypeAlias): Vector values on a mesh
"""

from typing import Any, TypeAlias, TypeVar
from typing_extensions import TypeForm, TypeIs
from collections.abc import Callable
from typeguard import check_type, TypeCheckError
from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray


DIM = 3
TypeHint = TypeVar("TypeHint")
Real: TypeAlias = float | np.floating
Point: TypeAlias = Float[NDArray[np.floating], "coordinates"]  # noqa: F722, F821
Grid: TypeAlias = Float[NDArray[np.floating], "points coordinates"]  # noqa: F722, F821
ScalarValues: TypeAlias = Float[NDArray[np.floating], "points"]  # noqa: F722, F821
VectorValues: TypeAlias = Float[NDArray[np.floating], "points components"]  # noqa: F722, F821
ScalarField: TypeAlias = Callable[[Point | Grid], Real | ScalarValues]
VectorField: TypeAlias = Callable[[Point | Grid], Point | VectorValues]


def instance(arg: Any, hint: TypeForm[TypeHint] | Any) -> TypeIs[TypeHint]:
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
