"""Basis types"""

from jaxtyping import Float
from typing import TypeAlias, Any
from collections.abc import Callable, Sequence, Iterator
import numpy as np
from numpy.typing import NDArray, ArrayLike

Real: TypeAlias = float | np.floating
ScalarValues: TypeAlias = Float[NDArray[np.floating], "mesh"]
VectorValues: TypeAlias = Float[NDArray[np.floating], "mesh coordinate"]


def is_real(arg: Any) -> bool:
    """Check if a variable is instances of `Real`"""
    return isinstance(arg, (float, np.floating))
