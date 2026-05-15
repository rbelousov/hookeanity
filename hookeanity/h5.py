"""HDF5 utility functions.

Attributes:
    IntPair (typing.TypeAlias): Pair of integers.
    FloatPair (typing.TypeAlias): Pair of floating-point numbers.
"""

from typing import cast, Any, TypeAlias
import h5py
import numpy as np
from numpy.typing import NDArray

IntPair: TypeAlias = tuple[int, int]
FloatPair: TypeAlias = tuple[float, float]


def attr_any(file: h5py.File, key: str) -> Any:
    """Read an attribute from an HDF5 file handle.

    Args:
        file: File handle.
        key: Attribute name.

    Returns:
        Attribute value.
    """
    return cast(Any, file.attrs[key])


def attr_float(file: h5py.File, key: str) -> float:
    """Read floating-point attribute from an HDF5 file handle.

    Args:
        file: File handle.
        key: Attribute name.

    Returns:
        Attribute value.
    """
    return float(attr_any(file, key))


def attr_int(file: h5py.File, key: str) -> int:
    """Read integer attribute from an HDF5 file handle.

    Args:
        file: File handle.
        key: Attribute name.

    Returns:
        Attribute value.
    """
    return int(attr_any(file, key))


def attr_int_pair(file: h5py.File, key: str) -> IntPair:
    """Read pair of integer numbers from an HDF5 file handle attributes.

    Args:
        file: File handle.
        key: Attribute name.

    Returns:
        Pair of integers.

    Raises:
        ValueError: When the attribute values is malformed.
    """
    arr = np.asarray(attr_any(file, key))
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
    return (int(arr[0]), int(arr[1]))


def attr_float_pair(file: h5py.File, key: str) -> tuple[float, float]:
    """Read pair of floating-point numbers from an HDF5 file handle attributes.

    Args:
        file: File handle.
        key: Attribute name.

    Returns:
        Pair of floating-point numbers.

    Raises:
        ValueError: When the attribute values is malformed.
    """
    arr = np.asarray(attr_any(file, key))
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
    return (float(arr[0]), float(arr[1]))


def optional_nnint(value: int | None) -> int:
    """Return HDF5-safe representation of an optional nonnegative integer value.

    Args:
        value: Value to store in the HDF5 format.

    Returns:
        HDF5 representation of an optional nonnegative integer value.
    """
    return -1 if value is None else int(value)


def dataset_ndfloat(file: h5py.File, key: str) -> NDArray[np.floating]:
    """Load numpy array of floating-point numbers from an HDF5 file handle.

    Args:
        file: File handle.
        key: Dataset key.

    Returns:
        Numpy error loaded from the file.
    """
    dset = cast(h5py.Dataset, file[key])
    return np.array(dset[()], dtype=float)
