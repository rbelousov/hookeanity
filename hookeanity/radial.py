"""Numerical methods for functions of the radial component in spherical coordinates."""

from typing import cast
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.special import gamma, eval_jacobi
from .base import Real, instance


def jacobi_radau_quadrature(
    num: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Gauss–Jacobi-Radau quadrature with the fixed node at x = 1 for (0,2) Jacobi polynomials.

    The quadarature is defined on the interval [-1, 1] with the weighting function of the (0,2) Jacobi polynomials.

    Args:
        num: Number of quadtrature points.

    Returns:
        Quadrature nodes and weights of the quadrature.

    Raises:
        AssertionError: On `num < 1`.
    """
    assert num >= 1, f"nonpositive number of points ({num})"

    a, b = 0.0, 2.0
    x0 = 1.0
    mu0 = 2.0 ** (a + b + 1.0) * gamma(a + 1.0) * gamma(b + 1.0) / gamma(a + b + 2.0)

    if num == 1:
        return np.array([x0], dtype=float), np.array([mu0], dtype=float)

    # Recurrence
    a_rec = np.zeros(num + 1, dtype=float)
    b_rec = np.zeros(num + 1, dtype=float)
    for k in range(num + 1):
        if k == 0:
            a_rec[k] = (b - a) / (a + b + 2.0)
        else:
            a_rec[k] = (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2.0))

    b_rec[0] = 0.0
    for k in range(1, num + 1):
        numerator = 4.0 * k * (k + a) * (k + b) * (k + a + b)
        denominator = (
            (2.0 * k + a + b) ** 2 * (2.0 * k + a + b + 1.0) * (2.0 * k + a + b - 1.0)
        )
        b_rec[k] = numerator / denominator

    pi_m1 = 0.0
    pi_0 = 1.0
    vals = [pi_0]
    for k in range(num):  # produce π_1 ... π_n
        pi_p1 = (x0 - a_rec[k]) * pi_0 - b_rec[k] * pi_m1
        pi_m1, pi_0 = pi_0, pi_p1
        vals.append(pi_0)

    pi_n = vals[num]
    pi_nm1 = vals[num - 1]

    # Modify the last diagonal so that x0 is an eigenvalue (Radau condition)
    delta = pi_n / pi_nm1

    # Build the symmetric Jacobi matrix
    mtx = np.diag(a_rec[:num].copy())
    off = np.sqrt(b_rec[1:num])
    for i in range(num - 1):
        mtx[i, i + 1] = off[i]
        mtx[i + 1, i] = off[i]

    # Radau modification
    mtx[-1, -1] = a_rec[num - 1] + delta

    # Golub–Welsch: eigenvalues are nodes; weights from first eigenvector component
    x, vecs = np.linalg.eigh(mtx)
    w = mu0 * (vecs[0, :] ** 2)
    x[np.argmax(x)] = 1.0

    return x, w


class RadialPolynomial:
    """Jacobi (0,2)-polynomial shifted and orthonormalized on the interval [0, 1].

    These polynomials are suitable for numerical integration and differentiation with respect to the radial component of the spherical coordinates.

    Attributes:
        _order (int): Polynomial order.
        _prefactor (numpy.floating): Normalization prefactor.
    """

    def __init__(self, order: int):
        """Initialize radial polynomial of a given order.

        Args:
            order (int): Polynomial order.
        """
        self._order = order
        self._prefactor = np.sqrt(2 * order + 3)

    def __call__(
        self, r: ArrayLike | Real, derivative: bool = False
    ) -> NDArray[np.floating] | Real:
        """Evaluate the radial polynomial.

        Args:
            r: Values of the radial component within the interval [0, 1].
            derivative: When `True`, evaluate the derivative of the radial polynomial.

        Returns:
            Values of the radial polynomial or its derivative.
        """
        r_is_real = instance(r, Real)
        if self._order == 0:
            if r_is_real:
                return type(r)(0.0 if derivative else self._prefactor)
            return np.zeros_like(r) if derivative else np.full_like(r, self._prefactor)

        x = 2 * cast(float, r) - 1 if r_is_real else 2 * np.asarray(r) - 1
        if derivative:
            result = (
                self._prefactor
                * (self._order + 3)
                * eval_jacobi(self._order - 1, 1, 3, x)
            )
            return type(r)(result) if r_is_real else result
        result = self._prefactor * eval_jacobi(self._order, 0, 2, x)
        return type(r)(result) if r_is_real else result

    @staticmethod
    def quadrature(
        num: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return radial quadrature.

        Args:
            num: Number of points.

        Returns:
            Quadrature nodes and weights.
        """
        nodes, weights = jacobi_radau_quadrature(num)

        return (nodes + 1) / 2, weights / 8
