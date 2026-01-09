"""Transformed Jacobi polynomials for the radial component"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.special import gamma, eval_jacobi


def jacobi_radau_quadrature(
    num: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Gauss–Radau quadrature on [-1, 1] for (0,1)-Jacobi weights with the fixed node at x = 1

    Args:
        num (int): Number of points

    Returns:
        Nodes and weights of the quadrature

    Raises:
        AssertionError: One `num < 1`

    """
    assert num >= 1, f"Nonpositive number of points ({num})"

    a, b = 0.0, 1.0
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

    return x, w


class RadialPolynomial:
    """Orthonormalized Jacobi (0,1)-polynomial for the radial component of the spherical coordinates

    Attributes:
        _order (int): Polynomial order
        _prefactor (numpy.floating): Normalization prefactor

    """

    def __init__(self, order: int):
        """Initialize radial polynomial of a given order

        Args:
            order (int): Polynomial order

        """
        self._order = order
        self._prefactor = np.sqrt(2 * (order + 1))

    def __call__(self, r: ArrayLike, derivative: bool = False) -> NDArray[np.floating]:
        """Evaluate the radial polynomial

        Args:
            r: Grid or radial values within the interval [0, 1]
            derivative: When `True`, evaluate the derivative of the radial polynomial

        Returns:
            Values of the radial polynomial or its derivative

        """
        if self._order == 0:
            return np.zeros_like(r) if derivative else np.full_like(r, self._prefactor)

        x = 2 * np.asarray(r) - 1
        if derivative:
            return (
                self._prefactor
                * (self._order + 2)
                * eval_jacobi(self._order - 1, 1, 2, x)
            )
        return self._prefactor * eval_jacobi(self._order, 0, 1, x)

    @staticmethod
    def quadrature(
        spectral_order: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return radial quadrature

        Args:
            spectral_order: Order of spectral accuracy

        Returns:
            Grid of quadrature nodes and weights

        """
        nodes, weights = jacobi_radau_quadrature(spectral_order + 1)

        return (nodes + 1) / 2, weights / 4
