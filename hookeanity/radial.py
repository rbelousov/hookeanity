"""Transformed Jacobi polynomials for the radial component"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.special import gamma, eval_jacobi, roots_jacobi


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


def jacobi_lobatto_quadrature(
    num: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Jacobi–Lobatto quadrature on [-1, 1] for (0,1)-Jacobi weights with endpoints x=-1 and x=+1 included

    Args:
        num (int): Number of quadrature points

    Returns:
        Nodes and weights of the quadrature points

    Raises:
        AssertionError: if num < 2

    """
    assert num >= 2, f"No Lobatto quadrature rule with {num} points"

    # Jacobi parameters
    a, b = 0.0, 1.0

    # Zeroth moment (integral of the weight over [-1,1])
    mu0 = 2.0 ** (a + b + 1.0) * gamma(a + 1.0) * gamma(b + 1.0) / gamma(a + b + 2.0)

    # First moment for Jacobi weight with the mean is (b-a)/(a+b+2) yields mu1 = mu0 * mean
    mu1 = mu0 * (b - a) / (a + b + 2.0)

    # The pure endpoint Lobatto rule (no interior points)
    if num == 2:
        # Solve:
        #   wL + wR = mu0
        #  -wL + wR = mu1
        wR = 0.5 * (mu0 + mu1)
        wL = 0.5 * (mu0 - mu1)
        return np.array([-1.0, 1.0], dtype=float), np.array([wL, wR], dtype=float)

    # num >= 3: endpoints plus m interior nodes
    m = num - 2

    # Interior nodes are the roots of P_m^{(a+1,b+1)}(x)
    z, _ = roots_jacobi(m, a + 1.0, b + 1.0)

    # Interior weights (Petrova 202x, Thm 4.1; r=k=0 specialization):
    alpha = a + 1.0
    beta = b + 1.0

    delta = (
        2.0 ** (alpha + beta - 1.0)
        * gamma(m + alpha + 1.0)
        * gamma(m + beta + 1.0)
        * (2.0 * m + alpha + beta + 2.0) ** 2
        / (
            gamma(m + 1.0)
            * gamma(m + alpha + beta + 1.0)
            * (m + 1.0) ** 2
            * (m + alpha + beta + 1.0) ** 2
        )
    )

    Pm1 = eval_jacobi(m + 1, alpha, beta, z)
    w_int = delta / (Pm1**2)

    # Endpoint weights: determine them from exactness for degree 0 and 1
    sum_int = float(np.sum(w_int))
    sumx_int = float(np.sum(w_int * z))

    rhs0 = mu0 - sum_int
    rhs1 = mu1 - sumx_int

    # Solve:
    #   wL + wR = rhs0
    #  -wL + wR = rhs1
    wR = 0.5 * (rhs0 + rhs1)
    wL = 0.5 * (rhs0 - rhs1)

    x = np.concatenate(([-1.0], z, [1.0])).astype(float)
    w = np.concatenate(([wL], w_int, [wR])).astype(float)
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
