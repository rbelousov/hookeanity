"""Real spherical harmonics for the solid-angle components of spherical reference frame"""

import numpy as np
from collections.abc import Iterator
from itertools import product
from numpy.typing import NDArray, ArrayLike
from scipy.special import sph_legendre_p
from pylebedev import PyLebedev


def fourier(
    order: int, phi: ArrayLike, derivative: bool = False
) -> NDArray[np.floating]:
    """Evaluate real Fourier function of a given order

    Args:
        order: Order of the function
        phi: Values of the argument
        derivative: When `True`, evaluate the derivative of the Fourier function

    Returns:
        Value of the Fourier function or its derivative

    """
    if order == 0:
        return (
            np.zeros_like(phi, dtype=float)
            if derivative
            else np.ones_like(phi, dtype=float)
        )

    phi = np.asarray(phi)
    prefactor = np.sqrt(2)
    if derivative:
        order = -order
        prefactor *= order

    if order < 0:
        return prefactor * np.sin(-order * phi) * (1 if order % 2 == 0 else -1)
    return prefactor * np.cos(order * phi) * (1 if order % 2 == 0 else -1)


class SphericalHarmonic:
    """Real-valued spherical harmonics

    Attributes:
        _order (tuple[int, int]): Order (l, m) of the harmonics

    """

    def __init__(self, order: tuple[int, int]):
        """Initialize spherical harmonic of a given order

        Args:
            order: Order of the harmonic

        """
        assert abs(order[1]) <= order[0], (
            f"Azimuthal-angle order ({order[1]}) exceeds the polar order ({order[0]})"
        )
        self._order = order

    def __call__(self, grid: ArrayLike) -> NDArray[np.floating]:
        """Return values of the spherical harmonic on a grid

        Args:
            grid: Grid of solid angles

        Returns:
            Values of the function

        """
        grid = np.asarray(grid)
        return (
            sph_legendre_p(self._order[0], abs(self._order[1]), grid[..., 0])
            * fourier(self._order[1], grid[..., 1])
        ).reshape(-1)

    def partial_theta(self, grid: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the polar angle"""
        grid = np.asarray(grid)
        _, partial = sph_legendre_p(
            self._order[0], abs(self._order[1]), grid[..., 0], diff_n=1
        )
        return (partial * fourier(self._order[1], grid[..., 1])).reshape(-1)

    def partial_phi(self, grid: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the azimuthal angle"""
        grid = np.asarray(grid)
        return (
            sph_legendre_p(self._order[0], abs(self._order[1]), grid[..., 0])
            * fourier(self._order[1], grid[..., 1], derivative=True)
        ).reshape(-1)

    @staticmethod
    def find_order(lmax: int) -> int:
        """Find a tabulated order of the Lebedev quadrature that integrates spherical harmonics up to `lmax`"""
        for order in sorted(getattr(PyLebedev(), "_PyLebedev__datasource").keys()):
            num = int(order)
            if num >= lmax:
                return num
        raise ValueError(
            f"No quadrature found for the spectral accuracy of order {lmax}"
        )

    @staticmethod
    def from_spectral_order(spectral_order: int) -> int:
        """Find best order of the quadrature for a given spectral accuracy"""
        return SphericalHarmonic.find_order(2 * spectral_order + 1)

    @staticmethod
    def quadrature(
        spectral_order: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return Lebedev quadrature rules

        Note that weights of the Lebedev quadrature are already multiplied by the four-pi factor.

        Args:
            spectral_order: Desired spectral order of Lebedev quadrature

        Returns:
            Angular grid of solid angles (theta, phi) and their weights

        """
        lebedev = PyLebedev()
        nodes, weights = lebedev.get_points_and_weights(
            SphericalHarmonic.from_spectral_order(spectral_order), solid_angles=True
        )
        nodes = nodes[:, [1, 0]]
        nodes[:, 1] += np.pi
        return nodes, 4 * np.pi * weights


def geodesic_cos(angles_a: ArrayLike, angles_b: ArrayLike) -> NDArray[np.floating]:
    """Compute cosine of the geodesic distance between two sets of angles on a unit sphere

    Args:
        angles_a: First set of solid angles
        angles_b: Second set of solid angles

    Returns:
        Cosine of the geodesic distance between the angles

    """
    angles_a = np.asarray(angles_a)
    angles_b = np.asarray(angles_b)

    cos_theta_a = np.cos(angles_a[..., 0])
    sin_theta_a = np.sin(angles_a[..., 0])
    cos_theta_b = np.cos(angles_b[..., 0])
    sin_theta_b = np.sin(angles_b[..., 0])

    delta_phi = angles_a[..., 1] - angles_b[..., 1]
    cos_delta_phi = np.cos(delta_phi)

    cos_gamma = cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b * cos_delta_phi
    return np.clip(cos_gamma, -1.0, 1.0)


class MaternInterpolation:
    """Sobolev-Matérn interpolation of functions"""

    def __init__(
        self,
        nodes: NDArray[np.floating],
        smoothness: float = 3.0,
        lmax: int | None = None,
        weights: NDArray[np.floating] | None = None,
    ):
        """Initialize interpolation grid

        Smoothness parameter 3 should yield accurate second-order derivatives.

        Args:
            nodes: Nodes of interpolation
            smoothness: Smoothness of the interpolation
            lmax: Order of kernel expansion defaults to the square root of the number of particles
            weights: Optional quadrature weights

        """
        self.lmax = np.sqrt(len(nodes)).astype(int) if lmax is None else lmax
        basis = list(self.basis())

        self._bandfilter = np.array(
            [
                (1 + l * (l + 1)) ** -(1 + smoothness)
                for l in range(self.lmax + 1)
                for m in range(-l, l + 1)
            ]
        )
        self._values = np.array([sh(nodes) for sh in basis])
        self._theta = np.array([sh.partial_theta(nodes) for sh in basis])
        self._phi = np.array([sh.partial_phi(nodes) for sh in basis])
        self._weights = (
            np.ones(len(nodes), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )

        matrix = np.einsum("y,yp,yq", self._bandfilter, self._values, self._values)
        self._coefficients = np.linalg.pinv(matrix)

    def basis(self) -> Iterator[SphericalHarmonic]:
        """Return generator of basis functions"""
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                yield SphericalHarmonic((l, m))

    def __getitem__(self, index):
        """Get Lagrange function"""

        def wrapper(angles: ArrayLike) -> NDArray[np.floating]:
            return np.einsum(
                "y,q,yq,yp",
                self._bandfilter,
                self._coefficients[index],
                self._values,
                [sh(angles) for sh in self.basis()],
            ) / np.sqrt(self._weights[index])

        return wrapper

    def __len__(self):
        """Return number of Lagrange kernels"""
        return self._values.shape[1]

    def partial_theta(self):
        """Return matrix form of the derivative operator with respect tho the polar angle"""
        return np.einsum(
            "y,yr,yq,pq,p->rp",
            self._bandfilter,  # y
            self._theta,  # y r
            self._values,  # y q
            self._coefficients,  # p q
            1 / np.sqrt(self._weights),
        )

    def partial_phi(self):
        """Return matrix form of the derivative operator with respect tho the polar angle"""
        return np.einsum(
            "y,yr,yq,pq,p->rp",
            self._bandfilter,  # y
            self._phi,  # y r
            self._values,  # y q
            self._coefficients,  # p q
            1 / np.sqrt(self._weights),
        )


class LocalFiniteDifferences:
    """Method of local finite differences"""

    def __init__(self, shape: tuple[int, int]):
        """Initialize interpolating functions of a given order"""
        assert shape[0] >= 2 and shape[1] >= 4, "Shape must be at least (2, 4)"
        self._shape = (int(shape[0]), int(shape[1]))
        self._weights = np.pi * np.array([1, 2]) / np.array(shape)
        self._thetas = (np.arange(shape[0], dtype=float) + 0.5) * self._weights[0]
        self._phis = np.arange(shape[1], dtype=float) * self._weights[1]
        self._nodes = np.array(
            [[theta, phi] for theta in self._thetas for phi in self._phis]
        )

    def partial_theta(self):
        """Return derivative operator with respect to the polar angle"""

        def element(row, col):
            if row[1] != col[1]:
                return 0.0

            if row[0] == 0:
                if col[0] == row[0]:
                    return -1.0
                elif col[0] == row[0] + 1:
                    return 1.0
            elif row[0] == self._shape[0] - 1:
                if col[0] == row[0]:
                    return 1.0
                elif col[0] == row[0] - 1:
                    return -1.0
            elif row[0] == col[0] + 1:
                return 0.5
            elif row[0] == col[0] - 1:
                return -0.5

            return 0.0

        return (
            np.array(
                [
                    [
                        element(row, col)
                        for col in product(range(self._shape[0]), range(self._shape[1]))
                    ]
                    for row in product(range(self._shape[0]), range(self._shape[1]))
                ]
            )
            / self._weights[0]
        )
