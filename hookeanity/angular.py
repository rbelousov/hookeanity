"""Real spherical harmonics for the solid-angle components of spherical reference frame"""

import numpy as np
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
    def from_spectral_order(spectral_order: int) -> int:
        """Find best order of the quadrature for a given spectral accuracy"""
        n = 2 * spectral_order + 1
        for order in sorted(getattr(PyLebedev(), "_PyLebedev__datasource").keys()):
            num = int(order)
            if num >= n:
                return num
        raise ValueError(
            f"No quadrature found for the spectral accuracy of order {spectral_order}"
        )

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


class HarmonicInterpolation:
    """Interpolation of functions with a localized Lagrange kernel based on spherical harmonics"""

    def __init__(self, order: int):
        """Initialize interpolating functions of a given order"""
        self._nodes, self._weights = SphericalHarmonic.quadrature(order)

        lmax = np.sqrt(len(self._nodes)).astype(int) + 1
        self._basis = [
            SphericalHarmonic((l, m)) for l in range(lmax + 1) for m in range(-l, l + 1)
        ]

        self._values = np.array([sh(self._nodes) for sh in self._basis])
        self._matrix = np.einsum("yp,yq", self._values, self._values)
        self._inverse = np.linalg.inv(self._matrix)

    def __getitem__(self, index):
        """Get Lagrange function"""

        def wrapper(angles: ArrayLike) -> NDArray[np.floating]:
            return np.einsum(
                "q,yq,yp",
                self._inverse[index],
                self._values,
                [sh(angles) for sh in self._basis],
            )

        return wrapper

    def __len__(self):
        """Return number of Lagrange kernels"""
        return len(self._nodes)
