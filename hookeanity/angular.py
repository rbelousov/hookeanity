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
        return prefactor * np.sin(-order * phi)
    return prefactor * np.cos(order * phi)


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
            sph_legendre_p(self._order[0], self._order[1], grid[..., 0])
            * fourier(self._order[1], grid[..., 1])
        ).reshape(-1)

    def partial_theta(self, grid: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the polar angle"""
        grid = np.asarray(grid)
        _, partial = sph_legendre_p(
            self._order[0], self._order[1], grid[..., 0], diff_n=1
        )
        return (partial * fourier(self._order[1], grid[..., 1])).reshape(-1)

    def partial_phi(self, grid: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the azimuthal angle"""
        grid = np.asarray(grid)
        return (
            sph_legendre_p(self._order[0], self._order[1], grid[..., 0])
            * fourier(self._order[1], grid[..., 1], derivative=True)
        ).reshape(-1)

    @staticmethod
    def from_spectral_order(spectral_order: int) -> int:
        """Find best order of the quadrature for a given spectral accuracy"""
        n = 2 * spectral_order
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
        return nodes[:, [1, 0]], 4 * np.pi * weights
