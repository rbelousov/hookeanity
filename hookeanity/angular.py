"""Real spherical harmonics for the solid-angle components of spherical reference frame"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.special import sph_legendre_p
from pylebedev import PyLebedev


class SphericalHarmonic:
    """Real-valued spherical harmonics

    Attributes:
        _order (tuple[int, int]): Order (l, m) of the harmonics
        _fourier (Callable[[np.Array], numpy.typing.NDArray[numpy.floating]]): Fourier basis functions

    """

    def __init__(self, order: tuple[int, int]):
        """Initialize spherical harmonic of a given order

        Args:
            order: Order of the harmonic

        """
        assert abs(order[1]) <= order[0], (
            f"Polar-angle order ({order[1]}) exceeds the azimuthal order ({order[0]})"
        )
        self._order = order
        self._fourier = (
            (lambda x: np.full_like(x, 1.0))
            if order[1] == 0
            else (
                (lambda x: np.sqrt(2) * np.sin(-self._order[1] * x))
                if order[1] < 0
                else (lambda x: np.sqrt(2) * np.cos(self._order[1] * x))
            )
        )

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
            * self._fourier(grid[..., 1])
        ).reshape(-1)

    @staticmethod
    def quadrature(order: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return Lebedev quadrature rules

        Note that weights of the Lebedev quadrature are already multiplied by the four-pi factor.

        Args:
            order: Order of Lebedev quadrature

        Returns:
            Angular grid of solid angles (theta, phi) and their weights

        """
        lebedev = PyLebedev()
        nodes, weights = lebedev.get_points_and_weights(order, solid_angles=True)
        return nodes[:, [1, 0]], 4 * np.pi * weights
