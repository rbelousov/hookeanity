"""Numerical methods for spherical coordinates over a unit ball"""

from itertools import product
from typing import TypeAlias
from collections.abc import Callable, Sequence, Iterator
import numpy as np
from numpy.typing import NDArray, ArrayLike
from .radial import RadialPolynomial
from .angular import SphericalHarmonic, MaternInterpolation

Float: TypeAlias = float | np.floating
ScalarValues: TypeAlias = NDArray[np.floating]


class Interpolation:
    """Expansion of a scalar-valued function in spherical coordinates"""

    def __init__(self, order: tuple[int, int], basis: None | tuple[int, int] = None):
        """Initialize expansion of a given order

        Args:
            order: Order of the spectral accuracy
            basis: Number of radial and angular basis functions

        """
        self._radii, self._rws = RadialPolynomial.quadrature(order[0])
        self._angles, self._aws = SphericalHarmonic.quadrature(order[1])
        self._radial_functions = [
            RadialPolynomial(n)
            for n in range(order[0] + 1 if basis is None else basis[0])
        ]
        self._angular_functions = [
            SphericalHarmonic((l, m))
            for l in range(order[1] + 1 if basis is None else basis[1])
            for m in range(-l, l + 1)
        ]
        self._matern = MaternInterpolation(self._angles, weights=self._aws)

    def points(self) -> Iterator[NDArray[np.floating]]:
        """Return generator of quadrature points"""
        for pnt in product(self._radii, self._angles):
            r = pnt[0]
            theta, phi = pnt[1]
            yield np.array([r, theta, phi])

    def points_and_weights(self) -> Iterator[tuple[NDArray[np.floating], np.floating]]:
        """Return generator of quadrature points and weights"""
        points = product(self._radii, self._angles)
        weights = product(self._rws, self._aws)
        for pnt, wgt in zip(points, weights):
            r = pnt[0]
            theta, phi = pnt[1]
            yield np.array([r, theta, phi]), wgt[0] * wgt[1]

    # @staticmethod
    # def basis_function(
    #     poly: RadialPolynomial, sph: SphericalHarmonic
    # ) -> ScalarFunction:
    #     """Return wrapper of radial and angular functions"""

    #     def wrapper(r, theta, phi):
    #         return poly([r])[0] * sph([[theta, phi]])[0]

    #     return wrapper

    # def basis_functions(self) -> Iterator[ScalarFunction]:
    #     """Return generator of basis functions"""
    #     for poly, sph in product(self._radial_functions, self._angular_functions):
    #         yield self.basis_function(poly, sph)

    def transform(self) -> NDArray[np.floating]:
        """Return matrix that transforms sampled values to expansion coefficients"""
        polys = np.array([p(self._radii) for p in self._radial_functions])

        return np.array(
            [
                [
                    wn * wp * poly([rn])[0] * (1 if q == p else 0) / np.sqrt(wq)
                    for wn, rn in zip(self._rws, self._radii)
                    for p, wp in enumerate(self._aws)
                ]
                for poly in self._radial_functions
                for q, wq in enumerate(self._aws)
            ]
        )

    def inverse(self) -> NDArray[np.floating]:
        """Return matrix that transforms expansion coefficients into `ScalarValues`"""
        return np.array(
            [
                [
                    pk([rn])[0] * (1 if p == q else 0) / np.sqrt(wq)
                    for pk in self._radial_functions
                    for q, wq in enumerate(self._angles)
                ]
                for rn in self._radii
                for p in range(len(self._angles))
            ]
        )

    def partial_r(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the radial component"""
        poly = np.array([p(self._radii) for p in self._radial_functions])
        poly_r = np.array([p(self._radii, True) for p in self._radial_functions])
        sph = np.array([s(self._angles) for s in self._angular_functions])
        return np.array(
            [
                [
                    np.einsum("a,b,a,b,a,b", p1, s1, p2, s2, self._rws, self._aws)
                    for p2, s2 in product(poly_r, sph)
                ]
                for p1, s1 in product(poly, sph)
            ]
        )

    def partial_theta(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the polar component"""
        poly = np.array([p(self._radii) for p in self._radial_functions])
        sph = np.array([s(self._angles) for s in self._angular_functions])
        sph_theta = np.array(
            [s.partial_theta(self._angles) for s in self._angular_functions]
        )
        return np.array(
            [
                [
                    np.einsum("a,b,a,b,a,b", p1, s1, p2, s2, self._rws, self._aws)
                    for p2, s2 in product(poly, sph_theta)
                ]
                for p1, s1 in product(poly, sph)
            ]
        )

    def partial_phi(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the azimuthal component"""
        poly = np.array([p(self._radii) for p in self._radial_functions])
        sph = np.array([s(self._angles) for s in self._angular_functions])
        sph_phi = np.array(
            [s.partial_phi(self._angles) for s in self._angular_functions]
        )
        return np.array(
            [
                [
                    np.einsum("a,b,a,b,a,b", p1, s1, p2, s2, self._rws, self._aws)
                    for p2, s2 in product(poly, sph_phi)
                ]
                for p1, s1 in product(poly, sph)
            ]
        )

    @property
    def size(self) -> int:
        """Return number of modes"""
        return len(self._radial_functions) * len(self._angular_functions)

    @property
    def npoints(self) -> int:
        """Return number of quadrature points"""
        return len(self._radii) * len(self._angles)
