"""Pseudospectral methods for spherical coordinates over a unit ball"""

from itertools import product
from typing import TypeAlias
from collections.abc import Callable, Sequence, Iterator
import numpy as np
from numpy.typing import NDArray, ArrayLike
from .radial import RadialPolynomial
from .angular import SphericalHarmonic

Float: TypeAlias = float | np.floating
ScalarFunction: TypeAlias = Callable[[Float, Float, Float], Float]
VectorFunction: TypeAlias = Sequence[ScalarFunction]


class Expansion:
    """Expansion of a vector-valued function in spherical coordinates"""

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

    @staticmethod
    def basis_scalar(poly: RadialPolynomial, sph: SphericalHarmonic) -> ScalarFunction:
        """Return wrapper of radial and angular functions"""

        def wrapper(r, theta, phi):
            return poly([r])[0] * sph([[theta, phi]])[0]

        return wrapper

    def basis_functions(self) -> Iterator[ScalarFunction]:
        """Return generator of basis functions"""
        for poly, sph in product(self._radial_functions, self._angular_functions):
            yield self.basis_scalar(poly, sph)

    def to_scalar(
        self, coefficients: NDArray[np.floating], copy=True
    ) -> ScalarFunction:
        """Create a scalar function identified by its expansion coefficients

        Args:
            coefficients: Values of the coefficients
            copy: When `True`, the values of the coefficients are copied

        Returns:
            A function identified by its expansion coefficients

        """
        cfs = np.copy(coefficients) if copy else coefficients

        def wrapper(r, theta, phi):
            return np.sum(
                [cf * bf(r, theta, phi) for cf, bf in zip(cfs, self.basis_functions())]
            )

        return wrapper

    def expand_scalar(self, scalar: ScalarFunction) -> NDArray[np.floating]:
        """Expand a scalar function in the pseudospectral basis

        Args:
            scalar: Scalar function to expand

        Returns:
            Expansion coefficients

        """
        return np.array(
            [
                np.sum(
                    [
                        wgt * scalar(*pnt) * bf(*pnt)
                        for pnt, wgt in self.points_and_weights()
                    ]
                )
                for bf in self.basis_functions()
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
