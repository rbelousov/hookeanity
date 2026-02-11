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
ScalarValues: TypeAlias = NDArray[np.floating]


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
        self._radial_values = np.array(
            [poly(self._radii) for poly in self._radial_functions]
        )
        self._angular_values = np.array(
            [sph(self._angles) for sph in self._angular_functions]
        )

        self._bfs = np.einsum(
            "pr,sa->psra", self._radial_values, self._angular_values
        ).reshape(self.size, self.npoints)
        self._wts = (self._rws[:, None] * self._aws[None, :]).ravel()

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
    def basis_function(
        poly: RadialPolynomial, sph: SphericalHarmonic
    ) -> ScalarFunction:
        """Return wrapper of radial and angular functions"""

        def wrapper(r, theta, phi):
            return poly([r])[0] * sph([[theta, phi]])[0]

        return wrapper

    def basis_functions(self) -> Iterator[ScalarFunction]:
        """Return generator of basis functions"""
        for poly, sph in product(self._radial_functions, self._angular_functions):
            yield self.basis_function(poly, sph)

    def to_scalar_function(
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
    
    def to_scalar_values(self, coefficients, point_indices=[]):
        """Returns the values of the approximated function on the pseudo-spectral grid"""
        return np.einsum("b,bp", coefficients, self._bfs[:, point_indices])

    def sample_scalar_function(self, scalar: ScalarFunction) -> ScalarValues:
        """Sample scalar function over the pseudospectral grid"""
        return np.array([scalar(*pnt) for pnt in self.points()])

    def expand_scalar_values(self, scalar: ScalarValues) -> NDArray[np.floating]:
        """Expand in the pseudospectral basis a scalar function defined by its values on the quadrature point

        Args:
            scalar: Values of the scalar function at the quadrature points

        Returns:
            Expansion coefficients

        """
        return np.einsum("bp,p,p", self._bfs, self._wts, scalar)

    def expand_scalar_function(self, scalar: ScalarFunction) -> NDArray[np.floating]:
        """Expand a scalar function in the pseudospectral basis

        Args:
            scalar: Scalar function to expand

        Returns:
            Expansion coefficients

        """
        return self.expand_scalar_values(self.sample_scalar_function(scalar))
    
    def _bfs_r(self):
        """Return basis functions derived with respect to r and evaluated on the points of the pseudo-spectral grid"""
        poly_r = np.array([p(self._radii, True) for p in self._radial_functions])
        return np.einsum("pr,sa->psra", poly_r, self._angular_values).reshape(self.size, self.npoints)
    
    def _bfs_theta(self):
        """Return basis functions derived with respect to theta and evaluated on the points of the pseudo-spectral grid"""
        sph_theta = np.array(
            [s.partial_theta(self._angles) for s in self._angular_functions]
        )
        return np.einsum("pr,sa->psra", self._radial_values, sph_theta).reshape(self.size, self.npoints)
    
    def _bfs_phi(self):
        """Return basis functions derived with respect to phi and evaluated on the points of the pseudo-spectral grid"""
        sph_phi = np.array(
            [s.partial_phi(self._angles) for s in self._angular_functions]
        )
        return np.einsum("pr,sa->psra", self._radial_values, sph_phi).reshape(self.size, self.npoints)

    def partial_r(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the radial component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_r())

    def partial_theta(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the polar component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_theta())

    def partial_phi(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the azimuthal component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_phi())

    # def partial_phi(self) -> NDArray[np.floating]:
    #     """Return matrix form of the partial-derivative operator with respect to the azimuthal component"""
    #     poly = np.array([p(self._radii) for p in self._radial_functions])
    #     sph = np.array([s(self._angles) for s in self._angular_functions])
    #     sph_phi = np.array(
    #         [s.partial_phi(self._angles) for s in self._angular_functions]
    #     )
    #     return np.array(
    #         [
    #             [
    #                 np.einsum("a,b,a,b,a,b", p1, s1, p2, s2, self._rws, self._aws)
    #                 for p2, s2 in product(poly, sph_phi)
    #             ]
    #             for p1, s1 in product(poly, sph)
    #         ]
    #     )

    # def _jacobian(self, xi, theta, phi, eps=1e-12):
    #     """Return the jacobian matrix evaluated on a point in spherical coordinates"""
    #     return np.hstack((
    #         self._jacobian_x(xi, theta, phi, eps), 
    #         self._jacobian_y(xi, theta, phi, eps), 
    #         self._jacobian_z(xi, theta, phi, eps)
    #         ))

    def _jacobian_x(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to x of the transformation of coordinates on a point in spherical coordinates"""
        jx = np.zeros((3, 1))
        jx[0] = np.sin(theta) * np.cos(phi)
        jx[1] = np.cos(theta) * np.cos(phi) / xi if np.abs(xi) > eps else 0.0
        jx[2] = - np.sin(phi) / (xi * np.sin(theta)) if np.abs(xi * np.sin(theta)) > eps else 0.0
        return jx

    def _jacobian_y(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to y of the transformation of coordinates on a point in spherical coordinates"""
        jy = np.zeros((3, 1))
        jy[0] = np.sin(theta) * np.sin(phi)
        jy[1] = np.cos(theta) * np.sin(phi) / xi if xi > eps else 0.0
        jy[2] = np.cos(phi) / (xi * np.sin(theta)) if (xi * np.sin(theta)) > eps else 0.0
        return jy

    def _jacobian_z(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to z of the transformation of coordinates on a point in spherical coordinates"""
        jz = np.zeros((3, 1))  # jz[2] = 0
        jz[0] = np.cos(theta)
        jz[1] = - np.sin(theta) / xi if np.abs(xi) > eps else 0.0 
        return jz 

    def partial_x(self):
        """Return matrix form of the partial-derivative operator with respect to the x component"""
        jxs = self.sample_scalar_function(self._jacobian_x)

        term1 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jxs[:, 0, 0], self._bfs_r())
        term2 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jxs[:, 1, 0], self._bfs_theta())
        term3 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jxs[:, 2, 0], self._bfs_phi())

        return term1 + term2 + term3
    
    def partial_y(self):
        """Return matrix form of the partial-derivative operator with respect to the y component"""
        jys = self.sample_scalar_function(self._jacobian_y)

        term1 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jys[:, 0, 0], self._bfs_r())
        term2 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jys[:, 1, 0], self._bfs_theta())
        term3 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jys[:, 2, 0], self._bfs_phi())

        return term1 + term2 + term3
    
    def partial_z(self):
        """Return matrix form of the partial-derivative operator with respect to the x component"""
        jzs = self.sample_scalar_function(self._jacobian_z)

        term1 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jzs[:, 0, 0], self._bfs_r())
        term2 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jzs[:, 1, 0], self._bfs_theta())
        term3 = np.einsum('p,ap,p,bp', self._wts, self._bfs, jzs[:, 2, 0], self._bfs_phi())

        return term1 + term2 + term3

    @property
    def size(self) -> int:
        """Return number of modes"""
        return len(self._radial_functions) * len(self._angular_functions)

    @property
    def npoints(self) -> int:
        """Return number of quadrature points"""
        return len(self._radii) * len(self._angles)
