"""Numerical methods for spherical coordinates over a unit ball"""

from itertools import product
from typing import TypeAlias
from collections.abc import Callable, Sequence, Iterator
import numpy as np
from numpy.typing import NDArray, ArrayLike
from .radial import RadialPolynomial
from .angular import SphericalHarmonic, MaternInterpolation

Float: TypeAlias = float | np.floating
Values: TypeAlias = NDArray[np.floating]
ScalarFunction: TypeAlias = Callable[[Float, Float, Float], Float]
Function: TypeAlias = Callable[[Float, Float, Float], Float | Values]


def fibonacci_points(n: int) -> NDArray[np.floating]:
    """Distribute Fibonacci golden-angle points on a unit sphere avoiding poles

    Args:
        n: Number of points

    Returns:
        A grid of nearly uniform points

    """
    k = np.arange(n)

    # Golden angle
    gr = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (gr * gr)

    # Midpoint z-spacing avoiding poles
    z = 1.0 - (2.0 * k + 1.0) / n

    theta = np.arccos(z)  # in (0, pi)
    phi = ((k + 0.5) * golden_angle) % (2 * np.pi)

    return np.column_stack([theta, phi])


def reduced_gauss_legendre_grid_angles(N, min_lon=8):
    """Generate a reduced Gauss–Legendre grid as an array of angles.

    Parameters
    ----------
    N : int
        Number of Gaussian latitudes.
    min_lon : int
        Minimum number of longitude points near the poles.

    Returns
    -------
    angles : ndarray, shape (total_points, 2)
        Grid points: angles[i,0] = theta, angles[i,1] = phi (both in radians)
    nlon_per_lat : ndarray
        Number of longitude points at each latitude.
    weights : ndarray
        Gauss–Legendre quadrature weights for latitudes.

    """
    # Gaussian nodes and weights
    x, weights = np.polynomial.legendre.leggauss(N)

    # Convert nodes to colatitude theta (0 = North Pole, pi = South Pole)
    theta = np.arccos(x)

    # Max number of longitudes at equator
    max_lon = 2 * N

    # Reduced number of longitudes per latitude
    nlon_per_lat = np.maximum(min_lon, (max_lon * np.sin(theta)).astype(int))
    nlon_per_lat += nlon_per_lat % 2  # make even

    # Build the grid
    points = []
    for th, nlon in zip(theta, nlon_per_lat):
        phi = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        th_array = np.full(nlon, th)
        points.append(np.column_stack([th_array, phi]))

    angles = np.vstack(points)  # shape (total_points, 2)
    return angles, nlon_per_lat, weights


class Interpolation:
    """Expansion of a scalar-valued function in spherical coordinates"""

    def __init__(
        self,
        order: tuple[int, int],
        basis: None | tuple[int, int] = None,
        lmax: int | None = None,
    ):
        """Initialize expansion of a given order

        Args:
            order: Order of the spectral accuracy
            basis: Number of radial and angular basis functions
            lmax: Order of kernel expansion defaults to the square root of the number of particles

        """
        self._radii, self._rws = RadialPolynomial.quadrature(order[0])
        self._angles = fibonacci_points(
            order[1]
        )  # SphericalHarmonic.quadrature(order[1])
        # print('\nangles:\n', self._angles)
        self._aws = np.full(self._angles.shape[0], 4 * np.pi / order[1])

        self._radial_functions = [RadialPolynomial(n) for n in range(order[0] + 1)]
        # self._angular_functions = [
        #     SphericalHarmonic((l, m))
        #     for l in range(order[1] + 1 if basis is None else basis[1])
        #     for m in range(-l, l + 1)
        # ]
        # print('\n\nangular weights:\n', self._aws)
        self._matern = MaternInterpolation(self._angles, weights=self._aws, lmax=lmax)

        self._radial_values = np.array(
            [poly(self._radii) for poly in self._radial_functions]
        )
        self._angular_values = self._matern.chi()

        self._bfs = np.einsum(
            "pr,sa->psra", self._radial_values, self._angular_values
        ).reshape(self.npoints, self.npoints)
        self._wts = (self._rws[:, None] * self._aws[None, :]).ravel()

    def points(self) -> Iterator[NDArray[np.floating]]:
        """Return generator of quadrature points"""
        for pnt in product(self._radii, self._angles):
            r = pnt[0]
            theta, phi = pnt[1]
            yield np.array([r, theta, phi])

    # def points_and_weights(self) -> Iterator[tuple[NDArray[np.floating], np.floating]]:
    #     """Return generator of quadrature points and weights"""
    #     points = product(self._radii, self._angles)
    #     weights = product(self._rws, self._aws)
    #     for pnt, wgt in zip(points, weights):
    #         r = pnt[0]
    #         theta, phi = pnt[1]
    #         yield np.array([r, theta, phi]), wgt[0] * wgt[1]

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
        """Return matrix that transforms `Values` to expansion coefficients"""
        return self._wts * self._bfs

    def inverse(self) -> NDArray[np.floating]:
        """Return matrix that transforms expansion coefficients into `Values`"""
        return self._bfs.T

    def _bfs_r(self):
        """Return basis functions derived with respect to r and evaluated on the points of the pseudo-spectral grid"""
        poly_r = np.array([p(self._radii, True) for p in self._radial_functions])
        return np.einsum("pr,sa->psra", poly_r, self._angular_values).reshape(
            self.npoints, self.npoints
        )

    def _bfs_theta(self):
        """Return basis functions derived with respect to theta and evaluated on the points of the pseudo-spectral grid"""
        return np.einsum(
            "pr,sa->psra", self._radial_values, self._matern.chi_theta()
        ).reshape(self.npoints, self.npoints)

    def _bfs_phi(self):
        """Return basis functions derived with respect to phi and evaluated on the points of the pseudo-spectral grid"""
        return np.einsum(
            "pr,sa->psra", self._radial_values, self._matern.chi_phi()
        ).reshape(self.npoints, self.npoints)

    def partial_r(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the radial component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_r())

    def partial_theta(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the polar component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_theta())

    def partial_phi(self) -> NDArray[np.floating]:
        """Return matrix form of the partial-derivative operator with respect to the azimuthal component"""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_phi())

    def _jacobian_x(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to x of the transformation of coordinates on a point in spherical coordinates"""
        jx = np.zeros(3)
        jx[0] = np.sin(theta) * np.cos(phi)
        jx[1] = np.cos(theta) * np.cos(phi) / xi
        jx[2] = (
            -np.sin(phi) / (xi * np.sin(theta))
            if np.abs(xi * np.sin(theta)) > eps
            else 0.0
        )
        return jx

    def _jacobian_y(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to y of the transformation of coordinates on a point in spherical coordinates"""
        jy = np.zeros(3)
        jy[0] = np.sin(theta) * np.sin(phi)
        jy[1] = np.cos(theta) * np.sin(phi) / xi if xi > eps else 0.0
        jy[2] = (
            np.cos(phi) / (xi * np.sin(theta)) if (xi * np.sin(theta)) > eps else 0.0
        )
        return jy

    def _jacobian_z(self, xi, theta, phi, eps=1e-12):
        """Evaluate the partial derivative with respect to z of the transformation of coordinates on a point in spherical coordinates"""
        jz = np.zeros(3)  # jz[2] = 0
        jz[0] = np.cos(theta)
        jz[1] = -np.sin(theta) / xi if np.abs(xi) > eps else 0.0
        return jz

    def partial_x(self):
        """Return matrix form of the partial-derivative operator with respect to the x component"""
        jxs = self.sample_function(self._jacobian_x)

        term1 = np.einsum("p,ap,bp", self._wts * jxs[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jxs[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jxs[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    def partial_y(self):
        """Return matrix form of the partial-derivative operator with respect to the y component"""
        jys = self.sample_function(self._jacobian_y)

        term1 = np.einsum("p,ap,bp", self._wts * jys[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jys[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jys[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    def partial_z(self):
        """Return matrix form of the partial-derivative operator with respect to the x component"""
        jzs = self.sample_function(self._jacobian_z)

        term1 = np.einsum("p,ap,bp", self._wts * jzs[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jzs[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jzs[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    # @property
    # def size(self) -> int:
    #     """Return number of modes"""
    #     return len(self._radial_functions) * len(self._angular_functions)

    @property
    def npoints(self) -> int:
        """Return number of quadrature points"""
        return len(self._radii) * len(self._angles)

    def sample_function(self, scalar: Function) -> Values:
        """Sample scalar function at the quadrature points"""
        return np.array([scalar(*p) for p in self.points()])

    def expand_scalar_values(self, scalar: Values) -> NDArray[np.floating]:
        """Expand in the pseudospectral basis a scalar function defined by its values on the quadrature point

        Args:
            scalar: Values of the scalar function at the quadrature points

        Returns:
            Expansion coefficients

        """
        return (
            self.transform() @ scalar
        )  # np.einsum("bp,p,p", self._bfs, self._wts, scalar)

    def expand_scalar_function(self, scalar: ScalarFunction) -> NDArray[np.floating]:
        """Expand a scalar function in the pseudospectral basis

        Args:
            scalar: Scalar function to expand

        Returns:
            Expansion coefficients

        """
        return self.expand_scalar_values(self.sample_function(scalar))

    def to_scalar_values(self, coefficients, point_indices=[]):
        """Return the values of the approximated function on the pseudo-spectral grid"""
        return np.einsum("b,bp", coefficients, self._bfs[:, point_indices])
