"""Numerical methods for spherical coordinates over a unit ball."""

from itertools import product
from collections.abc import Iterator
import numpy as np
from numpy.typing import NDArray
from .base import DIM, Point, Grid, VectorValues, instance
from .radial import RadialPolynomial
from .angular import (
    fibonacci_points,
    fibonacci_weight,
    gauss_legendre_fourier_points,
    gauss_legendre_fourier_weights,
    MaternInterpolator,
)


class Interpolation:
    """Expansion of a scalar-valued function in spherical coordinates.

    Attributes:
        _resolution (tuple[int, int]): Number of radial points and angular resolution parameter
        _radii (numpy.NDArray[numpy.floating]): Radial quadrature points
        _rws (numpy.NDArray[numpy.floating]): Radial quadrature weights
        _angles (numpy.NDArray[numpy.floating]): Angular quadrature points (theta, phi)
        _aws (numpy.NDArray[numpy.floating]): Angular quadrature weights
        _radial_functions (numpy.NDArray[numpy.floating]): List of radial basis functions
        _angular_functions (numpy.NDArray[numpy.floating]): List of angular basis functions
        _radial_values (numpy.NDArray[numpy.floating]): Radial basis functions evaluated at the radial quadrature points
        _angular_values (numpy.NDArray[numpy.floating]): Angular basis functions evaluated at the angular quadrature points
        _bfs (numpy.NDArray[numpy.floating]): Basis functions evaluated at the quadrature points
        _wts (numpy.NDArray[numpy.floating]): Combined quadrature weights for the spherical grid
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        lmax: int | None = None,
        angular_grid: str | str = "fibonacci",
    ):
        """Initialize interpolation on the given resolution.

        Args:
            resolution: Tuple (nr, nang), where nr is the number of radial points and nang is an angular resolution parameter such that
                for 'fibonacci', it corresponds to the number of angular points
                for 'gauss-legendre-fourier', it corresponds to the number of latitudinal nodes
            lmax: Maximum order of the spherical harmonic expansion
            angular_grid: Type of discretization in the angular component
        """
        self._radii, self._rws = RadialPolynomial.quadrature(resolution[0] - 1)

        self._resolution = resolution
        if angular_grid == "fibonacci":
            self._angles = fibonacci_points(resolution[1])
            self._aws = np.full(resolution[1], fibonacci_weight(resolution[1]))
        elif angular_grid == "gauss-legendre-fourier":
            self._angles = gauss_legendre_fourier_points(resolution[1])
            self._aws = gauss_legendre_fourier_weights(resolution[1])
        else:
            raise ValueError(f"Unknown angular_grid: {angular_grid}")

        self._radial_functions = [RadialPolynomial(n) for n in range(resolution[0])]
        self._matern = MaternInterpolator(self._angles, weights=self._aws, lmax=lmax)
        self._angular_functions = list(self._matern)

        self._radial_values = np.array(
            [poly(self._radii) for poly in self._radial_functions]
        )
        self._angular_values = self._matern.chi()

        sz = len(self)
        self._bfs = np.einsum(
            "pr,ca->pcra", self._radial_values, self._angular_values
        ).reshape(sz, sz)
        self._wts = (self._rws[:, None] * self._aws[None, :]).ravel()

    def points(self) -> Iterator[Point]:
        """Return generator of quadrature points."""
        for pnt in product(self._radii, self._angles):
            r = pnt[0]
            theta, phi = pnt[1]
            yield np.array([r, theta, phi])

    def grid(self) -> Grid:
        """Return array of quadrature points."""
        return np.array(list(self.points()))

    def transform(self) -> NDArray[np.floating]:
        """Return transformation matrix to the normal modes."""
        return self._wts * self._bfs

    def inverse(self) -> NDArray[np.floating]:
        """Return transformation matrix from the normal modes."""
        return self._bfs.T

    def _bfs_r(self):
        """Return radial derivatives of the basis functions."""
        poly_r = np.array([p(self._radii, True) for p in self._radial_functions])
        sz = len(self)
        return np.einsum("pr,ca->pcra", poly_r, self._angular_values).reshape(sz, sz)

    def _bfs_theta(self):
        """Return polar derivatives of the basis functions."""
        sz = len(self)
        return np.einsum(
            "pr,ca->pcra", self._radial_values, self._matern.chi_theta()
        ).reshape(sz, sz)

    def _bfs_phi(self):
        """Return polar derivatives of the basis functions."""
        sz = len(self)
        return np.einsum(
            "pr,ca->pcra", self._radial_values, self._matern.chi_phi()
        ).reshape(sz, sz)

    def partial_r(self) -> NDArray[np.floating]:
        """Return derivative operator with respect to the radial component."""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_r())

    def partial_theta(self) -> NDArray[np.floating]:
        """Return derivative operator with respect to the polar component."""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_theta())

    def partial_phi(self) -> NDArray[np.floating]:
        """Return azimuthal derivative operator."""
        return np.einsum("p,ap,bp", self._wts, self._bfs, self._bfs_phi())

    def _jacobian_x(self, arg: Point | Grid) -> Point | VectorValues:
        """Return the derivative of spherical coordinates with respect to x.

        This method satisfies the signature of `VectorField`.

        Args:
            arg: Points to evaluate the Jacobian

        Returns:
            First column of the Jacobian matrix at the given point in spherical coordinates
        """
        if instance(arg, Point):
            jx = np.zeros(DIM)
            xi, theta, phi = arg
        else:
            jx = np.zeros((len(arg), DIM))
            xi, theta, phi = arg.T
        jx[..., 0] = np.sin(theta) * np.cos(phi)
        jx[..., 1] = np.cos(theta) * np.cos(phi) / xi
        jx[..., 2] = -np.sin(phi) / (xi * np.sin(theta))
        return jx

    def _jacobian_y(self, arg: Point | Grid) -> Point | VectorValues:
        """Return the derivative of spherical coordinates with respect to y.

        This method satisfies the signature of `VectorField`.

        Args:
            arg: Points to evaluate the Jacobian

        Returns:
            Second column of the Jacobian matrix at the given point in spherical coordinates
        """
        if instance(arg, Point):
            jy = np.zeros(DIM)
            xi, theta, phi = arg
        else:
            jy = np.zeros((len(arg), DIM))
            xi, theta, phi = arg.T
        jy[..., 0] = np.sin(theta) * np.sin(phi)
        jy[..., 1] = np.cos(theta) * np.sin(phi) / xi  # if xi > eps else 0.0
        tmp = xi * np.sin(theta)
        jy[..., 2] = np.cos(phi) / tmp  # if tmp > eps else 0.0
        return jy

    def _jacobian_z(self, arg: Point | Grid) -> Point | VectorValues:
        """Return the derivative of spherical coordinates with respect to z.

        This method satisfies the signature of `VectorField`.

        Args:
            arg: Points to evaluate the Jacobian

        Returns:
            Third column of the Jacobian matrix at the given point in spherical coordinates
        """
        if instance(arg, Point):
            jz = np.zeros(DIM)
            xi, theta, phi = arg
        else:
            jz = np.zeros((len(arg), DIM))
            xi, theta, phi = arg.T
        jz[..., 0] = np.cos(theta)
        jz[..., 1] = -np.sin(theta) / xi  # if np.abs(xi) > eps else 0.0
        return jz

    def partial_x(self):
        """Return derivative operator with respect to the Cartesian x component."""
        jxs = self._jacobian_x(self.grid())

        term1 = np.einsum("p,ap,bp", self._wts * jxs[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jxs[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jxs[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    def partial_y(self):
        """Return derivative operator with respect to the Cartesian y component."""
        jys = self._jacobian_y(self.grid())

        term1 = np.einsum("p,ap,bp", self._wts * jys[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jys[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jys[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    def partial_z(self):
        """Return derivative operator with respect to the Cartesian z component."""
        jzs = self._jacobian_z(self.grid())

        term1 = np.einsum("p,ap,bp", self._wts * jzs[:, 0], self._bfs, self._bfs_r())
        term2 = np.einsum(
            "p,ap,bp", self._wts * jzs[:, 1], self._bfs, self._bfs_theta()
        )
        term3 = np.einsum("p,ap,bp", self._wts * jzs[:, 2], self._bfs, self._bfs_phi())

        return term1 + term2 + term3

    def __len__(self) -> int:
        """Return number of quadrature points or basis functions."""
        return self.nradii * self.nangles

    @property
    def nradii(self) -> int:
        """Return number of radial quadrature points."""
        return len(self._radii)

    @property
    def nangles(self) -> int:
        """Return number of angular quadrature points."""
        return len(self._angles)

    @property
    def mesh(self) -> tuple[int, int]:
        """Return mesh specification of the interpolation grid."""
        return self.nradii, self.nangles

    @property
    def resolution(self) -> tuple[int, int]:
        """Return number of radial points and angular resolution parameter."""
        return self._resolution

    @property
    def lmax(self) -> int:
        """Return maximum spherical-harmonic expansion order."""
        return self._matern._lmax
