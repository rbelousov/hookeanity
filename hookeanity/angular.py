"""Numerical tools for solid angles in spherical coordinates."""

import numpy as np
from collections.abc import Iterator
from numpy.typing import NDArray, ArrayLike
from scipy.special import sph_legendre_p
from .base import Real, instance


def fibonacci_points(n: int) -> NDArray[np.floating]:
    """Distribute Fibonacci golden-angle points on a unit sphere avoiding poles.

    Args:
        n: Number of points

    Returns:
        A grid of solid angles that covers the sphere with nearly uniform-distributed points
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


def fibonacci_weight(n: int) -> float:
    """Return weights for the given number of Fibonacci points."""
    return 4 * np.pi / n


def fourier(
    order: int, phi: ArrayLike, derivative: bool = False
) -> NDArray[np.floating] | Real:
    """Evaluate real Fourier function of a given order.

    Args:
        order: Order of the function
        phi: Values of the argument
        derivative: When `True`, evaluate the derivative of the Fourier function

    Returns:
        Value of the Fourier function or its derivative
    """
    phi_is_real = instance(phi, Real)
    if order == 0:
        if phi_is_real:
            return type(phi)(0.0 if derivative else 1.0)
        return (
            np.zeros_like(phi, dtype=float)
            if derivative
            else np.ones_like(phi, dtype=float)
        )

    if not phi_is_real:
        phi = np.asarray(phi)

    prefactor = np.sqrt(2)
    if derivative:
        order = -order
        prefactor *= order
    prefactor *= 1 if order % 2 == 0 else -1
    kernel = (
        np.cos(np.multiply(order, phi))
        if order > 0
        else np.sin(np.multiply(-order, phi))
    )
    return prefactor * kernel


class SphericalHarmonic:
    """Real-valued spherical harmonics.

    Attributes:
        _order (tuple[int, int]): Order (l, m) of the harmonics
    """

    def __init__(self, order: tuple[int, int]):
        """Initialize a spherical harmonic of the given order.

        Args:
            order: Order (l, m) of the harmonic
        """
        assert abs(order[1]) <= order[0], (
            f"Polar order ({order[1]}) exceeds the azimuthatl order ({order[0]})"
        )
        self._order = order

    def __call__(self, angles: ArrayLike) -> NDArray[np.floating] | np.floating:
        """Return values of the spherical harmonic at the given solid angles.

        Args:
            angles: Solid angles to evaluate the spherical harmonic at

        Returns:
            Values of the function
        """
        angles = np.asarray(angles)
        result = sph_legendre_p(
            self._order[0], abs(self._order[1]), angles[..., 0]
        ).reshape(-1)
        result *= fourier(self._order[1], angles[..., 1])
        return result if angles.ndim > 1 else result[0]

    def partial_theta(self, angles: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the polar angle.

        Args:
            angles: Solid angles to evaluate the derivative at

        Returns:
            Values of the function derivative with respect to the polar angle
        """
        angles = np.asarray(angles)
        _, partial = sph_legendre_p(
            self._order[0], abs(self._order[1]), angles[..., 0], diff_n=1
        )
        result = partial.reshape(-1) * fourier(self._order[1], angles[..., 1])
        return result if angles.ndim > 1 else result[0]

    def partial_phi(self, angles: ArrayLike) -> NDArray[np.floating]:
        """Return the derivative with respect to the azimuthal angle.

        Args:
            angles: Solid angles to evaluate the derivative at

        Returns:
            Values of the function derivative with respect to the azimuthal angle
        """
        angles = np.asarray(angles)
        result = sph_legendre_p(
            self._order[0], abs(self._order[1]), angles[..., 0]
        ).reshape(-1) * fourier(self._order[1], angles[..., 1], derivative=True)
        return result if angles.ndim > 1 else result[0]


def geodesic_cos(
    angles_a: ArrayLike, angles_b: ArrayLike
) -> NDArray[np.floating] | np.floating:
    """Compute cosine of the geodesic distance between two arrays of solid angles.

    Args:
        angles_a: First array of angles
        angles_b: Second array of angles

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
    result = np.clip(cos_gamma, -1.0, 1.0)
    return result if angles_a.ndim + angles_b.ndim > 2 else result[0]


class MaternInterpolator:
    """Sobolev-Matérn interpolator.

    Attributes:
        _lmax (int): Maximum order of the spherical harmonic expansion
        _bandfilter (numpy.typing.NDArray[numpy.floating]): Band-limiting filter for the spherical harmonic expansion
        _values (numpy.typing.NDArray[numpy.floating]): Values of the spherical harmonics at the interpolation nodes
        _theta (numpy.typing.NDArray[numpy.floating]): Values of the spherical harmonic derivatives with respect to the polar angle at the interpolation nodes
        _phi (numpy.typing.NDArray[numpy.floating]): Values of the spherical harmonic derivatives with respect to the azimuthal angle at the interpolation nodes
        _weights (numpy.typing.NDArray[numpy.floating]): Quadrature weights for the interpolation nodes
        _coefficients (numpy.typing.NDArray[numpy.floating]): Coefficients of the cardinal functions in the Matérn kernel basis
    """

    def __init__(
        self,
        nodes: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None,
        smoothness: float = 3.0,
        lmax: int | None = None,
    ):
        """Initialize interpolator over the given nodes.

        Smoothness parameter 3 should yield accurate second-order derivatives.

        Args:
            nodes: Nodes of interpolation
            weights: Optional quadrature weights
            smoothness: Smoothness of the interpolation defaults to 3.0
            lmax: Order of kernel expansion defaults to the square root of the number of particles
        """
        sz = len(nodes)
        self._lmax = np.sqrt(sz).astype(int) if lmax is None else lmax
        self._weights = (
            np.ones(len(nodes), dtype=float)
            if weights is None
            else np.array(weights, dtype=float)
        )
        grain = 4 * np.pi / sz
        self._bandfilter = np.array(
            [
                (1 + ell * (ell + 1) * grain) ** -(1 + smoothness)
                for ell in range(self._lmax + 1)
                for m in range(-ell, ell + 1)
            ]
        )

        basis = list(self.basis())
        self._values = np.array([sh(nodes) for sh in basis])
        self._theta = np.array([sh.partial_theta(nodes) for sh in basis])
        self._phi = np.array([sh.partial_phi(nodes) for sh in basis])
        self._coefficients = np.linalg.pinv(
            np.einsum("y,yp,yq", self._bandfilter, self._values, self._values)
        )

    def basis(self) -> Iterator[SphericalHarmonic]:
        """Return generator of basis functions."""
        for ell in range(self._lmax + 1):
            for m in range(-ell, ell + 1):
                yield SphericalHarmonic((ell, m))

    def __getitem__(self, index):
        """Return cardinal function associated with the given point index."""

        def wrapper(angles: ArrayLike) -> NDArray[np.floating] | np.floating:
            if np.asarray(angles).ndim == 1:
                return np.einsum(
                    "y,q,yq,y",
                    self._bandfilter,
                    self._coefficients[index],
                    self._values,
                    [sh(angles) for sh in self.basis()],
                ) / np.sqrt(self._weights[index])

            return np.einsum(
                "y,q,yq,yp",
                self._bandfilter,
                self._coefficients[index],
                self._values,
                [sh(angles) for sh in self.basis()],
            ) / np.sqrt(self._weights[index])

        return wrapper

    def __iter__(self):
        """Return Matern cardinal functions."""
        return (self[i] for i in range(len(self)))

    def __len__(self):
        """Return number of cardinal functions."""
        return self._values.shape[1]

    def chi(
        self, angles: ArrayLike | None = None
    ) -> NDArray[np.floating] | np.floating:
        """Return values of the cardinal functions.

        Args:
            angles: Solid angles to evaluate the cardinal functions at, which default to the interpolation nodes

        Returns:
            Values of the cardinal functions at the given angles
        """
        if angles is None:
            return np.eye(len(self)) / np.sqrt(self._weights)

        angles = np.asarray(angles)
        if angles.ndim == 1:
            return np.einsum(
                "r,rp,y,yp,y->r",
                1 / np.sqrt(self._weights),  # r
                self._coefficients,  # rp
                self._bandfilter,  # y
                self._values,  # yp
                [sh(angles) for sh in self.basis()],
            )

        return np.einsum(
            "r,rp,y,yp,yq->rq",
            1 / np.sqrt(abs(self._weights)),  # r
            self._coefficients,  # rp
            self._bandfilter,  # y
            self._values,  # yp
            [sh(angles) for sh in self.basis()],
        )

    def chi_theta(
        self, angles: ArrayLike | None = None
    ) -> NDArray[np.floating] | np.floating:
        """Return derivatives of the cardinal functions with respect to the polar angle.

        Args:
            angles: Solid angles to evaluate the derivatives at, which default to the interpolation nodes

        Returns:
            Derivatives of the cardinal functions with respect to the polar angle at the given angles
        """
        if angles is None:
            return np.einsum(
                "r,rp,y,yp,yq->rq",
                1 / np.sqrt(abs(self._weights)),  # r
                self._coefficients,  # rp
                self._bandfilter,  # y
                self._values,  # yp
                self._theta,  # yq
            )

        angles = np.asarray(angles)
        if angles.ndim == 1:
            return np.einsum(
                "r,rp,y,yp,y->r",
                1 / np.sqrt(self._weights),  # r
                self._coefficients,  # rp
                self._bandfilter,  # y
                self._values,  # yp
                [sh.partial_theta(angles) for sh in self.basis()],
            )

        return np.einsum(
            "r,rp,y,yp,yq->rq",
            1 / np.sqrt(abs(self._weights)),  # r
            self._coefficients,  # rp
            self._bandfilter,  # y
            self._values,  # yp
            [sh.partial_theta(angles) for sh in self.basis()],
        )

    def chi_phi(
        self, angles: ArrayLike | None = None
    ) -> NDArray[np.floating] | np.floating:
        """Return derivatives of the cardinal functions with respect to the azimuth.

        Args:
            angles: Solid angles to evaluate the derivatives at, which default to the interpolation nodes

        Returns:
            Derivatives of the cardinal functions with respect to the azimuthal angle at the given angles
        """
        if angles is None:
            return np.einsum(
                "r,rp,y,yp,yq->rq",
                1 / np.sqrt(abs(self._weights)),  # r
                self._coefficients,  # rp
                self._bandfilter,  # y
                self._values,  # yp
                self._phi,  # yq
            )

        angles = np.asarray(angles)
        if angles.ndim == 1:
            return np.einsum(
                "r,rp,y,yp,y->r",
                1 / np.sqrt(self._weights),  # r
                self._coefficients,  # rp
                self._bandfilter,  # y
                self._values,  # yp
                [sh.partial_phi(angles) for sh in self.basis()],
            )

        return np.einsum(
            "r,rp,y,yp,yq->rq",
            1 / np.sqrt(abs(self._weights)),  # r
            self._coefficients,  # rp
            self._bandfilter,  # y
            self._values,  # yp
            [sh.partial_phi(angles) for sh in self.basis()],
        )

    def partial_theta(self):
        """Return matrix form of the derivative operator with respect to theta."""
        return np.einsum(
            "q,pq,rq",
            self._weights,  # q,
            self.chi(),  # pq
            self.chi_theta(),  # rq
        )

    def partial_phi(self):
        """Return matrix form of the derivative operator with respect the phi."""
        return np.einsum(
            "q,pq,rq",
            self._weights,  # q,
            self.chi(),  # pq
            self.chi_phi(),  # rq
        )
