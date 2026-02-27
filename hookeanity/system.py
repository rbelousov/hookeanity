"""Numerical methods for integration over time"""

import numpy as np
from .interpolation import Interpolation
from typing import Self
from .base import Real, VectorValues


def to_cartesian(xi, theta, phi):
    """Return cartesian coordinates, given a point in spherical coordinates"""
    x = xi * np.sin(theta) * np.cos(phi)
    y = xi * np.sin(theta) * np.sin(phi)
    z = xi * np.cos(theta)
    return (x, y, z)


class System:
    """Elastodynamic system"""

    def __init__(
        self,
        state: VectorValues,
        lam: Real,
        mu: Real,
        f_ext: Real,
        order: tuple[int, int],
        lmax: int | None = None,
        kbt: Real = 1.0,
        gamma: Real = 1.0,
        radius: Real = 1.0,
    ):
        """Initialize the system with physical parameters for a given order of approximation

        Args:
            state: Current sampled state of the system
            lam: Lamé's first parameter
            mu: Lamé's second parameter (Shear modulus)
            f_ext: Value of the external force applied at the poles
            order: Order of the spectral accuracy
            lmax: Order of kernel expansion defaults to the square root of the number of particles
            kbt: unit of energy
            gamma: unit of friction
            radius: unit of length

        """
        self._kbt = kbt
        self._gamma = gamma
        self._radius = radius
        self._tau = self._gamma / self._kbt

        self._state = state
        self._lam = lam
        self._mu = mu
        self._f_ext = f_ext

        self._inter = Interpolation(order, lmax=lmax)
        self._points = list(self._inter.points())
        assert self._state.shape == (3, self.npoints)

        # self._p_surface = self._inter.radial_values[:, -1]  # Indexing assumption, the function can be directly evaluated
        self._radial_surface, self._radial_on_points = self._evaluate_radial()

        self.rng = np.random.default_rng()

        self._inverse_lhs = np.linalg.inv(self._lhs())

    @property
    def points(self):
        """Return grid points"""
        return self._points

    @property
    def npoints(self):
        """Return number of grid points"""
        return len(self._points)

    def _evaluate_radial(self):
        radii_points = np.asarray([p[0] for p in self._points])
        radial_surface, radial_on_points = zip(
            *[(poly(1.0), poly(radii_points)) for poly in self._inter.radial_functions]
        )
        return np.asarray(radial_surface), np.asarray(radial_on_points)

    def _lhs(self):
        wts, _, _ = self._inter.weights
        return self._gamma * np.einsum(
            "p,ap,s,sp->ap",
            wts,
            self._inter.bfs,
            self._radial_surface,
            self._radial_on_points,
        )

    def _elastic_stress(self):
        partial = np.asarray(
            [self._inter.partial_x(), self._inter.partial_y(), self._inter.partial_z()]
        )
        laplacian = np.asarray([oper @ oper for oper in partial])

        transform = self._inter.transform()
        state_modes = np.asarray([transform @ s for s in self._state])

        # divergence = sum(partial[i] @ state_modes[i] for i in range(3))
        divergence = np.einsum("abc,ac", partial, state_modes)

        term1 = np.asarray(
            [(self._lam + self._mu) * (oper @ divergence) for oper in partial]
        )
        term2 = np.asarray(
            [self._mu * np.einsum("abc,c->b", laplacian, s) for s in state_modes]
        )

        return term1 + term2

    def _external_force(self):
        chi_north_pole = np.asarray(
            [chi([0.0, 0.0]) for chi in self._inter.angular_functions]
        )

        return (
            self._f_ext
            * (self._radial_surface[:, None] * chi_north_pole[None, :]).ravel()
        )

    def _fluctuations(self, dt):
        normal = self.rng.normal(0.0, 1.0, size=(3, self._inter.nangles))

        return np.asarray(
            [
                np.sqrt(2 * self._kbt * self._gamma * dt)
                * (self._radial_surface[:, None] * nor[None, :]).ravel()
                for nor in normal
            ]
        )

    def __call__(self, dt: Real) -> Self:
        """Euler-Maruyama integration step

        Args:
            dt: integration time step

        """
        elastic_stress = self._elastic_stress()
        external_force = self._external_force()
        fluct = self._fluctuations(dt)

        dim = self._state.shape[0]  # presumably 3: x,y,z
        for a in range(dim):
            for b in range(dim):
                rhs = fluct[b] + dt * elastic_stress[b]
                if a == dim - 1:  # if a is the last component (z-component)
                    rhs += external_force
                self._state[a] += self._inverse_lhs @ rhs
        return self
