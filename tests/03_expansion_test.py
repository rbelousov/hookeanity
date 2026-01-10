"""Test `hookeanity.expansion` model"""

import pytest
import numpy as np
import scipy as sp
from hookeanity.expansion import Expansion, ScalarFunction
from hookeanity.radial import RadialPolynomial
from hookeanity.angular import SphericalHarmonic


def test_generators():
    """Test generator methods of `Expansion`"""
    expansion = Expansion((3, 3))

    counter = 0
    for ps, _ in expansion.points_and_weights():
        counter += 1
        assert ps.shape == (3,)
    assert counter == len(expansion._radii) * len(expansion._angles)
    assert counter == 4 * 26

    for bf in expansion.basis_functions():
        assert np.isfinite(bf(0.5, 0.1, 0.7))


@pytest.fixture
def scalar() -> ScalarFunction:
    """Fixture of a test scalar function"""

    def wrapper(r, theta, phi):
        return sp.special.j0(r * np.cos(theta) * np.sin(phi))

    return wrapper


@pytest.fixture
def scalar_r() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the radial component"""

    def wrapper(r, theta, phi):
        angular_factor = np.cos(theta) * np.sin(phi)
        return -sp.special.j1(r * angular_factor) * angular_factor

    return wrapper


@pytest.fixture
def scalar_theta() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the azimuthal component"""

    def wrapper(r, theta, phi):
        return (
            sp.special.j1(r * np.cos(theta) * np.sin(phi)) * np.sin(theta) * np.sin(phi)
        )

    return wrapper


@pytest.fixture
def scalar_phi() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the azimuthal component"""

    def wrapper(r, theta, phi):
        return (
            -sp.special.j1(r * np.cos(theta) * np.sin(phi))
            * np.cos(theta)
            * np.cos(phi)
        )

    return wrapper


def test_expand_scalar(scalar):
    """Test `Expansion.expand_scalar"""
    expansion = Expansion((3, 3))
    assert len(expansion._radial_functions) == 4
    assert len(expansion._angular_functions) == (3 + 1) ** 2
    assert expansion.size == 4 * 16

    bf = expansion.basis_function(RadialPolynomial(2), SphericalHarmonic((3, -1)))
    idx = 2 * 16 + 16 - 5
    cfs = expansion.expand_scalar_function(bf)
    fnc = expansion.to_scalar_function(cfs)
    assert len(cfs) == expansion.size
    assert np.sum(cfs) == pytest.approx(1)
    assert cfs[idx] == pytest.approx(1)
    assert np.delete(cfs, idx) == pytest.approx(np.zeros(expansion.size - 1))
    for n, point in enumerate(expansion.points()):
        assert fnc(*point) == pytest.approx(bf(*point), rel=1e-2), (
            f"Error at the point {n}/{expansion.npoints}"
        )

    cfs = expansion.expand_scalar_function(scalar)
    fnc = expansion.to_scalar_function(cfs)
    for n, point in enumerate(expansion.points()):
        assert fnc(*point) == pytest.approx(scalar(*point), rel=10e-2), (
            f"Error at the point {n}/{expansion.npoints}"
        )


def test_derivatives(scalar, scalar_r, scalar_theta, scalar_phi):
    """Test spectral derivatives"""
    expansion = Expansion((10, 10), basis=(5, 5))
    cfs = expansion.expand_scalar_function(scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    points = rng.choice(list(expansion.points()), size=npoints, replace=False)

    oper = expansion.partial_r()
    cfs_r = np.dot(oper, cfs)
    f_r = expansion.to_scalar_function(cfs_r)
    for n, point in enumerate(points):
        assert f_r(*point) == pytest.approx(scalar_r(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

    oper_theta = expansion.partial_theta()
    cfs_theta = np.dot(oper_theta, cfs)
    f_theta = expansion.to_scalar_function(cfs_theta)
    for n, point in enumerate(points):
        assert f_theta(*point) == pytest.approx(scalar_theta(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

    oper_phi = expansion.partial_phi()
    cfs_phi = np.dot(oper_phi, cfs)
    f_phi = expansion.to_scalar_function(cfs_phi)
    for n, point in enumerate(points):
        assert f_phi(*point) == pytest.approx(scalar_phi(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )
