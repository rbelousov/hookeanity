"""Tests for `hookeanity.interpolation` module"""

import pytest
import numpy as np
import scipy as sp
from hookeanity.interpolation import ScalarFunction, Interpolation


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


def test_transform():
    """Test `Interpolation.transform()` and `Interpolation.inverse()`"""
    inter = Interpolation((5, 5))
    transform = inter.transform()
    inverse = inter.inverse()
    assert transform.shape == (inter.npoints, inter.npoints)
    assert inverse.shape == (inter.npoints, inter.npoints)
    assert inverse @ transform == pytest.approx(np.eye(transform.shape[-1]))


def test_partial_r(scalar, scalar_r):
    """Test `Interpolation.partial_r`"""
    inter = Interpolation((5, 5))
    values = inter.sample_scalar(scalar)
    modes = inter.transform() @ values
    modes_r = inter.partial_r() @ modes
    values_r = inter.inverse() @ modes_r
    expected = inter.sample_scalar(scalar_r)

    precision = 1e-3
    assert values_r == pytest.approx(expected, abs=precision)


def test_partial_theta(scalar, scalar_theta):
    """Test `Interpolation.partial_r`"""
    inter = Interpolation((5, 5))
    values = inter.sample_scalar(scalar)
    modes = inter.transform() @ values
    modes_theta = inter.partial_theta() @ modes
    values_theta = inter.inverse() @ modes_theta
    expected = inter.sample_scalar(scalar_theta)

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(range(inter.npoints), size=100, replace=False)
    for indx in indices:
        val = expected[indx]
        if abs(val) < 1:
            assert values_theta[indx] == pytest.approx(val, abs=5e-1)
        else:
            assert values_theta[indx] == pytest.approx(val, rel=5e-2)
